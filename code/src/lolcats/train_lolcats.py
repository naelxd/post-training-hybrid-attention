#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoLCATs Stage-1 training (attention transfer) for Qwen3.

Loads the model, applies the LoLCATs hybrid patch (full attention on retrieval
heads from compressible_heads.pt, SW + GLA hybrid on the rest), then trains
ONLY the new modules (phi_q, phi_k, g_proj, mix_logit) so that the patched
attention output matches the original softmax attention output (per-layer MSE).

Training corpus: Paul Graham essays tokenised and chunked to ``TRAIN_SEQ_LEN``.
Random sequences also supported (set ``USE_RANDOM_CORPUS = True``) but give a
much weaker signal — feature maps end up biased toward random-token statistics.

Saves the trained LoLCATs weights to ``WEIGHTS_OUTPUT`` for later use by the
inference / NIAH script.
"""

import gc
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_lolcats import Qwen3LoLCATsPatcher


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_PATH = "Qwen/Qwen3-4B"
MODEL_DTYPE = torch.bfloat16
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoLCATs hyperparameters (must match niah_lolcats.py at load time)
WINDOW_SIZE = 512
GATE_INIT_BIAS = 5.0
MIX_INIT_LOGIT = 5.0

# Training
TRAIN_STEPS = 500
TRAIN_SEQ_LEN = 256
TRAIN_BATCH_SIZE = 1
TRAIN_LR = 3e-4         # AdamW lr for feature maps + g_proj + mix_logit
WARMUP_STEPS = 50       # linear warmup from 0 to TRAIN_LR
GRAD_CLIP = 1.0
LOG_EVERY = 25

# Corpus
USE_RANDOM_CORPUS = False
ESSAYS_DIR = "LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays"

# Files
COMPRESSIBLE_HEADS_PATH = "compressible_heads_0.pt"
WEIGHTS_OUTPUT = "lolcats_weights.pt"

# Random seed
SEED = 42


# =============================================================================
# Text corpus → batches
# =============================================================================

def build_text_batches(
    tokenizer, essays_dir: str, seq_len: int, batch_size: int,
    device: str,
) -> Iterable[torch.Tensor]:
    """
    Tokenise every essay, concatenate, chunk to ``seq_len`` non-overlapping
    blocks, then yield random batches of these blocks forever.
    """
    path = Path(essays_dir)
    if not path.exists():
        raise FileNotFoundError(f"Essays directory not found: {essays_dir}")
    txt_files = sorted(path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {essays_dir}")

    print(f"Tokenising {len(txt_files)} essays...")
    all_ids: List[int] = []
    for f in txt_files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id
                       if tokenizer.eos_token_id is not None else 0)

    tokens = torch.tensor(all_ids, dtype=torch.long)
    n_chunks = tokens.numel() // seq_len
    if n_chunks == 0:
        raise RuntimeError(
            f"Corpus too small ({tokens.numel()} tokens) for seq_len={seq_len}"
        )
    chunks = tokens[: n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  {tokens.numel():,} tokens → {n_chunks} chunks of {seq_len}")

    def gen():
        while True:
            idx = torch.randint(0, n_chunks, (batch_size,))
            yield chunks[idx].to(device)

    return gen()


def build_random_batches(vocab_size: int, seq_len: int, batch_size: int,
                        device: str) -> Iterable[torch.Tensor]:
    def gen():
        while True:
            yield torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return gen()


# =============================================================================
# compressible_heads loader
# =============================================================================

def load_keep_heads(filepath: str) -> Dict[int, List[int]]:
    data = torch.load(filepath, weights_only=False, map_location="cpu")
    induction = data.get("prefix_matching", {})
    echo = data.get("copying", {})
    keep: Dict[int, List[int]] = {}
    for layer_idx, heads in induction.items():
        keep[layer_idx] = keep.get(layer_idx, []) + heads
    for layer_idx, heads in echo.items():
        keep[layer_idx] = keep.get(layer_idx, []) + heads
    return keep


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 80)
    print("LoLCATs Stage-1 attention-transfer training")
    print("=" * 80)
    print(f"Model:           {MODEL_PATH}")
    print(f"Window:          {WINDOW_SIZE}")
    print(f"Gate init bias:  {GATE_INIT_BIAS}")
    print(f"Mix init logit:  {MIX_INIT_LOGIT}")
    print(f"Steps:           {TRAIN_STEPS}")
    print(f"Seq len:         {TRAIN_SEQ_LEN}")
    print(f"Batch size:      {TRAIN_BATCH_SIZE}")
    print(f"Learning rate:   {TRAIN_LR}")
    print(f"Corpus:          {'random' if USE_RANDOM_CORPUS else ESSAYS_DIR}")
    print(f"Heads file:      {COMPRESSIBLE_HEADS_PATH}")
    print(f"Output:          {WEIGHTS_OUTPUT}")
    print("=" * 80)

    print(f"\nLoading tokenizer + model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=MODEL_DTYPE,
        device_map=MODEL_DEVICE,
        attn_implementation="flash_attention_2",
    )

    # Patch
    patcher = Qwen3LoLCATsPatcher(
        model,
        window_size=WINDOW_SIZE,
        gate_init_bias=GATE_INIT_BIAS,
        mix_init_logit=MIX_INIT_LOGIT,
    )

    if os.path.exists(COMPRESSIBLE_HEADS_PATH):
        keep_heads = patcher.load_compressible_heads(COMPRESSIBLE_HEADS_PATH)
    else:
        print(f"  {COMPRESSIBLE_HEADS_PATH} not found — using all heads as "
              f"retrieval (hybrid degenerates to full attention).")
        num_layers = len(model.model.layers)
        num_heads = model.config.num_attention_heads
        keep_heads = {l: list(range(num_heads)) for l in range(num_layers)}

    patcher.patch_model(keep_heads)

    # Corpus
    if USE_RANDOM_CORPUS:
        batches = build_random_batches(
            model.config.vocab_size, TRAIN_SEQ_LEN, TRAIN_BATCH_SIZE, MODEL_DEVICE,
        )
    else:
        batches = build_text_batches(
            tokenizer, ESSAYS_DIR, TRAIN_SEQ_LEN, TRAIN_BATCH_SIZE, MODEL_DEVICE,
        )

    # Train
    torch.cuda.empty_cache()
    gc.collect()
    patcher.attention_transfer_train(
        batches,
        steps=TRAIN_STEPS,
        lr=TRAIN_LR,
        warmup_steps=WARMUP_STEPS,
        log_every=LOG_EVERY,
        grad_clip=GRAD_CLIP,
    )

    # Save
    patcher.save_weights(WEIGHTS_OUTPUT, keep_heads=keep_heads)

    print("\n" + "=" * 80)
    print("Done. To use these weights for evaluation, set in niah_lolcats.py:")
    print(f"    WEIGHTS_PATH = \"{WEIGHTS_OUTPUT}\"")
    print("=" * 80)


if __name__ == "__main__":
    main()
