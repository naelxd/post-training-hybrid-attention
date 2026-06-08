#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perplexity benchmark across the four attention setups you've built:

  1. Original Qwen3        — full softmax (quality ceiling)
  2. Selective SW          — qwen3_sliding_window.py, 15% retrieval heads
  3. Partial LoLCATs       — qwen3_lolcats.py + lolcats_weights.pt
                             (15% retrieval heads + SW+linear on the rest)
  4. Full LoLCATs          — qwen3_lolcats.py with empty keep_heads
                             + lolcats_full_weights.pt (all heads hybrid)

Perplexity is computed at multiple sequence lengths (default 512 / 2048 / 8192)
to show how each method handles short vs long context.  Lower is better.

Eval corpus: WikiText-2 test split (out-of-distribution for all setups since
none were trained on it).  Falls back to a held-out slice of Paul Graham
essays if `datasets` isn't installed or no internet.

Models are loaded one at a time and freed before the next to stay within
24 GB VRAM.
"""

import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "Qwen/Qwen3-4B"
MODEL_DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Eval set
USE_WIKITEXT = True             # Set False to use the PG fallback only
PG_ESSAYS_DIR = "LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays"
PG_HOLDOUT_FRACTION = 0.2       # Last 20% of PG essays used as eval (rough holdout)

# Eval seq lengths.  Use a small set — perplexity changes smoothly with seq_len
# so 3-4 points are enough to see the trend.
EVAL_SEQ_LENS = [512, 2048, 8192]
MAX_CHUNKS_PER_SEQLEN = 32      # 32 chunks × 8192 tokens ≈ 260k tokens evaluated
BATCH_SIZE = 1

# Files
COMPRESSIBLE_HEADS_PATH = "compressible_heads_0.pt"
LOLCATS_PARTIAL_WEIGHTS = "lolcats_weights.pt"
LOLCATS_FULL_WEIGHTS = "lolcats_full_weights.pt"
OUTPUT_FILE = "perplexity_results.json"

# LoLCATs hyperparameters — must match those used in train_lolcats[_full].py
LC_WINDOW_SIZE = 512
LC_GATE_INIT_BIAS = 0.0
LC_MIX_INIT_LOGIT = 5.0

# SW hyperparameters — must match niah.py / qwen3_sliding_window.py
SW_WINDOW_SIZE = 512
SW_NUM_SINKS = 4

# Setups to evaluate.  Comment out any you don't have weights for.
RUN_SETUPS = [
    "original",
    "selective_sw",
    "lolcats_partial",
    "lolcats_full",
]


# =============================================================================
# Eval corpus
# =============================================================================

def load_eval_text() -> Tuple[str, str]:
    """Returns (text, source_name)."""
    if USE_WIKITEXT:
        try:
            from datasets import load_dataset
            print("Loading WikiText-2 test split...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(d["text"] for d in ds if d["text"].strip())
            print(f"  WikiText-2: {len(text):,} chars")
            return text, "wikitext-2"
        except Exception as e:
            print(f"  Failed to load WikiText ({e}), falling back to PG holdout.")

    # Fallback: last fraction of PG essays
    path = Path(PG_ESSAYS_DIR)
    if not path.exists():
        raise FileNotFoundError(f"No eval corpus: install `datasets` or "
                                f"point PG_ESSAYS_DIR at a valid path.")
    txt_files = sorted(path.glob("*.txt"))
    n_holdout = max(1, int(len(txt_files) * PG_HOLDOUT_FRACTION))
    holdout_files = txt_files[-n_holdout:]
    print(f"PG holdout: last {n_holdout}/{len(txt_files)} essays "
          f"({[f.name for f in holdout_files[:3]]}...)")
    text = "\n\n".join(f.read_text(encoding="utf-8").strip()
                       for f in holdout_files)
    print(f"  PG holdout text: {len(text):,} chars")
    return text, "pg-holdout"


def tokenize_and_chunk(
    tokenizer, text: str, seq_len: int, max_chunks: int,
) -> torch.Tensor:
    """Returns [n_chunks, seq_len] LongTensor on CPU."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids) // seq_len
    n = min(n, max_chunks)
    if n == 0:
        raise RuntimeError(f"Corpus too small ({len(ids)} tokens) for "
                           f"seq_len={seq_len}")
    chunks = torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)
    return chunks


# =============================================================================
# Perplexity computation
# =============================================================================

@torch.no_grad()
def compute_perplexity(
    model, chunks: torch.Tensor, batch_size: int = 1,
) -> Tuple[float, float, float]:
    """
    Returns (perplexity, avg_loss_nats, tokens_per_sec).

    Causal-LM perplexity: shift labels by one and take cross-entropy on the
    full sequence at once (no rolling window).  This measures how well the
    model predicts each next token given the prefix — straightforward apples-
    to-apples between architectures.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size].to(model.device)
        logits = model(batch, use_cache=False).logits     # [B, T, V]
        # Shifted cross-entropy.  Note: this counts each non-first token once,
        # so total_tokens = B*(T-1) per batch.
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = batch[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
        # Free as we go
        del logits, shift_logits, shift_labels, loss, batch

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    tokens_per_sec = total_tokens / max(elapsed, 1e-6)
    return ppl, avg_loss, tokens_per_sec


# =============================================================================
# Setup loaders — each returns a patched model ready for eval
# =============================================================================

def _fresh_model() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=MODEL_DTYPE,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )


def setup_original():
    print("\n[Setup] Original Qwen3 (full softmax)")
    return _fresh_model()


def setup_selective_sw():
    print("\n[Setup] Selective Sliding Window (qwen3_sliding_window.py)")
    from qwen3_sliding_window import Qwen3SelectiveSlidingWindowPatcher
    model = _fresh_model()
    patcher = Qwen3SelectiveSlidingWindowPatcher(
        model, window_size=SW_WINDOW_SIZE, num_sinks=SW_NUM_SINKS,
    )
    keep = patcher.load_compressible_heads(COMPRESSIBLE_HEADS_PATH)
    patcher.patch_model(keep)
    return model


def setup_lolcats_partial():
    print("\n[Setup] Partial LoLCATs (selective retrieval + SW+GLA on rest)")
    from qwen3_lolcats import Qwen3LoLCATsPatcher
    model = _fresh_model()
    patcher = Qwen3LoLCATsPatcher(
        model,
        window_size=LC_WINDOW_SIZE,
        gate_init_bias=LC_GATE_INIT_BIAS,
        mix_init_logit=LC_MIX_INIT_LOGIT,
    )
    keep = patcher.load_compressible_heads(COMPRESSIBLE_HEADS_PATH)
    patcher.patch_model(keep)
    if Path(LOLCATS_PARTIAL_WEIGHTS).exists():
        patcher.load_weights(LOLCATS_PARTIAL_WEIGHTS)
    else:
        print(f"  Warning: {LOLCATS_PARTIAL_WEIGHTS} not found — using init params")
    return model


def setup_lolcats_full():
    print("\n[Setup] Full LoLCATs (all heads hybrid, no retrieval)")
    from qwen3_lolcats import Qwen3LoLCATsPatcher
    model = _fresh_model()
    patcher = Qwen3LoLCATsPatcher(
        model,
        window_size=LC_WINDOW_SIZE,
        gate_init_bias=LC_GATE_INIT_BIAS,
        mix_init_logit=LC_MIX_INIT_LOGIT,
    )
    patcher.patch_model({})    # empty → all hybrid
    if Path(LOLCATS_FULL_WEIGHTS).exists():
        patcher.load_weights(LOLCATS_FULL_WEIGHTS)
    else:
        print(f"  Warning: {LOLCATS_FULL_WEIGHTS} not found — using init params")
    return model


SETUP_FNS = {
    "original":         ("Original Qwen3",          setup_original),
    "selective_sw":     ("Selective SW (15%)",      setup_selective_sw),
    "lolcats_partial":  ("Partial LoLCATs (15%)",   setup_lolcats_partial),
    "lolcats_full":     ("Full LoLCATs (0%)",       setup_lolcats_full),
}


# =============================================================================
# Main
# =============================================================================

def _free(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    print("=" * 80)
    print("Perplexity benchmark")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Seq lengths: {EVAL_SEQ_LENS}")
    print(f"Max chunks per seq_len: {MAX_CHUNKS_PER_SEQLEN}")
    print(f"Setups: {RUN_SETUPS}")
    print("=" * 80)

    print("\nLoading tokenizer + eval corpus...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    text, source = load_eval_text()

    # Pre-tokenise & chunk for every seq_len once — saves redoing per setup.
    chunks_by_seqlen: Dict[int, torch.Tensor] = {}
    for sl in EVAL_SEQ_LENS:
        chunks_by_seqlen[sl] = tokenize_and_chunk(
            tokenizer, text, sl, MAX_CHUNKS_PER_SEQLEN,
        )
        print(f"  seq_len={sl}: {chunks_by_seqlen[sl].shape[0]} chunks")

    # Run all setups
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for key in RUN_SETUPS:
        if key not in SETUP_FNS:
            print(f"\nSkipping unknown setup '{key}'")
            continue
        label, loader = SETUP_FNS[key]

        try:
            model = loader()
        except Exception as e:
            print(f"  Failed to set up {label}: {e}")
            continue

        per_seq: Dict[int, Dict[str, float]] = {}
        for sl in EVAL_SEQ_LENS:
            chunks = chunks_by_seqlen[sl]
            print(f"\n  Evaluating {label} @ seq_len={sl} "
                  f"({chunks.shape[0]} chunks)...")
            try:
                ppl, loss, tps = compute_perplexity(model, chunks, BATCH_SIZE)
                per_seq[sl] = {
                    "perplexity": ppl,
                    "loss_nats": loss,
                    "tokens_per_sec": tps,
                    "n_chunks": chunks.shape[0],
                }
                print(f"    ppl = {ppl:8.3f}   loss = {loss:.4f} nats   "
                      f"{tps:6.0f} tok/s")
            except torch.cuda.OutOfMemoryError as e:
                print(f"    OOM at seq_len={sl}: {e}")
                per_seq[sl] = {"perplexity": float("inf"), "oom": True}
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Error at seq_len={sl}: {e}")
                per_seq[sl] = {"perplexity": float("nan"), "error": str(e)}

        results[label] = per_seq
        _free(model)

    # ---- Print comparison ----
    print("\n" + "=" * 80)
    print(f"Perplexity comparison  (corpus: {source})")
    print("=" * 80)

    header = f"{'Setup':<28}|" + "|".join(f"{f'seq={sl}':>12}" for sl in EVAL_SEQ_LENS)
    print(header)
    print("-" * len(header))
    for label in results:
        row = f"{label:<28}|"
        for sl in EVAL_SEQ_LENS:
            ppl = results[label].get(sl, {}).get("perplexity", float("nan"))
            row += f" {ppl:>11.2f}"
        print(row)
    print("-" * len(header))

    # Throughput row
    print("\nDecode throughput (tokens/sec, single chunk forward):")
    print(header)
    print("-" * len(header))
    for label in results:
        row = f"{label:<28}|"
        for sl in EVAL_SEQ_LENS:
            tps = results[label].get(sl, {}).get("tokens_per_sec", float("nan"))
            row += f" {tps:>11.0f}"
        print(row)

    # ---- Save JSON ----
    import json
    payload = {
        "config": {
            "model": MODEL_PATH,
            "eval_corpus": source,
            "seq_lens": EVAL_SEQ_LENS,
            "max_chunks_per_seqlen": MAX_CHUNKS_PER_SEQLEN,
            "lc_window_size": LC_WINDOW_SIZE,
            "lc_gate_init_bias": LC_GATE_INIT_BIAS,
            "lc_mix_init_logit": LC_MIX_INIT_LOGIT,
            "sw_window_size": SW_WINDOW_SIZE,
            "sw_num_sinks": SW_NUM_SINKS,
        },
        "results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
