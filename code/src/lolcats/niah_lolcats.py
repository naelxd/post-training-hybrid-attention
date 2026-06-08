#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Needle In A Haystack — LoLCATs version.

Compares the LoLCATs hybrid (SW + GLA on non-retrieval heads, full attention
on retrieval heads from compressible_heads.pt) against the same hybrid with a
*random* set of full-attention heads (same per-layer count).

Optionally runs LoLCATs Stage-1 attention transfer (random batches by default
— replace ``random_batches()`` with a real tokenised corpus for a meaningful
training signal) before evaluation.

Mirrors the structure of niah.py.
"""

import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_lolcats import Qwen3LoLCATsPatcher


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_PATH = "Qwen/Qwen3-4B"
MODEL_DTYPE = torch.bfloat16
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_HEADS_PER_LAYER = 32

# LoLCATs hyperparameters
WINDOW_SIZE = 512
GATE_INIT_BIAS = 5.0    # logits → gates ≈ 1 (near-identity decay)
MIX_INIT_LOGIT = 5.0    # sigmoid(5) ≈ 0.99 → mostly SW at init

# Trained LoLCATs weights.  Generate via train_lolcats.py.  Set to None (or
# point at a missing file) to evaluate the pre-train baseline: with
# MIX_INIT_LOGIT large the hybrid is ≈ pure sliding window.
WEIGHTS_PATH = "lolcats_weights.pt"

# Files
COMPRESSIBLE_HEADS_PATH = "compressible_heads_2.pt"
OUTPUT_FILE = "niah_lolcats.json"

# Random seed
SEED = 42

# "selective" | "random" | "both"
TEST_MODE = "both"

# NIAH
NEEDLE = "The secret code is: XJ9-ALPHA-7428-BETA. Remember this code."
QUESTION = "What is the secret code?"
EXPECTED_ANSWER = "XJ9-ALPHA-7428-BETA"
CONTEXT_LENGTHS = [4000, 8000, 16000, 32000]
DEPTH_PERCENTS = [0.0, 0.5, 1.0]
CHARS_PER_TOKEN = 4
MAX_NEW_TOKENS = 20
REPETITIONS = 3

ESSAYS_DIR = "LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays"


# =============================================================================


@dataclass
class TestResult:
    context_length: int
    depth_percent: float
    test_type: str
    correct: bool
    response: str
    expected: str
    latency_ms: float


class PaulGrahamEssays:
    """Paul Graham essays loader (singleton)."""

    _instance = None
    _essays = None

    def __new__(cls, essays_dir: str = ESSAYS_DIR):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_essays(essays_dir)
        return cls._instance

    def _load_essays(self, essays_dir: str):
        self.essays_dir = Path(essays_dir)
        self.essays: List[str] = []
        if not self.essays_dir.exists():
            raise FileNotFoundError(f"Essays directory not found: {self.essays_dir}")
        txt_files = sorted(self.essays_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.essays_dir}")
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    self.essays.append(content)
        print(f"Loaded {len(self.essays)} essays")
        PaulGrahamEssays._essays = self.essays

    def get_essays_text(self, target_chars: int) -> str:
        if not self.essays:
            raise ValueError("No essays loaded")
        text = ""
        essay_indices = list(range(len(self.essays)))
        while len(text) < target_chars:
            for idx in essay_indices:
                text += self.essays[idx] + "\n\n"
                if len(text) >= target_chars:
                    break
            random.shuffle(essay_indices)
        return text[:target_chars]


class NeedleHaystackTest:
    """NIAH for the LoLCATs hybrid: selective vs random retrieval heads."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.model_selective = None
        self.model_random = None
        self.patcher_selective = None
        self.patcher_random = None

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        print("\n" + "=" * 80)
        print(f"NIAH (LoLCATs) - Mode: {TEST_MODE}")
        print(f"Device: {MODEL_DEVICE}")
        print("=" * 80)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load compressible heads
        print(f"\nLoading compressible heads from {COMPRESSIBLE_HEADS_PATH}...")
        compressible_data = torch.load(
            COMPRESSIBLE_HEADS_PATH, weights_only=False, map_location='cpu'
        )

        self.selective_heads: Dict[int, List[int]] = {}
        induction_heads = compressible_data.get("prefix_matching", {})
        echo_heads = compressible_data.get("copying", {})
        for layer_idx, heads in induction_heads.items():
            self.selective_heads[layer_idx] = (
                self.selective_heads.get(layer_idx, []) + heads
            )
        for layer_idx, heads in echo_heads.items():
            self.selective_heads[layer_idx] = (
                self.selective_heads.get(layer_idx, []) + heads
            )

        self.random_heads = self._generate_random_heads()

        total_sel = sum(len(h) for h in self.selective_heads.values())
        total_rnd = sum(len(h) for h in self.random_heads.values())
        print(f"\nSelective full-attn heads: {total_sel}")
        print(f"Random   full-attn heads: {total_rnd}")

        if TEST_MODE in ("selective", "both"):
            print("\nLoading model with SELECTIVE retrieval heads...")
            self.model_selective, self.patcher_selective = (
                self._load_and_patch_model(self.selective_heads)
            )
            self._maybe_load_weights(self.patcher_selective, "selective")

        if TEST_MODE in ("random", "both"):
            print("\nLoading model with RANDOM retrieval heads...")
            self.model_random, self.patcher_random = (
                self._load_and_patch_model(self.random_heads)
            )
            self._maybe_load_weights(self.patcher_random, "random")

        print("\n" + "=" * 80)
        print(f"Ready! Mode: {TEST_MODE}")
        print(f"Window: {WINDOW_SIZE}, gate_bias={GATE_INIT_BIAS}, "
              f"mix_init={MIX_INIT_LOGIT}")
        print("=" * 80)

    # ---------- model loading / patching ----------

    def _load_and_patch_model(self, keep_heads: Dict[int, List[int]]):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=MODEL_DTYPE,
            device_map=MODEL_DEVICE,
            attn_implementation="flash_attention_2",
        )
        patcher = Qwen3LoLCATsPatcher(
            model,
            window_size=WINDOW_SIZE,
            gate_init_bias=GATE_INIT_BIAS,
            mix_init_logit=MIX_INIT_LOGIT,
        )
        patcher.patch_model(keep_heads)
        return model, patcher

    def _maybe_load_weights(self, patcher: Qwen3LoLCATsPatcher, label: str):
        """Load trained LoLCATs weights from WEIGHTS_PATH if available."""
        if WEIGHTS_PATH is None or not Path(WEIGHTS_PATH).exists():
            print(f"  [{label}] No weights file at {WEIGHTS_PATH!r} — using "
                  f"INIT weights (mix_init={MIX_INIT_LOGIT} → behaves ≈ pure "
                  f"SW).  Train via train_lolcats.py to populate the GLA "
                  f"branch.")
            return
        print(f"  [{label}] Loading trained weights from {WEIGHTS_PATH}")
        patcher.load_weights(WEIGHTS_PATH)

    def _unload_model(self, model):
        if model is not None:
            del model
            torch.cuda.empty_cache()
            gc.collect()

    def _generate_random_heads(self) -> Dict[int, List[int]]:
        random_heads = {}
        for layer_idx, heads in self.selective_heads.items():
            n = len(heads)
            random_heads[layer_idx] = sorted(
                random.sample(range(NUM_HEADS_PER_LAYER), n)
            )
        return random_heads

    # ---------- prompt / scoring ----------

    def get_haystack(self, target_chars: int) -> str:
        essays = PaulGrahamEssays()
        return essays.get_essays_text(target_chars)

    def insert_needle(self, haystack: str, depth_percent: float) -> str:
        if depth_percent == 0.0:
            return f"{NEEDLE}\n\n{haystack}"
        if depth_percent == 1.0:
            return f"{haystack}\n\n{NEEDLE}"
        insert_pos = int(len(haystack) * depth_percent)
        return f"{haystack[:insert_pos]}\n\n{NEEDLE}\n\n{haystack[insert_pos:]}"

    def build_prompt(self, context: str) -> str:
        return (
            "Below is a collection of essays. Read them carefully and answer "
            "the question at the end.\n\n"
            f"{context}\n\n"
            f"Question: {QUESTION}\n\n"
            "Answer the question directly and concisely. Do not include any "
            "explanation.\nAnswer:\n"
        )

    def check_answer(self, response: str) -> bool:
        return EXPECTED_ANSWER in response

    # ---------- single test ----------

    @torch.no_grad()
    def run_single_test(
        self, model, context_length: int, depth_percent: float, test_type: str,
    ) -> TestResult:
        target_chars = context_length * CHARS_PER_TOKEN
        haystack = self.get_haystack(target_chars)
        context = self.insert_needle(haystack, depth_percent)
        prompt = self.build_prompt(context)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        latency_ms = (time.time() - start_time) * 1000

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        correct = self.check_answer(response)

        print(f"  {test_type:10s} | Depth {depth_percent:.1f} | "
              f"{'✓' if correct else '✗'} | {response[:35]}... | {latency_ms:5.0f}ms")

        return TestResult(
            context_length=context_length,
            depth_percent=depth_percent,
            test_type=test_type,
            correct=correct,
            response=response.strip(),
            expected=EXPECTED_ANSWER,
            latency_ms=latency_ms,
        )

    # ---------- driver ----------

    def run_all_tests(self) -> List[TestResult]:
        num_models = 2 if TEST_MODE == "both" else 1
        total_tests = (
            len(CONTEXT_LENGTHS) * len(DEPTH_PERCENTS) * REPETITIONS * num_models
        )

        print("\n" + "=" * 80)
        print(f"NIAH (LoLCATs): {TEST_MODE.upper()}")
        print("-" * 80)
        print(f"Model: {MODEL_PATH}")
        print(f"Window: {WINDOW_SIZE}, weights={WEIGHTS_PATH}")
        print(f"Seed: {SEED}")
        print(f"Context lengths: {CONTEXT_LENGTHS}")
        print(f"Depth percents: {DEPTH_PERCENTS}")
        print(f"Total tests: {total_tests}")
        print("-" * 80)

        results = []
        with tqdm(total=total_tests, desc="Testing") as pbar:
            for ctx_len in CONTEXT_LENGTHS:
                print(f"\n{'─'*60}")
                print(f"Context: {ctx_len} tokens")
                print(f"{'─'*60}")

                for depth in DEPTH_PERCENTS:
                    for rep in range(REPETITIONS):
                        if TEST_MODE in ("selective", "both"):
                            try:
                                result = self.run_single_test(
                                    self.model_selective, ctx_len, depth, "selective"
                                )
                                results.append(result)
                            except Exception as e:
                                print(f"  selective   | Error: {e}")
                            pbar.update(1)

                        if TEST_MODE in ("random", "both"):
                            try:
                                result = self.run_single_test(
                                    self.model_random, ctx_len, depth, "random"
                                )
                                results.append(result)
                            except Exception as e:
                                print(f"  random      | Error: {e}")
                            pbar.update(1)

        self.results = results
        return results

    # ---------- analysis ----------

    def analyze_results(self) -> Dict:
        analysis = {}
        for ctx_len in CONTEXT_LENGTHS:
            ctx_results = [r for r in self.results if r.context_length == ctx_len]
            sel = [r for r in ctx_results if r.test_type == "selective"]
            rnd = [r for r in ctx_results if r.test_type == "random"]
            analysis[ctx_len] = {
                "selective": {
                    "accuracy": (sum(1 for r in sel if r.correct) / len(sel) * 100
                                 if sel else 0),
                    "avg_latency_ms": (sum(r.latency_ms for r in sel) / len(sel)
                                       if sel else 0),
                },
                "random": {
                    "accuracy": (sum(1 for r in rnd if r.correct) / len(rnd) * 100
                                 if rnd else 0),
                    "avg_latency_ms": (sum(r.latency_ms for r in rnd) / len(rnd)
                                       if rnd else 0),
                },
                "by_depth": {},
            }
            for depth in DEPTH_PERCENTS:
                s = [r for r in sel if r.depth_percent == depth]
                r_ = [r for r in rnd if r.depth_percent == depth]
                analysis[ctx_len]["by_depth"][depth] = {
                    "selective_accuracy": (sum(1 for r in s if r.correct) / len(s) * 100
                                           if s else 0),
                    "random_accuracy": (sum(1 for r in r_ if r.correct) / len(r_) * 100
                                        if r_ else 0),
                }
        return analysis

    def print_results(self):
        analysis = self.analyze_results()
        print("\n" + "=" * 80)
        print(f"NIAH (LoLCATs) RESULTS: {TEST_MODE.upper()}")
        print("=" * 80)
        print(f"Model: {MODEL_PATH}")
        print(f"Needle: {NEEDLE[:50]}...")
        print("\n" + "-" * 60)

        if TEST_MODE == "both":
            print(f"\n{'Context':<10}| {'Selective':<12}| {'Random':<12}| {'Δ':<10}")
            print("-" * 60)
            for ctx_len in CONTEXT_LENGTHS:
                sel_acc = analysis[ctx_len]["selective"]["accuracy"]
                rnd_acc = analysis[ctx_len]["random"]["accuracy"]
                delta = sel_acc - rnd_acc
                print(f"{ctx_len:<10}| {sel_acc:>5.0f}%       | "
                      f"{rnd_acc:>5.0f}%       | {delta:>+5.0f}%")
            print("-" * 60)
            print("\nBreakdown by depth:")
            print("-" * 60)
            for ctx_len in CONTEXT_LENGTHS:
                print(f"\nContext {ctx_len}:")
                for depth in DEPTH_PERCENTS:
                    d = analysis[ctx_len]["by_depth"][depth]
                    delta = d["selective_accuracy"] - d["random_accuracy"]
                    print(f"  Depth {depth:.1f}: Selective={d['selective_accuracy']:>5.0f}%, "
                          f"Random={d['random_accuracy']:>5.0f}%, Δ={delta:>+5.0f}%")
        else:
            test_type = TEST_MODE
            print(f"\n{'Context':<10}| {'Accuracy':<12}| {'Avg Latency':<15}")
            print("-" * 60)
            for ctx_len in CONTEXT_LENGTHS:
                acc = analysis[ctx_len][test_type]["accuracy"]
                latency = analysis[ctx_len][test_type]["avg_latency_ms"]
                print(f"{ctx_len:<10}| {acc:>5.0f}%       | {latency:>8.0f}ms")
            print("-" * 60)
            print("\nBreakdown by depth:")
            print("-" * 60)
            for ctx_len in CONTEXT_LENGTHS:
                print(f"\nContext {ctx_len}:")
                for depth in DEPTH_PERCENTS:
                    d = analysis[ctx_len]["by_depth"][depth]
                    acc_key = f"{test_type}_accuracy"
                    print(f"  Depth {depth:.1f}: Accuracy={d[acc_key]:>5.0f}%")

        print("\n" + "-" * 60)
        print(f"Window: {WINDOW_SIZE}, gate_bias={GATE_INIT_BIAS}, "
              f"mix_init={MIX_INIT_LOGIT}, weights={WEIGHTS_PATH}, seed={SEED}")

    def save_results(self, filepath: str):
        analysis = self.analyze_results()
        selective_info = {str(k): v for k, v in self.selective_heads.items()}
        random_info = {str(k): v for k, v in self.random_heads.items()}
        data = {
            "config": {
                "model": MODEL_PATH,
                "test_mode": TEST_MODE,
                "num_heads_per_layer": NUM_HEADS_PER_LAYER,
                "window_size": WINDOW_SIZE,
                "gate_init_bias": GATE_INIT_BIAS,
                "mix_init_logit": MIX_INIT_LOGIT,
                "weights_path": WEIGHTS_PATH,
                "weights_loaded": (WEIGHTS_PATH is not None and Path(WEIGHTS_PATH).exists()),
                "seed": SEED,
                "needle": NEEDLE,
                "question": QUESTION,
                "expected": EXPECTED_ANSWER,
                "context_lengths": CONTEXT_LENGTHS,
                "depth_percents": DEPTH_PERCENTS,
                "selective_heads": selective_info,
                "random_heads": random_info,
            },
            "results": [
                {
                    "context_length": r.context_length,
                    "depth_percent": r.depth_percent,
                    "test_type": r.test_type,
                    "correct": r.correct,
                    "response": r.response,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
            "analysis": analysis,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {filepath}")


def main():
    test = NeedleHaystackTest()
    test.run_all_tests()
    test.print_results()
    test.save_results(OUTPUT_FILE)
    print("\n" + "=" * 80)
    print("NIAH (LoLCATs) test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
