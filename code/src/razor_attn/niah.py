#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Needle In A Haystack test comparing:
1. Selective sliding window (proper retrieval heads from compressible_heads.pt)
2. Random head selection (same number of heads, randomly chosen)

Loads two models once at startup, then runs all tests.
"""

import torch
import json
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen3_sliding_window import Qwen3SelectiveSlidingWindowPatcher


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_PATH = "Qwen/Qwen3-4B"
MODEL_DTYPE = torch.bfloat16        # bf16 required: two 4B models = ~16 GB, fits in 25 GB
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_HEADS_PER_LAYER = 32            # Qwen3-4B: 32 attention heads per layer

# Sliding window
WINDOW_SIZE = 512
NUM_SINKS = 2                       # not supported in flash_attn path, kept for API compat

# Files
COMPRESSIBLE_HEADS_PATH = "compressible_heads_2.pt"
OUTPUT_FILE = "niah_selective_vs_random.json"

# Random seed
SEED = 42

# Test mode: "selective", "random", or "both"
# Note: "both" loads two 4B models simultaneously (~16 GB bf16 + KV cache).
# Switch to "selective" or "random" if VRAM is tight at long contexts.
TEST_MODE = "random"  # "selective" | "random" | "both"

# NIAH test
NEEDLE = "The secret code is: XJ9-ALPHA-7428-BETA. Remember this code."
QUESTION = "What is the secret code?"
EXPECTED_ANSWER = "XJ9-ALPHA-7428-BETA"
# Start from 2k — sliding window has no effect below window_size.
# 8k is the main stress test where SW compression matters for 25 GB VRAM.
CONTEXT_LENGTHS = [4000, 8000, 16000, 32000]
DEPTH_PERCENTS = [0.0, 0.5, 1.0]
CHARS_PER_TOKEN = 4
MAX_NEW_TOKENS = 20
REPETITIONS = 3

# Essays
ESSAYS_DIR = "LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays"

# =============================================================================


@dataclass
class TestResult:
    context_length: int
    depth_percent: float
    test_type: str  # "selective" or "random"
    correct: bool
    response: str
    expected: str
    latency_ms: float


class PaulGrahamEssays:
    """Paul Graham essays loader from txt files."""

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
    """NIAH test comparing selective vs random head selection."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.model_selective = None
        self.model_random = None

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        print("\n" + "=" * 80)
        print(f"NIAH test - Mode: {TEST_MODE}")
        print(f"Device: {MODEL_DEVICE}")
        print("=" * 80)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load compressible heads
        print(f"\nLoading compressible heads from {COMPRESSIBLE_HEADS_PATH}...")
        compressible_data = torch.load(COMPRESSIBLE_HEADS_PATH, weights_only=False, map_location='cpu')

        # Build selective heads from induction + echo heads
        self.selective_heads = {}
        induction_heads = compressible_data.get("prefix_matching", {})
        echo_heads = compressible_data.get("copying", {})

        for layer_idx, heads in induction_heads.items():
            self.selective_heads[layer_idx] = self.selective_heads.get(layer_idx, []) + heads
        for layer_idx, heads in echo_heads.items():
            self.selective_heads[layer_idx] = self.selective_heads.get(layer_idx, []) + heads

        # Generate random heads with same distribution
        self.random_heads = self._generate_random_heads()

        total_selective = sum(len(h) for h in self.selective_heads.values())
        total_random = sum(len(h) for h in self.random_heads.values())

        print(f"\nSelective heads: {total_selective}")
        print(f"Random heads: {total_random}")

        # Load models based on TEST_MODE
        if TEST_MODE in ("selective", "both"):
            print("\nLoading model with SELECTIVE heads...")
            self.model_selective = self._load_and_patch_model(self.selective_heads)

        if TEST_MODE in ("random", "both"):
            print("\nLoading model with RANDOM heads...")
            self.model_random = self._load_and_patch_model(self.random_heads)

        print("\n" + "=" * 80)
        print(f"Ready! Mode: {TEST_MODE}")
        print(f"Window: ±{WINDOW_SIZE}, Sinks: {NUM_SINKS}")
        print("=" * 80)

    def _load_and_patch_model(self, keep_heads: Dict[int, List[int]]):
        """Load fresh model and apply patch."""
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=MODEL_DTYPE,
            device_map=MODEL_DEVICE,
            attn_implementation="flash_attention_2",
        )
        patcher = Qwen3SelectiveSlidingWindowPatcher(model, WINDOW_SIZE, NUM_SINKS)
        return patcher.patch_model(keep_heads)

    def _unload_model(self, model):
        """Unload model to free VRAM."""
        if model is not None:
            del model
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def _generate_random_heads(self) -> Dict[int, List[int]]:
        """Generate random heads with same per-layer distribution as selective_heads."""
        random_heads = {}

        for layer_idx, heads in self.selective_heads.items():
            num_heads_to_select = len(heads)
            all_head_indices = list(range(NUM_HEADS_PER_LAYER))
            random_heads[layer_idx] = sorted(random.sample(all_head_indices, num_heads_to_select))

        return random_heads

    def get_haystack(self, target_chars: int) -> str:
        essays = PaulGrahamEssays()
        return essays.get_essays_text(target_chars)

    def insert_needle(self, haystack: str, depth_percent: float) -> str:
        if depth_percent == 0.0:
            return f"{NEEDLE}\n\n{haystack}"
        elif depth_percent == 1.0:
            return f"{haystack}\n\n{NEEDLE}"
        else:
            insert_pos = int(len(haystack) * depth_percent)
            return f"{haystack[:insert_pos]}\n\n{NEEDLE}\n\n{haystack[insert_pos:]}"

    def build_prompt(self, context: str) -> str:
        return f"""Below is a collection of essays. Read them carefully and answer the question at the end.

{context}

Question: {QUESTION}

Answer the question directly and concisely. Do not include any explanation.
Answer:\n"""

    def check_answer(self, response: str) -> bool:
        return EXPECTED_ANSWER in response

    @torch.no_grad()
    def run_single_test(self, model, context_length: int, depth_percent: float, test_type: str) -> TestResult:
        import time

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
            pad_token_id=self.tokenizer.pad_token_id
        )

        latency_ms = (time.time() - start_time) * 1000

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        correct = self.check_answer(response)

        print(f"  {test_type:10s} | Depth {depth_percent:.1f} | {'✓' if correct else '✗'} | "
              f"{response[:35]}... | {latency_ms:5.0f}ms")

        return TestResult(
            context_length=context_length,
            depth_percent=depth_percent,
            test_type=test_type,
            correct=correct,
            response=response.strip(),
            expected=EXPECTED_ANSWER,
            latency_ms=latency_ms,
        )

    def run_all_tests(self) -> List[TestResult]:
        num_models = 2 if TEST_MODE == "both" else 1
        total_tests = len(CONTEXT_LENGTHS) * len(DEPTH_PERCENTS) * REPETITIONS * num_models

        print("\n" + "=" * 80)
        print(f"NIAH TEST: {TEST_MODE.upper()}")
        print("-" * 80)
        print(f"Model: {MODEL_PATH}")
        print(f"Window: ±{WINDOW_SIZE}, Sinks: {NUM_SINKS}")
        print(f"Seed: {SEED}")
        print("-" * 80)
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
                        # Test SELECTIVE
                        if TEST_MODE in ("selective", "both"):
                            try:
                                result = self.run_single_test(self.model_selective, ctx_len, depth, "selective")
                                results.append(result)
                            except Exception as e:
                                print(f"  selective   | Error: {e}")
                            pbar.update(1)

                        # Test RANDOM
                        if TEST_MODE in ("random", "both"):
                            try:
                                result = self.run_single_test(self.model_random, ctx_len, depth, "random")
                                results.append(result)
                            except Exception as e:
                                print(f"  random      | Error: {e}")
                            pbar.update(1)

        self.results = results
        return results

    def analyze_results(self) -> Dict:
        analysis = {}

        for ctx_len in CONTEXT_LENGTHS:
            ctx_results = [r for r in self.results if r.context_length == ctx_len]

            selective_results = [r for r in ctx_results if r.test_type == "selective"]
            random_results = [r for r in ctx_results if r.test_type == "random"]

            analysis[ctx_len] = {
                "selective": {
                    "accuracy": sum(1 for r in selective_results if r.correct) / len(selective_results) * 100 if selective_results else 0,
                    "avg_latency_ms": sum(r.latency_ms for r in selective_results) / len(selective_results) if selective_results else 0,
                },
                "random": {
                    "accuracy": sum(1 for r in random_results if r.correct) / len(random_results) * 100 if random_results else 0,
                    "avg_latency_ms": sum(r.latency_ms for r in random_results) / len(random_results) if random_results else 0,
                },
                "by_depth": {},
            }

            for depth in DEPTH_PERCENTS:
                sel_depth = [r for r in selective_results if r.depth_percent == depth]
                rand_depth = [r for r in random_results if r.depth_percent == depth]

                analysis[ctx_len]["by_depth"][depth] = {
                    "selective_accuracy": sum(1 for r in sel_depth if r.correct) / len(sel_depth) * 100 if sel_depth else 0,
                    "random_accuracy": sum(1 for r in rand_depth if r.correct) / len(rand_depth) * 100 if rand_depth else 0,
                }

        return analysis

    def print_results(self):
        analysis = self.analyze_results()

        print("\n" + "=" * 80)
        print(f"NIAH RESULTS: {TEST_MODE.upper()}")
        print("=" * 80)
        print(f"Model: {MODEL_PATH}")
        print(f"Needle: {NEEDLE[:50]}...")
        print("\n" + "-" * 60)

        if TEST_MODE == "both":
            # Comparison table
            print(f"\n{'Context':<10}| {'Selective':<12}| {'Random':<12}| {'Δ':<10}")
            print("-" * 60)

            for ctx_len in CONTEXT_LENGTHS:
                sel_acc = analysis[ctx_len]["selective"]["accuracy"]
                rand_acc = analysis[ctx_len]["random"]["accuracy"]
                delta = sel_acc - rand_acc

                print(f"{ctx_len:<10}| {sel_acc:>5.0f}%       | {rand_acc:>5.0f}%       | {delta:>+5.0f}%")

            print("-" * 60)

            # Per-depth breakdown
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
            # Single mode
            test_type = TEST_MODE
            print(f"\n{'Context':<10}| {'Accuracy':<12}| {'Avg Latency':<15}")
            print("-" * 60)

            for ctx_len in CONTEXT_LENGTHS:
                acc = analysis[ctx_len][test_type]["accuracy"]
                latency = analysis[ctx_len][test_type]["avg_latency_ms"]
                print(f"{ctx_len:<10}| {acc:>5.0f}%       | {latency:>8.0f}ms")

            print("-" * 60)

            # Per-depth breakdown
            print("\nBreakdown by depth:")
            print("-" * 60)

            for ctx_len in CONTEXT_LENGTHS:
                print(f"\nContext {ctx_len}:")
                for depth in DEPTH_PERCENTS:
                    d = analysis[ctx_len]["by_depth"][depth]
                    acc_key = f"{test_type}_accuracy"
                    print(f"  Depth {depth:.1f}: Accuracy={d[acc_key]:>5.0f}%")

        print("\n" + "-" * 60)
        print(f"Window: ±{WINDOW_SIZE}, Sinks: {NUM_SINKS}, Seed: {SEED}")

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
                "num_sinks": NUM_SINKS,
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
    results = test.run_all_tests()
    test.print_results()
    test.save_results(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("NIAH test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

