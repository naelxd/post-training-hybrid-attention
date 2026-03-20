"""
Needle In A Haystack test for VLLM models.
Uses Paul Graham essays as haystack, following the original test design.
Tests model's ability to retrieve information at different context lengths and depths.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from tqdm import tqdm


@dataclass
class TestResult:
    context_length: int
    depth_percent: float
    correct: bool
    response: str
    expected: str
    latency_ms: float


class VLLMClient:
    """Client for VLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "qwen3-4b"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)

    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> str:
        """Generate completion using VLLM API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }

        start = datetime.now()
        response = await self.client.post(f"{self.base_url}/completions", json=payload)
        latency_ms = (datetime.now() - start).total_seconds() * 1000

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"], latency_ms

    async def close(self):
        await self.client.aclose()


class PaulGrahamEssays:
    """Paul Graham essays loader from txt files."""

    def __init__(self, essays_dir: str = "PaulGrahamEssays"):
        self.essays_dir = Path(essays_dir)
        self.essays: list[str] = []
        self._load_essays()

    def _load_essays(self):
        """Load all .txt files from the essays directory."""
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

        print(f"Loaded {len(self.essays)} essays from {self.essays_dir}")
        total_chars = sum(len(e) for e in self.essays)
        print(f"Total characters: {total_chars:,} (~{total_chars // 4:,} tokens)")

    def get_essays_text(self, target_chars: int) -> str:
        """Get combined essays text of approximately target length."""
        if not self.essays:
            raise ValueError("No essays loaded")

        text = ""
        essay_indices = list(range(len(self.essays)))

        # Use essays in order first, then repeat if needed
        while len(text) < target_chars:
            for idx in essay_indices:
                text += self.essays[idx] + "\n\n"
                if len(text) >= target_chars:
                    break
            # Shuffle for next round if we need more text
            random.shuffle(essay_indices)

        return text[:target_chars]


class NeedleHaystackTest:
    """Needle In A Haystack test implementation using Paul Graham essays."""

    NEEDLE = "The secret code is: XJ9-ALPHA-7428-BETA. Remember this code."
    QUESTION = "What is the secret code?"
    EXPECTED_ANSWER = "XJ9-ALPHA-7428-BETA"

    # Target context lengths (in tokens)
    CONTEXT_LENGTHS = [4000, 8000, 16000, 32000]
    DEPTH_PERCENTS = [0.0, 0.5, 1.0]

    # Characters per token approximation
    CHARS_PER_TOKEN = 4

    def __init__(self, client: VLLMClient, essays: PaulGrahamEssays):
        self.client = client
        self.essays = essays
        self.results: list[TestResult] = []

    def get_haystack(self, target_chars: int) -> str:
        """Get haystack text (Paul Graham essays) of target length."""
        return self.essays.get_essays_text(target_chars)

    def insert_needle(self, haystack: str, depth_percent: float) -> str:
        """Insert needle at specified depth percentage in haystack."""
        if depth_percent == 0.0:
            # Insert at the beginning
            return f"{self.NEEDLE}\n\n{haystack}"
        elif depth_percent == 1.0:
            # Insert at the end
            return f"{haystack}\n\n{self.NEEDLE}"
        else:
            # Insert at percentage point
            insert_pos = int(len(haystack) * depth_percent)
            return f"{haystack[:insert_pos]}\n\n{self.NEEDLE}\n\n{haystack[insert_pos:]}"

    def build_prompt(self, context: str) -> str:
        """Build the full prompt with context and question."""
        return f"""Below is a collection of essays. Read them carefully and answer the question at the end.

{context}

Question: {self.QUESTION}

Answer the question directly and concisely. Do not include any explanation.
Answer:\\nothink"""

    def check_answer(self, response: str) -> bool:
        """Check if response contains the expected answer."""
        return self.EXPECTED_ANSWER in response

    async def run_single_test(self, context_length: int, depth_percent: float) -> TestResult:
        """Run a single test configuration."""
        target_chars = context_length * self.CHARS_PER_TOKEN

        # Get haystack and insert needle
        haystack = self.get_haystack(target_chars)
        context = self.insert_needle(haystack, depth_percent)

        # Build prompt and query model
        prompt = self.build_prompt(context)
        response, latency_ms = await self.client.generate(prompt, max_tokens=50)

        correct = self.check_answer(response)

        return TestResult(
            context_length=context_length,
            depth_percent=depth_percent,
            correct=correct,
            response=response.strip(),
            expected=self.EXPECTED_ANSWER,
            latency_ms=latency_ms,
        )

    async def run_all_tests(self, repetitions: int = 3) -> list[TestResult]:
        """Run all test configurations."""
        total_tests = len(self.CONTEXT_LENGTHS) * len(self.DEPTH_PERCENTS) * repetitions

        print(f"Running Needle In A Haystack test")
        print(f"Haystack: Paul Graham essays")
        print(f"Context lengths: {self.CONTEXT_LENGTHS}")
        print(f"Depth percents: {self.DEPTH_PERCENTS}")
        print(f"Repetitions: {repetitions}")
        print(f"Total tests: {total_tests}")
        print("-" * 50)

        results = []

        with tqdm(total=total_tests, desc="Testing") as pbar:
            for ctx_len in self.CONTEXT_LENGTHS:
                for depth in self.DEPTH_PERCENTS:
                    for rep in range(repetitions):
                        try:
                            result = await self.run_single_test(ctx_len, depth)
                            results.append(result)
                        except Exception as e:
                            print(f"\nError at {ctx_len}/{depth}: {e}")
                            results.append(
                                TestResult(
                                    context_length=ctx_len,
                                    depth_percent=depth,
                                    correct=False,
                                    response=f"ERROR: {e}",
                                    expected=self.EXPECTED_ANSWER,
                                    latency_ms=0,
                                )
                            )
                        pbar.update(1)

        self.results = results
        return results

    def analyze_results(self) -> dict:
        """Analyze and summarize results."""
        analysis = {}

        for ctx_len in self.CONTEXT_LENGTHS:
            ctx_results = [r for r in self.results if r.context_length == ctx_len]
            analysis[ctx_len] = {
                "overall_accuracy": sum(1 for r in ctx_results if r.correct) / len(ctx_results) * 100
                if ctx_results
                else 0,
                "by_depth": {},
            }

            for depth in self.DEPTH_PERCENTS:
                depth_results = [r for r in ctx_results if r.depth_percent == depth]
                if depth_results:
                    accuracy = sum(1 for r in depth_results if r.correct) / len(depth_results) * 100
                    avg_latency = sum(r.latency_ms for r in depth_results) / len(depth_results)
                    analysis[ctx_len]["by_depth"][depth] = {
                        "accuracy": accuracy,
                        "avg_latency_ms": avg_latency,
                        "correct": sum(1 for r in depth_results if r.correct),
                        "total": len(depth_results),
                    }

        return analysis

    def print_results(self):
        """Print formatted results table."""
        analysis = self.analyze_results()

        print("\n" + "=" * 70)
        print("NEEDLE IN A HAYSTACK RESULTS")
        print("=" * 70)
        print(f"\nNeedle: {self.NEEDLE}")
        print(f"Question: {self.QUESTION}")
        print(f"Expected: {self.EXPECTED_ANSWER}")
        print("\n" + "-" * 70)

        # Header
        header = f"{'Context':<12}"
        for depth in self.DEPTH_PERCENTS:
            header += f"| Depth {depth:.1f} "
        header += "| Overall"
        print(header)
        print("-" * 70)

        # Results by context length
        for ctx_len in self.CONTEXT_LENGTHS:
            row = f"{ctx_len:<12}"
            ctx_data = analysis.get(ctx_len, {})

            for depth in self.DEPTH_PERCENTS:
                depth_data = ctx_data.get("by_depth", {}).get(depth, {})
                acc = depth_data.get("accuracy", 0)
                marker = "✓" if acc == 100 else ("~" if acc >= 50 else "✗")
                row += f"| {acc:>5.0f}% {marker}  "

            overall = ctx_data.get("overall_accuracy", 0)
            row += f"| {overall:>5.0f}%"
            print(row)

        print("-" * 70)
        print("\nLegend: ✓ = 100% accuracy, ~ = 50-99%, ✗ = <50%")

        # Detailed failure analysis
        failures = [r for r in self.results if not r.correct]
        if failures:
            print(f"\nFailed tests: {len(failures)}")
            print("\nSample failures:")
            for fail in failures[:3]:
                print(f"  {fail.context_length} @ depth {fail.depth_percent}: got '{fail.response[:50]}...'")

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        data = {
            "test_config": {
                "needle": self.NEEDLE,
                "question": self.QUESTION,
                "expected": self.EXPECTED_ANSWER,
                "context_lengths": self.CONTEXT_LENGTHS,
                "depth_percents": self.DEPTH_PERCENTS,
                "haystack_source": f"Paul Graham essays ({len(self.essays.essays)} files)",
            },
            "results": [
                {
                    "context_length": r.context_length,
                    "depth_percent": r.depth_percent,
                    "correct": r.correct,
                    "response": r.response,
                    "expected": r.expected,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
            "analysis": self.analyze_results(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {filepath}")


async def main():
    # Configuration - adjust as needed
    VLLM_URL = "http://localhost:8000/v1"
    MODEL_NAME = "Qwen/Qwen3-4B"
    ESSAYS_DIR = "needlehaystack/PaulGrahamEssays"

    print(f"Connecting to VLLM at {VLLM_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Essays directory: {ESSAYS_DIR}")
    print("-" * 50)

    # Load essays
    essays = PaulGrahamEssays(ESSAYS_DIR)

    client = VLLMClient(base_url=VLLM_URL, model=MODEL_NAME)

    try:
        test = NeedleHaystackTest(client, essays)
        await test.run_all_tests(repetitions=3)
        test.print_results()
        test.save_results("niah_results.json")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
