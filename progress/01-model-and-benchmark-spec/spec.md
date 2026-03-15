# Model + Benchmark Spec

This task defines the baseline model choice and the evaluation protocol for long-context inference.

## Model choice

- **Model**: Qwen3 4B
- **Rationale**: Fits long-context inference on a single 24GB GPU while keeping a reasonably capable base model.

## Context lengths

We evaluate long-context behavior at the following **input (prompt/prefill) lengths**:

- 4k, 8k, 16k, 32k input tokens

We keep generation length (`output_len`) as a separate axis. When we need the total sequence length:

- `total_len = input_len + output_len`

Throughout this spec, when we say “context length”, we mean `input_len` (prompt/prefill tokens).

## Throughput evaluation (vLLM)

### Tooling

- Use vLLM throughput benchmark (the standard vLLM benchmark script) with **random prompts**.
- Measure across three decode lengths:
  - `output_len` in {128, 512, 2048}

### Reporting metrics (throughput)

At minimum, report for each `(input_len, output_len)`:

- **Throughput**:
  - total (input+output) tokens/sec (tok/s) and req/sec (both reported to json by vllm)
  - separately record **prefill tok/s** and **decode tok/s** (reported to stdout by vllm)
- **Memory**:
  - peak GPU memory (or best available proxy)
  - whether any run OOMs

### Protocol / reproducibility rules

- Fix sampling parameters for all throughput tests (e.g., temperature, top-p) so decode behavior is consistent across runs.
- Run each configuration >= 3 times and report the **median** (also keep raw runs in logs).
- Record:
  - GPU model + VRAM
  - CUDA driver
  - vLLM version/commit

## Accuracy evaluation

### Benchmarks

- **NIAH** (Needle In A Haystack)
  - Evaluate at input lengths 4k/8k/16k/32k.
  - Report accuracy vs input length (`input_len`).
  - If the NIAH setup supports varying needle positions, also report the aggregate across positions (and optionally a plot/summary by position bucket).

- **LongBench V2**
  - Evaluate using the standard LongBench V2 evaluation recipe.
  - Report:
    - overall score (primary)
    - per-task/per-category breakdown (secondary, but required for debugging regressions)
  - If LongBench V2 requires a specific prompt template or decoding settings, pin them and report them.

### Reporting metrics (accuracy)

For each (model/config, input length):

- NIAH: accuracy (and, if applicable, accuracy by needle position buckets)
- LongBench V2: overall score + per-task scores

## How results are reported

All results should be reportable as a single “main table” plus small supporting tables/plots.

### Main table

One row per configuration per length setting:

- Model/config name (e.g., `qwen3-4b-full`, `qwen3-4b-hybrid-Lx`, etc.)
- dtype/quantization
- `input_len`
- `output_len`
- `total_len`
- throughput (tok/s)
- output throughput (tok/s)
- peak GPU memory (GiB)
- notes (OOM, fallback, instability)

### Accuracy tables

- NIAH table: rows=config, cols=`input_len`, cells=accuracy
- LongBench V2 table: rows=config, cols=`input_len`, cells=overall score

### Minimal reporting conventions (to keep results comparable)

- Use a consistent identifier scheme for configs: `model + attention_variant + length_setting + dtype`.
- For any averaged number, state the aggregation: `median over N runs` for throughput, and the exact dataset split/version for accuracy.
- Always include a small “environment block” with: GPU, CUDA, driver, PyTorch, vLLM, and the exact model revision/hash.
