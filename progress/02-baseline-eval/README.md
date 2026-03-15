# Task 02 — Baseline Evaluation (Qwen3 Full Attention)

Implement the baseline evaluation pipeline for **Qwen3-4B (full attention)** and produce baseline results following the protocol in `progress/01-model-and-benchmark-spec/spec.md`.

## Pinned spec
- **Model**: Qwen3-4B
- **Input lengths**: 4k / 8k / 16k / 32k tokens
- **Decode lengths** (`output_len`): 128 / 512 / 2048
- **Stacks**:
  - vLLM: throughput + memory (random prompts)
  - HF Transformers: accuracy eval (LongBench V2, NIAH)

## What “done” means
- Baseline scripts exist (even if minimal) to run:
  - vLLM throughput sweep over `(input_len, output_len)`
  - NIAH sweep over `(input_len, depth_rel)` where `depth_rel` in {0, 0.5, 1}
  - LongBench V2 sweep over `input_len`
- `baseline_report.md` exists and summarizes the environment + results
- `baseline_longbench_v2.csv` exists and contains per-sample information about LongBench-V2 runs

## Required artifacts to save
- `progress/02-baseline-eval/baseline_report.md`
    - **Logs location (not committed)**: absolute path(s) to raw logs + timestamps
    - **Environment block**: GPU, VRAM, driver, CUDA
    - **Model identity**: exact model id
    - **Protocol**: input/output lengths, number of runs, aggregation (median), decoding params
    - **Key results overview**: main tables (throughput + NIAH + LongBench V2) + notes on OOMs/anomalies
    - **Repro notes**: minimal “how to rerun” pointers

- `progress/02-baseline-eval/baseline_longbench_v2.csv`
  - LongBench V2 results (one row per sample; includes subset, task type, and prompt length bin)

## Checklist
- [ ] vLLM throughput
  - [ ] Use random prompts; fix sampling params across runs
  - [ ] Run >= 3 times per config; report median
  - [ ] Record decode tok/s, total tok/s, peak GPU mem, OOMs
- [ ] NIAH accuracy
  - [ ] Run at 4k/8k/16k/32k and depth 0/0.5/1; record accuracy vs length and depth
- [ ] LongBench V2 accuracy
  - [ ] Run standard recipe; pin prompts/decoding settings
  - [ ] Save per-sample results with subset + task type + prompt length bins
- [ ] Fill the “results overview” in `baseline_report.md`
