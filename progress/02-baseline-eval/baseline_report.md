# Baseline Report — Qwen3-4B (Full Attention)

This report the environment, protocol, and results for the **full-attention** Qwen3-4B baseline, following `progress/01-model-and-benchmark-spec/spec.md`.

## Logs location (not committed)
- Run date(s): YYYY-MM-DD
- Machine: <hostname>
- Absolute log directory:
  - `/ABS/PATH/TO/LOGS/`  
- Notes on what is inside that directory (naming scheme, runs, etc.):

## Environment
- GPU: <model>
- VRAM: <GiB>
- Driver: <version>
- CUDA: <version>

## Model
- Model id: `Qwen/Qwen3-4B`
- Dtype / quantization: `bf16`

## Protocol (pinned)
- Spec: `progress/01-model-and-benchmark-spec/spec.md`

### Decoding / sampling params (pin exactly)
- temperature: <...>
- top_p: <...>
- top_k: <... or null>
- repetition_penalty: <...>
- max_new_tokens: = `output_len`
- seed: <... or rule>

## Result files
- Detailed results:
  - `progress/02-baseline-eval/baseline_longbench_v2.csv`

## Results overview

### Throughput (vLLM)

| input_len | output_len | decode tok/s | total tok/s | req/s | peak mem (GiB) | OOM | notes |
|----------:|-----------:|-------------:|------------:|------:|---------------:|:---:|------|
| 4k        | 128        |              |             |       |                |     |      |
| 4k        | 512        |              |             |       |                |     |      |
| 4k        | 2048       |              |             |       |                |     |      |
| 8k        | 128        |              |             |       |                |     |      |
| 8k        | 512        |              |             |       |                |     |      |
| 8k        | 2048       |              |             |       |                |     |      |
| 16k       | 128        |              |             |       |                |     |      |
| 16k       | 512        |              |             |       |                |     |      |
| 16k       | 2048       |              |             |       |                |     |      |
| 32k       | 128        |              |             |       |                |     |      |
| 32k       | 512        |              |             |       |                |     |      |
| 32k       | 2048       |              |             |       |                |     |      |

### NIAH
- Dataset/version: <...>
- Prompt template / evaluation recipe: <...>

#### Summary (aggregate over depths)

| input_len | depth_rel | accuracy | notes |
|----------:|----------:|---------:|------|
| 4k        | 0.00      |          |      |
| 4k        | 0.50      |          |      |
| 4k        | 1.00      |          |      |
| 8k        | 0.00      |          |      |
| 8k        | 0.50      |          |      |
| 8k        | 1.00      |          |      |
| 16k       | 0.00      |          |      |
| 16k       | 0.50      |          |      |
| 16k       | 1.00      |          |      |
| 32k       | 0.00      |          |      |
| 32k       | 0.50      |          |      |
| 32k       | 1.00      |          |      |

### LongBench V2
- Dataset/version: <...>
- Prompt template / evaluation recipe: <...>
- Aggregation: <mean/median/etc> (pin what “score” means)

LongBench V2 consists of multiple subsets (e.g. `narrativeqa`, `qasper`, ...). Each sample belongs to:
- a **subset**
- a **task type** (one of: single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, code completion)
- a **prompt length bin** (by `prompt_len_tok`): 0-8k, 8k-16k, 16k-24k, 24k-32k

#### Summary (overall)
Fill this table with the corresponding scores by aggregating scores from `baseline_longbench_v2.csv` grouped by `task_type` (rows) and `prompt_len_bin` (columns)

| task_type | 0-8k | 8k-16k | 16k-24k | 24k-32k | overall | notes |
|----------|---:|---:|----:|----:|----:|------|
| single-document QA | | | | | | |
| multi-document QA  | | | | | | |
| summarization      | | | | | | |
| few-shot learning  | | | | | | |
| synthetic tasks    | | | | | | |
| code completion    | | | | | | |
| __overall__        | | | | | | |

## Notes / anomalies
- OOMs (where/why):
- Performance cliffs (where/why):
- Any deviations from spec:
- Anything that may affect comparability (batch size caps, fallback kernels, etc.):

## Repro pointers
High-level pointers only (raw commands/scripts live elsewhere).
- Throughput script entrypoint: <path or name>
- NIAH eval entrypoint: <path or name>
- LongBench V2 eval entrypoint: <path or name>
- How to regenerate tables from the three CSVs: <one-liner description>
