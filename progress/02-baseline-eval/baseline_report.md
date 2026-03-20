# Baseline Report — Qwen3-4B (Full Attention)

This report the environment, protocol, and results for the **full-attention** Qwen3-4B baseline, following `progress/01-model-and-benchmark-spec/spec.md`.

## Logs location (not committed)
- Run date(s): 2026-03-20
- Machine: ubuntu
- Absolute log directory:
  - `/home/ubuntu/linear-attn/`  
- Notes on what is inside that directory (naming scheme, runs, etc.):
  scripts and logs of benchmarks

## Environment
- GPU: RTX 3090
- VRAM: 25Gb
- Driver: 580.105.08
- CUDA: 13.0 

## Model
- Model id: `Qwen/Qwen3-4B`
- Dtype / quantization: `bf16`

## Protocol (pinned)
- Spec: `progress/01-model-and-benchmark-spec/spec.md`

### Decoding / sampling params (pin exactly)
- **temperature**: `1.0` (vLLM default, not overridden in the script)
- **top_p**: `1.0` (vLLM default, not overridden in the script)
- **top_k**: `null` (vLLM default, typically `-1` or `null`, meaning disabled)
- **repetition_penalty**: `1.0` (vLLM default, not overridden in the script)
- **max_new_tokens**: = `output_len`
- **seed**: random integer from 0 to 10000, generated via `np.random.randint(0, 10000)` for each benchmark run

## Result files
- Detailed results:
  - `progress/02-baseline-eval/baseline_longbench_v2.csv`

## Results overview

### Throughput (vLLM)
| input_len | output_len | decode tok/s | total tok/s | req/s | peak mem (GiB) | OOM | notes |
|----------:|-----------:|-------------:|------------:|------:|---------------:|:---:|------|
| 4k        | 128        | 6263.0       | 6263.0      | 5.44  | 22.60         | 0   |      |
| 4k        | 512        | 6272.2       | 6272.2      | 5.44  | 22.64         | 0   |      |
| 4k        | 2048       | 6266.7       | 6266.7      | 5.44  | 22.60         | 0   |      |
| 8k        | 128        | 5705.3       | 5705.3      | 4.95  | 22.60         | 0   |      |
| 8k        | 512        | 5703.3       | 5703.3      | 4.95  | 22.60         | 0   |      |
| 8k        | 2048       | 5710.8       | 5710.8      | 4.96  | 22.60         | 0   |      |
| 16k       | 128        | 4578.0       | 4578.0      | 3.97  | 22.60         | 0   |      |
| 16k       | 512        | 4564.4       | 4564.4      | 3.96  | 22.60         | 0   |      |
| 16k       | 2048       | 4580.1       | 4580.1      | 3.98  | 22.60         | 0   |      |
| 33k       | 128        | 3355.2       | 3355.2      | 2.91  | 22.34         | 0   |      |
| 33k       | 512        | 3359.8       | 3359.8      | 2.92  | 22.34         | 0   |      |
| 33k       | 2048       | 3334.1       | 3334.1      | 2.89  | 22.30         | 0   |      |

### NIAH
- Dataset/version: PaulGrahamEssays
- Prompt template / evaluation recipe: 
NEEDLE: The secret code is: XJ9-ALPHA-7428-BETA. Remember this code.
QUESTION: What is the secret code?

#### Summary (aggregate over depths)

| input_len | depth_rel | accuracy | notes |
|----------:|----------:|---------:|------|
| 4k        | 0.00      | 100%     |      |
| 4k        | 0.50      | 100%     |      |
| 4k        | 1.00      | 100%     |      |
| 8k        | 0.00      | 100%     |      |
| 8k        | 0.50      | 100%     |      |
| 8k        | 1.00      | 100%     |      |
| 16k       | 0.00      | 100%     |      |
| 16k       | 0.50      | 100%     |      |
| 16k       | 1.00      | 100%     |      |
| 32k       | 0.00      | 100%     |      |
| 32k       | 0.50      | 100%     |      |
| 32k       | 1.00      | 100%     |      |

### LongBench V2
- Dataset/version: LongBench V2

LongBench V2 consists of multiple subsets (e.g. `narrativeqa`, `qasper`, ...). Each sample belongs to:
- a **subset**
- a **task type** (one of: single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, code completion)
- a **prompt length bin** (by `prompt_len_tok`): 0-8k, 8k-16k, 16k-24k, 24k-32k

#### Summary (overall)
Fill this table with the corresponding scores by aggregating scores from `baseline_longbench_v2.csv` grouped by `task_type` (rows) and `prompt_len_bin` (columns)

| task_type | 0-8k | 8k-16k | 16k+ | overall | notes |
|----------|---:|---:|---:|----:|------|
| single-document QA | 47.3% | 16.9% | 37.5% | 32.6% | Long = 16k+ |
| multi-document QA  | 29.3% | 25.0% | 30.4% | 28.0% | Long = 16k+ |
| code repository understanding | 33.3% | 33.3% | 34.5% | 34.0% | Short/Medium/Long по домену |
| long in-context learning | 41.7% | 27.9% | 30.8% | 30.9% | Short/Medium/Long по домену |
| long structured data understanding | 25.0% | 26.1% | 16.7% | 24.2% | Short/Medium/Long по домену |
| long-dialogue history understanding | 35.0% | 26.3% | 0.0% | 30.8% | Short/Medium/Long по домену |
| __overall__ | 38.3% | 23.3% | 32.4% | 30.6% | Qwen3-4B |

## Notes / anomalies
- OOMs (where/why): no recorded
- Performance cliffs (where/why): -
- Any deviations from spec: -
- Anything that may affect comparability (batch size caps, fallback kernels, etc.): -

## Repro pointers
High-level pointers only (raw commands/scripts live elsewhere).
- Throughput script entrypoint: vllm_bench.py
- NIAH eval entrypoint: run_niah_test.py
- LongBench V2 eval entrypoint: get_longbench_metrics.py
- How to regenerate tables from the three CSVs:
  1. `vllm_bench.py` is run with Python; you just need to install the dependencies.
  2. `run_niah_test.py` needs to be placed in the official NIAH repository folder, as it requires files from there.
  3. LongBenchV2 was run using the official repository.