# Task 04 — Retrieval-Head Scoring and Mask Generation

Implement the RazorAttention retrieval-head identification pipeline for Qwen3-4B, following the protocol in `progress/03-razor-protocol/spec.md`.

## Protocol reference

All parameters (golden prompt construction, echo/induction formulas, three threshold configurations) are defined in `progress/03-razor-protocol/spec.md`.

## What "done" means

- [x] `code/src/razor_attn/run_ra.py` exists and is runnable
- [x] Runs the golden prompt (K=2500, 4 repetitions) through Qwen3-4B and extracts per-head attention maps
- [x] Computes per-head echo scores and induction scores
- [x] Generates three retrieval head masks: default (14%/1%), medium (30%/2%), safe (46%/4%)
- [x] Runs NIAH validation (4k/8k/16k/32k, `depth_rel` ∈ {0, 0.5, 1}) for all three configurations and fills results in `retrieval_head_scoring.md`
- [x] Reports which configuration(s) meet the ≥ 95% accuracy criterion

## Required artifacts to save

numpy arrays are **not committed** to the repository. Save them to an external location and provide the path here.

- **Retrieval head masks** (not committed):
  - `/home/ubuntu/linear-attn/razor_attn/retrieval_mask_default.npy` — binary mask (True = retrieval head), 14%/1% config
  - `/home/ubuntu/linear-attn/razor_attn/retrieval_mask_medium.npy` — binary mask, 30%/2% config
  - `/home/ubuntu/linear-attn/razor_attn/retrieval_mask_safe.npy` — binary mask, 46%/4% config
- `progress/04-retrieval-head-scoring/retrieval_head_scoring.md` — results report with masks, NIAH results, and array format documentation
- `progress/04-retrieval-head-scoring/README.md` — this file
