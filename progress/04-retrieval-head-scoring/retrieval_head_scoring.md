# Task 04 — Retrieval-Head Scoring Report

## Masks (not committed)

Masks are stored externally at `/home/ubuntu/linear-attn/razor_attn/`.

| File | Configuration | Description |
|------|--------------|-------------|
| `/home/ubuntu/linear-attn/razor_attn/retrieval_mask_default.npy` | 14% induction + 1% echo | Default RazorAttention thresholds |
| `/home/ubuntu/linear-attn/razor_attn/retrieval_mask_medium.npy` | 30% induction + 2% echo | More conservative |
| `/home/ubuntu/linear-attn/razor_attn/retrieval_mask_safe.npy` | 46% induction + 4% echo | Most conservative |

## Array format

Each mask is a **1D numpy boolean array** of shape `(num_layers * num_query_heads,)` = `(1152,)` for Qwen3-4B (36 layers × 32 Q-heads).

- Index `h = layer_idx * 32 + head_idx` (layer-major).
- `True` = this head is a retrieval head (must keep full attention).
- `False` = this head is non-retrieval (can be linearized / compressed).

**GQA handling**: Qwen3-4B uses grouped query attention (fewer KV heads than query heads). A KV group is marked as retrieval if **any** of its constituent query heads is marked retrieval. The implementation must propagate the per-query-head mask to per-KV-group before applying compression.

## NIAH Results

Each row reports accuracy for the **selective** (retrieval-head) mask and the **random** mask of the same size (ablation per spec §5).

### Default (14% induction + 1% echo)

| input_len | depth_rel | selective | random | notes |
|----------:|----------:|----------:|-------:|------|
| 4k        | 0.00      | 100%      |   0%   |      |
| 4k        | 0.50      | 100%      |   0%   |      |
| 4k        | 1.00      | 100%      | 100%   |      |
| 8k        | 0.00      | 100%      |   0%   |      |
| 8k        | 0.50      | 100%      |   0%   |      |
| 8k        | 1.00      | 100%      | 100%   |      |
| 16k       | 0.00      |   0%      |   0%   | selective fails at depth=0 |
| 16k       | 0.50      | 100%      |   0%   |      |
| 16k       | 1.00      | 100%      | 100%   |      |
| 32k       | 0.00      |   0%      |   0%   | selective fails at depth=0 |
| 32k       | 0.50      | 100%      |   0%   |      |
| 32k       | 1.00      | 100%      | 100%   |      |

### Medium (30% induction + 2% echo)

| input_len | depth_rel | selective | random | notes |
|----------:|----------:|----------:|-------:|------|
| 4k        | 0.00      | 100%      |   0%   |      |
| 4k        | 0.50      | 100%      |   0%   |      |
| 4k        | 1.00      | 100%      | 100%   |      |
| 8k        | 0.00      | 100%      |   0%   |      |
| 8k        | 0.50      | 100%      |   0%   |      |
| 8k        | 1.00      | 100%      | 100%   |      |
| 16k       | 0.00      | 100%      |   0%   |      |
| 16k       | 0.50      | 100%      |   0%   |      |
| 16k       | 1.00      | 100%      | 100%   |      |
| 32k       | 0.00      | 100%      |   0%   |      |
| 32k       | 0.50      | 100%      |   0%   |      |
| 32k       | 1.00      | 100%      | 100%   |      |

### Safe (46% induction + 4% echo)

| input_len | depth_rel | selective | random | notes |
|----------:|----------:|----------:|-------:|------|
| 4k        | 0.00      | 100%      | 100%   |      |
| 4k        | 0.50      | 100%      | 100%   |      |
| 4k        | 1.00      | 100%      | 100%   |      |
| 8k        | 0.00      | 100%      | 100%   |      |
| 8k        | 0.50      | 100%      | 100%   |      |
| 8k        | 1.00      | 100%      | 100%   |      |
| 16k       | 0.00      | 100%      | 100%   |      |
| 16k       | 0.50      | 100%      | 100%   |      |
| 16k       | 1.00      | 100%      | 100%   |      |
| 32k       | 0.00      | 100%      | 100%   |      |
| 32k       | 0.50      | 100%      | 100%   |      |
| 32k       | 1.00      | 100%      | 100%   |      |

## Summary

Which configurations meet the ≥ 95% criterion (selective) at every (input_len, depth_rel):

- **default**: ✗ — fails at 16k/32k, depth=0 (0% accuracy). 10/12 cells pass.
- **medium**: ✓ — 100% at every cell.
- **safe**: ✓ — 100% at every cell, but the random ablation also scores 100% everywhere, so head selection is not informative at this budget (≈50% of heads protected ⇒ enough capacity that any subset suffices).

Random-ablation insight: for **default** and **medium**, the random baseline fails on depth ∈ {0, 0.5} (the needle is outside the recent-window region), confirming that the selected retrieval heads carry the long-range information. For **safe**, random matches selective, meaning the budget is too generous to discriminate.

**Recommended configuration for downstream tasks: `medium` (30% induction + 2% echo).** It is the smallest mask that meets the criterion at every (input_len, depth_rel) and is meaningfully better than random — i.e., the retrieval heads are doing real work, and we are not wasting capacity as in `safe`.
