"""Convert compressible_heads_{0,1,2}.pt → retrieval_mask_{default,medium,safe}.npy.

Output format (per spec):
  1D numpy bool array, shape (num_layers * num_query_heads,)
  Index h = layer_idx * num_q_heads + head_idx_in_layer (layer-major)
  True  → retrieval head (keep full attention)
  False → non-retrieval (compressible)

GQA: any Q-head True in a KV group ⇒ entire group True.
"""

import numpy as np
import torch

# Qwen3-4B
NUM_LAYERS = 36
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 4

CONFIGS = [
    ("compressible_heads_0.pt", "retrieval_mask_default.npy", "14% induction + 1% echo"),
    ("compressible_heads_1.pt", "retrieval_mask_medium.npy",  "30% induction + 2% echo"),
    ("compressible_heads_2.pt", "retrieval_mask_safe.npy",    "46% induction + 4% echo"),
]


def build_mask(pt_path: str) -> np.ndarray:
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    induction = data.get("prefix_matching", {})
    echo = data.get("copying", {})

    mask = np.zeros(NUM_LAYERS * NUM_Q_HEADS, dtype=bool)

    for layer_dict in (induction, echo):
        for layer_idx, heads in layer_dict.items():
            for h in heads:
                mask[layer_idx * NUM_Q_HEADS + h] = True

    # Propagate to entire KV group: any Q-head True ⇒ all 4 heads in group True
    mask_2d = mask.reshape(NUM_LAYERS, NUM_KV_HEADS, GROUP_SIZE)
    group_any = mask_2d.any(axis=2, keepdims=True)
    mask = np.broadcast_to(group_any, mask_2d.shape).reshape(-1).copy()

    return mask


def main():
    for pt_path, npy_path, desc in CONFIGS:
        mask = build_mask(pt_path)
        np.save(npy_path, mask)
        n_true = int(mask.sum())
        total = mask.size
        print(f"{npy_path:30s}  {desc:30s}  {n_true:4d}/{total} retrieval ({100*n_true/total:.1f}%)")


if __name__ == "__main__":
    main()
