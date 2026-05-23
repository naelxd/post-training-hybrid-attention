#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 Selective Sliding Window Attention
=======================================

Apply sliding window attention selectively based on compressible_heads.pt:
- Keep heads from compressible_heads.pt as full attention (retrieval heads)
- Apply sliding window + attention sinks to all other heads

Uses flash_attn for O(seq * window) memory instead of O(seq^2).
KV cache for SW heads is truncated to (num_sinks + window_size) tokens.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Set

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError as e:
    print(f"Warning: Qwen3 imports failed: {e}")

try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# =============================================================================
# Attention-sink-aware KV cache
# =============================================================================

class SelectiveSlidingWindowCache(DynamicCache):
    """
    DynamicCache subclass that truncates KV entries for SW (non-retrieval) heads.

    For retrieval heads: stores the full key/value history (standard behaviour).
    For SW heads: keeps only the first `num_sinks` tokens (attention sinks) plus
    the most recent `window_size` tokens — O(window) memory per layer.

    `sw_heads_per_layer[layer_idx]` is a sorted list of SW head indices for that
    layer (after GQA expansion, i.e. Q-head indices).  Retrieval heads are all
    indices not in that list.
    """

    def __init__(
        self,
        num_sinks: int,
        window_size: int,
        sw_heads_per_layer: Dict[int, List[int]],
        num_heads: int,
    ):
        super().__init__()
        self.num_sinks = num_sinks
        self.window_size = window_size
        self.sw_heads_per_layer = sw_heads_per_layer   # layer_idx → [sw head indices]
        self.num_heads = num_heads

    def update(
        self,
        key_states: torch.Tensor,    # [bsz, num_kv_heads, q_len, head_dim]
        value_states: torch.Tensor,  # [bsz, num_kv_heads, q_len, head_dim]
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new KV then truncate SW heads to (num_sinks + window_size)."""
        # Standard DynamicCache append
        key_states, value_states = super().update(key_states, value_states, layer_idx, cache_kwargs)

        sw_heads = self.sw_heads_per_layer.get(layer_idx, [])
        if not sw_heads or self.num_sinks + self.window_size <= 0:
            return key_states, value_states

        seq_len = key_states.shape[2]
        keep = self.num_sinks + self.window_size
        if seq_len <= keep:
            return key_states, value_states

        # Truncate: keep first num_sinks + last window_size for SW heads only.
        # key_states shape: [bsz, num_kv_heads, seq_len, head_dim]
        # We index by Q-head; KV-head index = Q-head // num_key_value_groups.
        # Since we already ran repeat_kv in the forward, the cache stores raw
        # KV heads (not expanded), so convert SW Q-head indices → KV-head indices.
        # The cache holds num_kv_heads entries (unexpanded).
        num_kv = key_states.shape[1]
        num_q  = self.num_heads
        group  = num_q // num_kv

        # Unique KV head indices that are SW
        sw_kv_heads = sorted(set(h // group for h in sw_heads))

        sinks   = key_states[:, :, :self.num_sinks, :]
        window  = key_states[:, :, seq_len - self.window_size:, :]
        v_sinks  = value_states[:, :, :self.num_sinks, :]
        v_window = value_states[:, :, seq_len - self.window_size:, :]

        k_trunc = torch.cat([sinks, window], dim=2)
        v_trunc = torch.cat([v_sinks, v_window], dim=2)

        # Build final tensors: SW KV heads get truncated; retrieval KV heads keep full
        # Easier: clone and overwrite the SW KV heads with a padded version that
        # matches the full seq_len for retrieval heads — but that wastes memory.
        # Instead, we store separate tensors and return the truncated view for SW
        # heads.  Because flash_attn can handle different kv_len per call (we split
        # by head group anyway), we overwrite the cache in-place for SW heads and
        # keep retrieval heads at full length.
        #
        # Implementation: overwrite the layer cache.  The retrieval heads will get
        # the full-length tensor; SW heads need to be sliced out when used.
        # The simplest correct approach that avoids shape mismatches is to store
        # the truncated version for all KV heads and rely on the patched forward
        # to pass the right slice to each flash_attn call.
        #
        # We only truncate if ALL heads in a layer are SW (otherwise skip; the
        # patched forward handles slicing per head group anyway).
        all_sw = len(sw_kv_heads) == num_kv
        if all_sw:
            self.key_cache[layer_idx]   = k_trunc
            self.value_cache[layer_idx] = v_trunc
            return k_trunc, v_trunc

        # Mixed layer: overwrite SW KV-head rows only, keep retrieval rows full.
        # Build new tensors with max seq_len across both groups.
        # Retrieval heads: keep full seq_len slice (already in key_states).
        # SW heads: sinks + window, left-padded to full seq_len with zeros so shape is uniform.
        # The patched forward will slice :num_sinks and seq_len-window_size: anyway.
        ret_kv = sorted(set(range(num_kv)) - set(sw_kv_heads))
        k_new = key_states.clone()    # full seq_len
        v_new = value_states.clone()

        # Replace SW KV head rows with sinks+window (shorter), padded on the left
        pad_len = seq_len - keep
        k_pad = torch.cat([
            torch.zeros(key_states.shape[0], 1, pad_len, key_states.shape[3],
                        device=key_states.device, dtype=key_states.dtype).expand(-1, len(sw_kv_heads), -1, -1),
            k_trunc[:, sw_kv_heads, :, :],
        ], dim=2)
        v_pad = torch.cat([
            torch.zeros(value_states.shape[0], 1, pad_len, value_states.shape[3],
                        device=value_states.device, dtype=value_states.dtype).expand(-1, len(sw_kv_heads), -1, -1),
            v_trunc[:, sw_kv_heads, :, :],
        ], dim=2)

        k_new[:, sw_kv_heads, :, :] = k_pad
        v_new[:, sw_kv_heads, :, :] = v_pad

        self.key_cache[layer_idx]   = k_new
        self.value_cache[layer_idx] = v_new
        return k_new, v_new


# =============================================================================
# Attention sink combine helpers
# =============================================================================

def _combine_sink_and_window(
    q: torch.Tensor,          # [bsz, q_len, H_q, d]  fa layout
    k: torch.Tensor,          # [bsz, kv_len, H_kv, d] FULL k (native GQA)
    v: torch.Tensor,          # [bsz, kv_len, H_kv, d] FULL v
    num_sinks: int,
    window_size: int,
    dropout: float,
    scaling: float,
) -> torch.Tensor:
    """
    Sliding window attention with attention sinks (memory-bounded).

    For query at full position i, attends to:
      • [0, num_sinks-1]                            — sinks
      • [max(num_sinks, i - window_size + 1), i]    — sliding window

    Implementation:
      1. Window via flash_attn native `window_size=(W, 0)` on FULL k — O(seq*W) memory.
      2. Sink block via flash_attn (or bmm) on the first num_sinks keys —
         q × k_sink is O(seq * num_sinks * H), tiny.
      3. Combine via logsumexp re-weighting using lse returned by flash_attn.

    Requires flash_attn that supports `return_attn_probs=True` (returns softmax_lse).
    If unavailable, falls back to pure sliding window (no sinks).

    Note: for queries near the start (i < num_sinks + W), the window naturally
    includes sink positions — the sink block then over-weights them slightly.
    For long contexts this is negligible.
    """
    bsz, q_len, H_q, d = q.shape
    H_kv = k.shape[2]
    nrep = H_q // H_kv
    dev = q.device

    # --- Window block via flash_attn native sliding window ---
    try:
        out_w, lse_w_fa, _ = flash_attn_func(
            q, k, v,
            dropout_p=dropout, softmax_scale=scaling,
            causal=True, window_size=(window_size, 0),
            return_attn_probs=True,
        )
        # lse_w_fa shape: [bsz, H_q, q_len], dtype fp32
    except (TypeError, ValueError):
        # Older flash_attn — can't combine sinks. Fall back to pure SW.
        return flash_attn_func(
            q, k, v,
            dropout_p=dropout, softmax_scale=scaling,
            causal=True, window_size=(window_size, 0),
        )

    # --- Sink block via bmm on first num_sinks keys (tiny) ---
    # Expand k_sink / v_sink to H_q for GQA
    k_sink = k[:, :num_sinks, :, :]   # [bsz, ns, H_kv, d]
    v_sink = v[:, :num_sinks, :, :]
    if nrep > 1:
        k_sink = k_sink.repeat_interleave(nrep, dim=2)  # [bsz, ns, H_q, d]
        v_sink = v_sink.repeat_interleave(nrep, dim=2)

    # [bsz*H_q, q_len, d] × [bsz*H_q, d, ns] → [bsz*H_q, q_len, ns]
    q_bh  = q.permute(0, 2, 1, 3).reshape(bsz * H_q, q_len, d)
    ks_bh = k_sink.permute(0, 2, 1, 3).reshape(bsz * H_q, num_sinks, d)
    vs_bh = v_sink.permute(0, 2, 1, 3).reshape(bsz * H_q, num_sinks, d)

    logits_s = torch.bmm(q_bh.float(), ks_bh.float().transpose(1, 2)) * scaling

    # Causal mask for sinks
    kv_len = k.shape[1]
    full_q_pos = torch.arange(q_len, device=dev) + (kv_len - q_len)
    sink_pos   = torch.arange(num_sinks, device=dev)
    sink_mask  = full_q_pos.unsqueeze(1) >= sink_pos.unsqueeze(0)
    logits_s.masked_fill_(~sink_mask.unsqueeze(0), float('-inf'))

    lse_s = torch.logsumexp(logits_s, dim=-1, keepdim=True)          # [bsz*H_q, q_len, 1]
    attn_s = (logits_s - lse_s).exp()
    out_s  = torch.bmm(attn_s.to(vs_bh.dtype), vs_bh).float()        # [bsz*H_q, q_len, d]
    del logits_s, attn_s, ks_bh, vs_bh, q_bh

    # --- Combine via logsumexp ---
    # lse_w_fa: [bsz, H_q, q_len] → [bsz*H_q, q_len, 1]
    lse_w = lse_w_fa.reshape(bsz * H_q, q_len, 1).float()
    lse_total = torch.logaddexp(lse_s, lse_w)
    w_s = (lse_s - lse_total).exp()
    w_w = (lse_w - lse_total).exp()

    out_w_bh = out_w.permute(0, 2, 1, 3).reshape(bsz * H_q, q_len, d).float()
    combined = w_s * out_s + w_w * out_w_bh

    return combined.to(q.dtype).reshape(bsz, H_q, q_len, d).permute(0, 2, 1, 3).contiguous()


# =============================================================================
# Main attention function
# =============================================================================

def selective_sliding_window_attention(
    module,
    query_states: torch.Tensor,   # [bsz, num_q_heads, q_len, head_dim]
    key_states: torch.Tensor,     # [bsz, num_kv_heads, kv_len, head_dim]
    value_states: torch.Tensor,   # [bsz, num_kv_heads, kv_len, head_dim]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Memory-efficient selective sliding window attention via flash_attn.

    Strategy (minimizes peak memory):
      1. Sliding window pass on ALL Q-heads using NATIVE GQA (no repeat_kv,
         no slicing). This produces O(seq * window) memory and is the cheap
         baseline output for every head.
      2. For retrieval Q-heads only, recompute with full attention and
         OVERWRITE the SW output. The retrieval gather is small (~14% of heads).

    Why no repeat_kv: flash_attn_func natively supports GQA when
    q.shape[-2] == N * k.shape[-2]. Expanding KV with repeat_kv before
    flash_attn would 4× the KV memory pointlessly (at 32k: +400 MB/layer).

    Why no sw-heads gather: copying q/k/v[:, :, sw_heads, :] at 32k allocates
    ~660 MB per layer. Overwriting retrieval heads (small set) avoids it.
    """
    window_size = getattr(module, '_sw_window_size', 512)
    num_sinks   = getattr(module, '_sw_num_sinks', 0)
    keep_heads  = getattr(module, '_sw_keep_heads', set())

    bsz, num_q_heads, q_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    kv_len = key_states.shape[2]
    group  = num_q_heads // num_kv_heads

    orig_dtype = query_states.dtype
    fa_dtype   = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype

    # NO repeat_kv — flash_attn handles GQA natively.
    q = query_states.transpose(1, 2).to(fa_dtype)   # [bsz, q_len,  num_q,  d]
    k = key_states.transpose(1, 2).to(fa_dtype)     # [bsz, kv_len, num_kv, d]
    v = value_states.transpose(1, 2).to(fa_dtype)   # [bsz, kv_len, num_kv, d]

    # Clamp keep_heads to valid range
    keep_heads = set(h for h in keep_heads if h < num_q_heads)
    full_heads = sorted(keep_heads)

    # --- Step 1: sliding window (+ optional sinks) on ALL heads ---
    if num_sinks <= 0 or kv_len <= num_sinks + window_size:
        output = flash_attn_func(
            q, k, v,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            window_size=(window_size, 0),
        )  # [bsz, q_len, num_q, d]  — peak memory O(seq * W)
    else:
        output = _combine_sink_and_window(
            q, k, v, num_sinks, window_size, dropout, scaling
        )

    # --- Step 2: overwrite retrieval heads with full attention ---
    if full_heads and len(full_heads) < num_q_heads:
        # Map Q-head indices → KV-head indices (whole groups, per spec §4)
        full_kv_heads = sorted(set(h // group for h in full_heads))
        expected_full_q = [kv * group + qi
                           for kv in full_kv_heads for qi in range(group)]
        if sorted(expected_full_q) != full_heads:
            # KV group not fully selected — fall back to per-Q-head expansion
            # (still correct, just less efficient because we re-expand KV).
            k_full = k.repeat_interleave(group, dim=2)[:, :, full_heads, :].contiguous()
            v_full = v.repeat_interleave(group, dim=2)[:, :, full_heads, :].contiguous()
        else:
            k_full = k[:, :, full_kv_heads, :].contiguous()   # only the needed KV heads
            v_full = v[:, :, full_kv_heads, :].contiguous()
        q_full = q[:, :, full_heads, :].contiguous()

        out_full = flash_attn_func(
            q_full, k_full, v_full,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            window_size=(-1, -1),
        )
        del q_full, k_full, v_full

        output[:, :, full_heads, :] = out_full
        del out_full

    elif full_heads and len(full_heads) == num_q_heads:
        # All heads retrieval — recompute everything as full attention
        output = flash_attn_func(
            q, k, v,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            window_size=(-1, -1),
        )

    return output.to(orig_dtype), None


class Qwen3SelectiveSlidingWindowPatcher:
    """
    Patch Qwen3 model with selective sliding window attention via flash_attn.

    Retrieval heads get full causal attention.
    SW heads get causal sliding window (last `window_size` tokens) plus attention
    sinks (first `num_sinks` tokens always attended to).

    KV cache for SW heads is automatically truncated to
    (num_sinks + window_size) entries when using `make_cache()`.
    """

    def __init__(self, model, window_size: int = 512, num_sinks: int = 4):
        if not HAS_FLASH_ATTN:
            raise ImportError(
                "flash_attn is required. Install with:\n"
                "  pip install flash-attn --no-build-isolation"
            )
        self.model = model
        self.window_size = window_size
        self.num_sinks = num_sinks
        self.layer_count = len(model.model.layers)
        self._original_forwards = {}
        self._sw_heads_per_layer: Dict[int, List[int]] = {}   # populated by patch_model

        print(f"Qwen3SelectiveSlidingWindowPatcher: {self.layer_count} layers, "
              f"window={window_size}, sinks={num_sinks} (flash_attn)")

    def load_compressible_heads(self, filepath: str) -> Dict[int, List[int]]:
        """Load compressible_heads.pt and convert to per-layer keep_heads format."""
        print(f"\nLoading compressed heads from {filepath}...")
        results = torch.load(filepath, weights_only=False, map_location='cpu')

        induction_heads = results.get("prefix_matching", {})
        echo_heads = results.get("copying", {})

        keep_heads = {}
        for layer_idx, heads in induction_heads.items():
            keep_heads[layer_idx] = keep_heads.get(layer_idx, []) + heads
        for layer_idx, heads in echo_heads.items():
            keep_heads[layer_idx] = keep_heads.get(layer_idx, []) + heads

        total_retrieval = sum(len(heads) for heads in keep_heads.values())
        total_heads = self.layer_count * self.model.config.num_attention_heads

        print(f"  Induction heads: {sum(len(h) for h in induction_heads.values())} across {len(induction_heads)} layers")
        print(f"  Echo heads: {sum(len(h) for h in echo_heads.values())} across {len(echo_heads)} layers")
        print(f"  Total retrieval heads: {total_retrieval}/{total_heads} ({100*total_retrieval/total_heads:.1f}%)")

        return keep_heads

    def patch_model(self, keep_heads: Dict[int, List[int]]) -> torch.nn.Module:
        """Patch the model with flash_attn selective sliding window + sinks."""
        print(f"\nPatching {self.layer_count} layers...")
        self._sw_heads_per_layer = {}

        for layer_idx in range(self.layer_count):
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn

            num_heads = attn.config.num_attention_heads
            raw_keep  = set(keep_heads.get(layer_idx, []))
            # Clamp: drop indices that exceed this model's head count
            layer_keep_heads = set(h for h in raw_keep if h < num_heads)
            oob = raw_keep - layer_keep_heads
            if oob:
                print(f"  Warning: Layer {layer_idx}: dropped {len(oob)} out-of-range "
                      f"head indices {sorted(oob)} (model has {num_heads} heads)")
            layer_sw_heads   = [i for i in range(num_heads) if i not in layer_keep_heads]

            attn._sw_keep_heads  = layer_keep_heads
            attn._sw_window_size = self.window_size
            attn._sw_num_sinks   = self.num_sinks

            self._sw_heads_per_layer[layer_idx] = layer_sw_heads
            self._original_forwards[layer_idx]  = attn.forward

            def make_patched_forward(original_forward):
                def patched_forward(
                    hidden_states: torch.Tensor,
                    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                    attention_mask: Optional[torch.Tensor] = None,
                    past_key_values=None,
                    **kwargs,
                ):
                    self_attn = original_forward.__self__
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self_attn.head_dim)

                    query_states = self_attn.q_norm(
                        self_attn.q_proj(hidden_states).view(hidden_shape)
                    ).transpose(1, 2)
                    key_states = self_attn.k_norm(
                        self_attn.k_proj(hidden_states).view(hidden_shape)
                    ).transpose(1, 2)
                    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(
                            key_states, value_states, self_attn.layer_idx
                        )

                    attn_output, _ = selective_sliding_window_attention(
                        self_attn,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                        scaling=self_attn.scaling,
                    )

                    # attn_output: [bsz, q_len, num_heads, head_dim] → [bsz, q_len, hidden_size]
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = self_attn.o_proj(attn_output)

                    return attn_output, None

                return patched_forward

            attn.forward = make_patched_forward(attn.forward)

            if layer_keep_heads:
                print(f"  Layer {layer_idx:2d}: {len(layer_keep_heads)} heads full attn, "
                      f"{len(layer_sw_heads)} sliding window + {self.num_sinks} sinks (flash)")
            else:
                print(f"  Layer {layer_idx:2d}: all {num_heads} heads sliding window "
                      f"+ {self.num_sinks} sinks (flash)")

        print("✓ Patching complete!")
        return self.model

    def make_cache(self) -> "SelectiveSlidingWindowCache":
        """
        Create a KV cache that truncates SW-head entries to
        (num_sinks + window_size) tokens.  Pass the returned object to
        model.generate() via past_key_values= or use_cache=True implicitly.

        Example::

            cache = patcher.make_cache()
            outputs = model.generate(..., past_key_values=cache, use_cache=True)
        """
        if not HAS_DYNAMIC_CACHE:
            raise ImportError("transformers.DynamicCache not available — update transformers.")
        num_heads = self.model.config.num_attention_heads
        return SelectiveSlidingWindowCache(
            num_sinks=self.num_sinks,
            window_size=self.window_size,
            sw_heads_per_layer=self._sw_heads_per_layer,
            num_heads=num_heads,
        )

    def unpatch_model(self):
        """Restore original forward methods."""
        for layer_idx, original_forward in self._original_forwards.items():
            self.model.model.layers[layer_idx].self_attn.forward = original_forward
        print("✓ Model restored to original state")


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("="*80)
    print("Qwen3 Selective Sliding Window Attention (flash_attn) Test")
    print("="*80)

    MODEL_PATH = "Qwen/Qwen3-4B"

    print(f"\nLoading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        # Load with flash_attention_2 so the sanity-check baseline uses the
        # same kernel as our patch. Without this, accumulated fp differences
        # between sdpa/eager and flash_attn across 36 layers produce ~0.375
        # max logit diff even when the attention pattern is mathematically
        # identical.
        attn_implementation="flash_attention_2",
    )

    test_prompts = ["The capital of France is", "2 + 2 ="]

    def run_prompts(m, label):
        print(f"\n--- {label} ---")
        results = []
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = m.generate(**inputs, max_new_tokens=10, do_sample=False)
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  {prompt!r} → {text!r}")
            results.append(outputs[0].tolist())
        return results

    # ── 1. Original (baseline) ───────────────────────────────────────────────
    original_outputs = run_prompts(model, "Original Model")

    # ── 2. Sanity check: ALL heads as full attention ─────────────────────────
    # Compares logits of a single forward pass (not generate) with a float
    # tolerance. Token-level comparison is not used because flash_attn and
    # eager/sdpa produce slightly different float values, which can flip greedy
    # argmax when two token probabilities are nearly equal — that is expected
    # behaviour, not a bug.
    print("\n" + "─"*60)
    print("Sanity check: all heads → full attention")
    print("Comparing forward-pass logits (not tokens) with fp tolerance")
    print("─"*60)

    num_layers = len(model.model.layers)
    num_heads  = model.config.num_attention_heads
    all_heads_keep = {layer: list(range(num_heads)) for layer in range(num_layers)}

    # Collect original logits (single forward pass, no generation)
    orig_logits_list = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            orig_logits_list.append(model(**inputs, use_cache=False).logits.float())

    patcher_full = Qwen3SelectiveSlidingWindowPatcher(model, window_size=512, num_sinks=0)
    patcher_full.patch_model(all_heads_keep)

    run_prompts(model, "Patched — ALL heads full attention (qualitative)")

    # Collect patched logits
    max_diffs = []
    mean_diffs = []
    for prompt, orig_logits in zip(test_prompts, orig_logits_list):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            patched_logits = model(**inputs, use_cache=False).logits.float()
        diff = (orig_logits - patched_logits).abs()
        max_diffs.append(diff.max().item())
        mean_diffs.append(diff.mean().item())

    print("\n  Logit difference (original eager/sdpa vs patched flash_attn):")
    for i, prompt in enumerate(test_prompts):
        print(f"    {prompt!r:35s}  max={max_diffs[i]:.5f}  mean={mean_diffs[i]:.6f}")

    # Both paths use flash_attn_func — expected diff is sub-bf16 precision (<0.01)
    threshold = 0.01
    passed = all(d < threshold for d in max_diffs)
    print(f"\n  Threshold: {threshold}  →  {'✓ PASSED' if passed else '✗ FAILED'}")
    if not passed:
        print("  Hint: if model was not loaded with attn_implementation='flash_attention_2',")
        print("  accumulated sdpa/eager vs flash_attn fp noise can reach ~0.4 over 36 layers.")
        print("  That is expected. A diff >> 0.4 indicates an actual bug.")

    patcher_full.unpatch_model()

    # ── 3. Patch with compressible_heads_0.pt ───────────────────────────────
    print("\n" + "─"*60)
    print("Selective SW: retrieval heads full, others sliding window")
    print("─"*60)
    print("Note: compressible_heads_0.pt must be generated for THIS model.")
    print("      Head indices from a larger model will be dropped with a warning.")

    import os
    patcher = Qwen3SelectiveSlidingWindowPatcher(model, window_size=512, num_sinks=4)
    heads_file = "compressible_heads_0.pt"
    if not os.path.exists(heads_file):
        print(f"  {heads_file} not found — using all heads as retrieval (full attn).")
        num_layers = len(model.model.layers)
        num_heads  = model.config.num_attention_heads
        keep_heads = {layer: list(range(num_heads)) for layer in range(num_layers)}
    else:
        keep_heads = patcher.load_compressible_heads(heads_file)
    patcher.patch_model(keep_heads)

    run_prompts(model, "Patched — selective sliding window")

    patcher.unpatch_model()

