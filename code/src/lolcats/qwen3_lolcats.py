#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 LoLCATs-style Linearization
=================================

LoLCATs (Linearizing LLMs with attention transfer + sliding window):
  Non-retrieval heads:
    out = α · sliding_window_softmax(Q, K, V)
        + (1-α) · gated_linear_attn(φ_q(Q), φ_k(K), V, g)

  Retrieval heads (from compressible_heads.pt):
    out = full causal softmax attention   (unchanged)

Trainable additions per attention module (everything else stays frozen):
  • phi_q, phi_k  — per-head feature maps:  softmax(W_h · x_h, dim=-1)
                    init: W_h = I  ⇒  feature map ≈ softmax of raw head
  • g_proj        — Linear(hidden → num_heads*head_dim)
                    GLA forget gates via log_sigmoid; init bias large ⇒
                    gates ≈ 1 ⇒ near-identity decay at start
  • mix_logit     — per-head scalar.  α = sigmoid(mix_logit).
                    init = +5 ⇒ α ≈ 0.99 ⇒ at start the layer is essentially
                    sliding window only.  Training learns how much linear
                    long-range contribution to mix in.

Attention transfer (LoLCATs Stage 1):
  • For every layer, compute the original full-softmax attention output once
    and the patched hybrid output once, on the SAME hidden_states.
  • Loss = mean MSE over layers (before o_proj).
  • Backprop trains only phi_q, phi_k, g_proj, mix_logit.

After 500-2000 steps on ~30-100 M tokens the patched model recovers most of
the original behaviour on short context.  For long-context recovery, extend
training and consider a brief LoRA / full-attention-output fine-tune
(LoLCATs Stage 2) on next-token prediction.

Requires:
  pip install flash-attn --no-build-isolation
  pip install flash-linear-attention
"""

import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Set, Iterable

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention, apply_rotary_pos_emb, repeat_kv,
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

try:
    from fla.ops.gla import chunk_gla, fused_recurrent_gla
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


# ------------------------------------------------------------------------
# fla version compatibility — older versions don't have ``head_first``
# ------------------------------------------------------------------------
if HAS_FLA:
    _GLA_CHUNK_HF = 'head_first' in inspect.signature(chunk_gla).parameters
    _GLA_REC_HF = 'head_first' in inspect.signature(fused_recurrent_gla).parameters
else:
    _GLA_CHUNK_HF = False
    _GLA_REC_HF = False


def _gla_call(fn, has_head_first, q, k, v, g, scale, initial_state):
    """Canonical [B, T, H, D] in/out, version-agnostic."""
    if has_head_first:
        return fn(q, k, v, g, scale=scale, initial_state=initial_state,
                  output_final_state=True, head_first=False)
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    gh = g.transpose(1, 2).contiguous()
    out, state = fn(qh, kh, vh, gh, scale=scale, initial_state=initial_state,
                    output_final_state=True)
    return out.transpose(1, 2).contiguous(), state


# ============================================================================
# Feature map: per-head learnable Linear + softmax  (LoLCATs / Hedgehog style)
# ============================================================================

class PerHeadFeatureMap(nn.Module):
    """
    out_h = softmax(W_h · x_h, dim=-1).

    Initialised to W_h = I so the patched model starts close to using raw
    head activations under a softmax — gradient signal in attention transfer
    moves W_h toward maps whose dot-product approximates exp(QKᵀ).
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        eye = torch.eye(head_dim).unsqueeze(0).expand(num_heads, -1, -1).clone()
        self.weight = nn.Parameter(eye)   # [H, D, D]

    def forward(
        self, x: torch.Tensor, head_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # x: [B, T, H', D] where H' = len(head_idx) if provided else num_heads
        w = self.weight if head_idx is None else self.weight[head_idx]
        out = torch.einsum('bthd,hde->bthe', x, w)
        return F.softmax(out, dim=-1)


# ============================================================================
# Cache (KV for retrieval/SW + GLA recurrent state for linear path)
# ============================================================================

class LoLCATsCache(DynamicCache):
    """
    Holds:
      • full K, V (used by retrieval and SW paths) via DynamicCache base
      • GLA recurrent state per layer for the linear contribution
    """

    def __init__(self, lc_heads_per_layer: Dict[int, List[int]]):
        super().__init__()
        self.lc_heads_per_layer = lc_heads_per_layer
        self.gla_state: Dict[int, torch.Tensor] = {}

    def get_gla_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.gla_state.get(layer_idx)

    def set_gla_state(self, layer_idx: int, state: torch.Tensor) -> None:
        self.gla_state[layer_idx] = state


# ============================================================================
# Hybrid attention kernel
# ============================================================================

def selective_lolcats_attention(
    module,
    query_states: torch.Tensor,      # [B, H_q, q_len, D]
    full_key_states: torch.Tensor,   # [B, H_kv, kv_len, D]  post-cache
    full_value_states: torch.Tensor,
    new_key_states: torch.Tensor,    # [B, H_kv, q_len, D]   pre-cache  (for GLA)
    new_value_states: torch.Tensor,
    new_gate_log: torch.Tensor,      # [B, H_q, q_len, D]
    initial_state: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (output, new_gla_state).  Output layout: [B, q_len, H_q, D].

    Strategy (minimises peak memory):
      1. Sliding window on ALL Q-heads with NATIVE GQA — O(seq · W) memory.
         This produces the SW output used by both LC heads and (unused)
         retrieval heads.
      2. Overwrite retrieval-head outputs with full attention.
      3. Mix LC-head outputs with the GLA contribution (gathered Q, expanded K/V).
    """
    keep_heads: Set[int] = getattr(module, '_lc_keep_heads', set())
    window_size: int = getattr(module, '_lc_window_size', 256)

    bsz, num_q_heads, q_len, head_dim = query_states.shape
    num_kv_heads = full_key_states.shape[1]
    group = num_q_heads // num_kv_heads

    orig_dtype = query_states.dtype
    fa_dtype = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype

    keep_heads = set(h for h in keep_heads if h < num_q_heads)
    full_heads = sorted(keep_heads)
    lc_heads = sorted(set(range(num_q_heads)) - keep_heads)

    # [B, T, H, D] for flash_attn / fla
    q = query_states.transpose(1, 2).to(fa_dtype)
    fk = full_key_states.transpose(1, 2).to(fa_dtype)
    fv = full_value_states.transpose(1, 2).to(fa_dtype)
    nk = new_key_states.transpose(1, 2).to(fa_dtype)
    nv = new_value_states.transpose(1, 2).to(fa_dtype)

    # --- Step 1: sliding window on ALL heads (native GQA) ---
    output = flash_attn_func(
        q, fk, fv,
        dropout_p=dropout, softmax_scale=scaling,
        causal=True, window_size=(window_size, 0),
    )    # [B, q_len, H_q, D]
    # flash_attn_func returns a view from a custom autograd Function; cloning
    # breaks the view chain so the following in-place head-slice writes are
    # allowed when backward is enabled.
    output = output.clone()

    # --- Step 2: overwrite retrieval heads with full attention ---
    if full_heads:
        full_kv_heads = sorted(set(h // group for h in full_heads))
        expected = [kv * group + qi
                    for kv in full_kv_heads for qi in range(group)]
        if sorted(expected) == full_heads:
            k_full = fk[:, :, full_kv_heads, :].contiguous()
            v_full = fv[:, :, full_kv_heads, :].contiguous()
        else:
            k_full = fk.repeat_interleave(group, dim=2)[:, :, full_heads, :].contiguous()
            v_full = fv.repeat_interleave(group, dim=2)[:, :, full_heads, :].contiguous()
        q_full = q[:, :, full_heads, :].contiguous()

        out_full = flash_attn_func(
            q_full, k_full, v_full,
            dropout_p=dropout, softmax_scale=scaling,
            causal=True, window_size=(-1, -1),
        )
        output[:, :, full_heads, :] = out_full
        del q_full, k_full, v_full, out_full

    # --- Step 3: mix SW + GLA for LC heads ---
    new_gla_state: Optional[torch.Tensor] = None
    if lc_heads:
        phi_q: PerHeadFeatureMap = module.phi_q
        phi_k: PerHeadFeatureMap = module.phi_k
        mix_logit: torch.Tensor = module.mix_logit  # [H_q]

        # Gather Q for LC heads (small)
        q_lc = q[:, :, lc_heads, :].contiguous()
        # New K, V expanded per Q-head for GLA
        nk_exp = nk.repeat_interleave(group, dim=2)
        nv_exp = nv.repeat_interleave(group, dim=2)
        k_lc_new = nk_exp[:, :, lc_heads, :].contiguous()
        v_lc_new = nv_exp[:, :, lc_heads, :].contiguous()
        del nk_exp, nv_exp

        # Feature maps (fp32 for softmax precision, then back to bf16).
        # Pass lc_heads so each module slices its per-head weight to match.
        q_phi = phi_q(q_lc.float(), head_idx=lc_heads).to(fa_dtype)
        k_phi = phi_k(k_lc_new.float(), head_idx=lc_heads).to(fa_dtype)
        g_lc = new_gate_log.transpose(1, 2)[:, :, lc_heads, :].contiguous().float()

        # GLA: chunk in parallel, fused_recurrent for single-step decode
        use_recurrent = (q_len == 1 and initial_state is not None)
        gla_fn = fused_recurrent_gla if use_recurrent else chunk_gla
        gla_hf = _GLA_REC_HF if use_recurrent else _GLA_CHUNK_HF

        out_lin, new_gla_state = _gla_call(
            gla_fn, gla_hf,
            q_phi, k_phi, v_lc_new, g_lc,
            scale=1.0,           # feature maps already produce normalised values
            initial_state=initial_state,
        )
        out_lin = out_lin.to(fa_dtype)

        # SW output for LC heads was computed in Step 1 → just index
        out_sw_lc = output[:, :, lc_heads, :]

        alpha = torch.sigmoid(mix_logit[lc_heads]).view(1, 1, -1, 1).to(fa_dtype)
        out_lc = alpha * out_sw_lc + (1.0 - alpha) * out_lin
        output[:, :, lc_heads, :] = out_lc

        del q_lc, k_lc_new, v_lc_new, q_phi, k_phi, g_lc, out_lin, out_lc

    return output.to(orig_dtype), new_gla_state


# ============================================================================
# Patcher
# ============================================================================

class Qwen3LoLCATsPatcher:
    """
    Patch Qwen3 attention with the LoLCATs hybrid kernel and add the small
    set of trainable modules (feature maps, gates, mix gates).
    """

    def __init__(
        self,
        model,
        window_size: int = 256,
        gate_init_bias: float = 5.0,
        mix_init_logit: float = 5.0,
    ):
        if not HAS_FLASH_ATTN:
            raise ImportError(
                "flash_attn required.  pip install flash-attn --no-build-isolation"
            )
        if not HAS_FLA:
            raise ImportError(
                "flash-linear-attention required.  pip install flash-linear-attention"
            )
        if not HAS_DYNAMIC_CACHE:
            raise ImportError("Update transformers — DynamicCache missing.")

        self.model = model
        self.window_size = window_size
        self.gate_init_bias = gate_init_bias
        self.mix_init_logit = mix_init_logit
        self.layer_count = len(model.model.layers)
        self._original_forwards: Dict[int, callable] = {}
        self._lc_heads_per_layer: Dict[int, List[int]] = {}

        # Attention-transfer scratch space — populated by patched_forward when
        # training_mode is on.
        self.training_mode: bool = False
        self.targets: Dict[int, torch.Tensor] = {}
        self.predictions: Dict[int, torch.Tensor] = {}

        print(f"Qwen3LoLCATsPatcher: {self.layer_count} layers, "
              f"window={window_size}, gate_bias={gate_init_bias}, "
              f"mix_init={mix_init_logit}")

    # ---------- compressible_heads.pt loader ----------

    def load_compressible_heads(self, filepath: str) -> Dict[int, List[int]]:
        print(f"\nLoading compressed heads from {filepath}...")
        results = torch.load(filepath, weights_only=False, map_location='cpu')
        induction = results.get("prefix_matching", {})
        echo = results.get("copying", {})

        keep: Dict[int, List[int]] = {}
        for layer_idx, heads in induction.items():
            keep[layer_idx] = keep.get(layer_idx, []) + heads
        for layer_idx, heads in echo.items():
            keep[layer_idx] = keep.get(layer_idx, []) + heads

        total_keep = sum(len(h) for h in keep.values())
        total = self.layer_count * self.model.config.num_attention_heads
        print(f"  Retrieval (full-attn) heads: {total_keep}/{total} "
              f"({100*total_keep/total:.1f}%)")
        print(f"  Remaining {total - total_keep} heads → LoLCATs hybrid (SW + GLA)")
        return keep

    # ---------- new trainable modules ----------

    def _attach_lc_modules(self, attn):
        cfg = attn.config
        hidden_size = cfg.hidden_size
        num_heads = cfg.num_attention_heads
        head_dim = attn.head_dim
        ref = attn.q_proj.weight
        dev, dt = ref.device, ref.dtype

        # Feature maps stay in fp32: per-head matrices are tiny (D×D) and the
        # softmax inside them benefits from fp32 stability.  The forward
        # already casts inputs to fp32 before einsum.
        attn.phi_q = PerHeadFeatureMap(num_heads, head_dim).to(device=dev)
        attn.phi_k = PerHeadFeatureMap(num_heads, head_dim).to(device=dev)

        g_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        nn.init.zeros_(g_proj.weight)
        nn.init.constant_(g_proj.bias, self.gate_init_bias)
        attn.g_proj = g_proj.to(device=dev, dtype=dt)

        # Per-head scalar mix logit.
        mix = torch.full((num_heads,), self.mix_init_logit, device=dev, dtype=dt)
        attn.mix_logit = nn.Parameter(mix)

    # ---------- patch ----------

    def patch_model(self, keep_heads: Dict[int, List[int]]) -> torch.nn.Module:
        print(f"\nPatching {self.layer_count} layers...")
        self._lc_heads_per_layer = {}

        for layer_idx in range(self.layer_count):
            attn = self.model.model.layers[layer_idx].self_attn
            num_heads = attn.config.num_attention_heads

            raw_keep = set(keep_heads.get(layer_idx, []))
            layer_keep = set(h for h in raw_keep if h < num_heads)
            oob = raw_keep - layer_keep
            if oob:
                print(f"  Warning: Layer {layer_idx}: dropped OOB heads {sorted(oob)}")
            layer_lc = [i for i in range(num_heads) if i not in layer_keep]

            attn._lc_keep_heads = layer_keep
            attn._lc_window_size = self.window_size
            self._attach_lc_modules(attn)

            self._lc_heads_per_layer[layer_idx] = layer_lc
            self._original_forwards[layer_idx] = attn.forward
            attn.forward = self._make_patched_forward(attn.forward)

            tag = (f"{len(layer_keep)} full / {len(layer_lc)} hybrid"
                   if layer_keep else f"all {num_heads} hybrid")
            print(f"  Layer {layer_idx:2d}: {tag}")

        print("✓ Patching complete!")
        return self.model

    def _make_patched_forward(self, original_forward):
        patcher = self

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
            num_q_heads = self_attn.config.num_attention_heads
            head_dim = self_attn.head_dim
            bsz, q_len = input_shape

            # ----- Q, K, V (frozen projections, but autograd still flows) -----
            query_states = self_attn.q_norm(
                self_attn.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self_attn.k_norm(
                self_attn.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # ----- Gate logits -----
            gate_raw = self_attn.g_proj(hidden_states)
            gate_raw = gate_raw.view(*input_shape, num_q_heads, head_dim)
            gate_log = F.logsigmoid(gate_raw).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # Save pre-cache new K, V for the GLA path
            new_key_states = key_states
            new_value_states = value_states

            gla_init: Optional[torch.Tensor] = None
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self_attn.layer_idx
                )
                if isinstance(past_key_values, LoLCATsCache):
                    gla_init = past_key_values.get_gla_state(self_attn.layer_idx)

            # ----- Attention transfer: compute the target (full softmax) -----
            if patcher.training_mode:
                with torch.no_grad():
                    q_fa = query_states.transpose(1, 2).to(torch.bfloat16)
                    k_fa = key_states.transpose(1, 2).to(torch.bfloat16)
                    v_fa = value_states.transpose(1, 2).to(torch.bfloat16)
                    tgt = flash_attn_func(
                        q_fa, k_fa, v_fa,
                        dropout_p=0.0, softmax_scale=self_attn.scaling,
                        causal=True, window_size=(-1, -1),
                    )    # [B, q_len, H_q, D]
                    tgt_flat = tgt.reshape(bsz, q_len, -1).contiguous()
                # Keep in bf16 to halve activation memory; MSE casts to fp32.
                patcher.targets[self_attn.layer_idx] = tgt_flat.detach()

            # ----- Patched (LoLCATs hybrid) -----
            attn_output, new_gla_state = selective_lolcats_attention(
                self_attn,
                query_states, key_states, value_states,
                new_key_states, new_value_states,
                gate_log,
                initial_state=gla_init,
                dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                scaling=self_attn.scaling,
            )

            if (past_key_values is not None
                    and isinstance(past_key_values, LoLCATsCache)
                    and new_gla_state is not None):
                past_key_values.set_gla_state(self_attn.layer_idx, new_gla_state)

            attn_output_flat = attn_output.reshape(bsz, q_len, -1).contiguous()

            if patcher.training_mode:
                # store with grad — used for MSE (bf16 to save memory)
                patcher.predictions[self_attn.layer_idx] = attn_output_flat
                # Teacher forcing: forward the (no_grad) softmax target
                # downstream so each subsequent layer sees the same hidden
                # state it would under the original model.  Without this,
                # patched outputs cascade through the stack: per-layer
                # targets drift off-distribution as training updates the
                # earlier layers, MSE plateaus then explodes (visible when
                # mix_init_logit is low, i.e. the linear branch contributes
                # meaningfully from step 0).  Each layer is therefore
                # trained INDEPENDENTLY on correct inputs.
                with torch.no_grad():
                    out = self_attn.o_proj(patcher.targets[self_attn.layer_idx])
                return out, None

            return self_attn.o_proj(attn_output_flat), None

        return patched_forward

    # ---------- cache & params ----------

    def make_cache(self) -> "LoLCATsCache":
        return LoLCATsCache(lc_heads_per_layer=self._lc_heads_per_layer)

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "phi_q"):
                params.extend(attn.phi_q.parameters())
                params.extend(attn.phi_k.parameters())
                params.extend(attn.g_proj.parameters())
                params.append(attn.mix_logit)
        return params

    # ---------- LoLCATs Stage-1: attention transfer ----------

    def attention_transfer_train(
        self,
        batches: Iterable[torch.Tensor],
        steps: int = 500,
        lr: float = 3e-4,
        warmup_steps: int = 50,
        log_every: int = 25,
        grad_clip: float = 1.0,
    ):
        """
        Stage-1 attention transfer.

        ``batches`` is an iterable yielding ``input_ids`` tensors of shape
        ``[B, T]`` on the model's device.  For each step we run one forward
        pass; the patched forward simultaneously computes the original
        full-softmax attention output (no_grad) and the patched hybrid output,
        per layer, before o_proj.  Loss is mean MSE over layers.

        Only ``trainable_parameters()`` receive gradients; everything else is
        frozen for the duration of this call.
        """
        # 1. Freeze the original model; un-freeze the new modules
        original_grad: Dict[int, bool] = {}
        for i, p in enumerate(self.model.parameters()):
            original_grad[i] = p.requires_grad
            p.requires_grad = False
        trainable = self.trainable_parameters()
        for p in trainable:
            p.requires_grad = True

        # 2. Setup
        self.training_mode = True
        self.model.eval()    # disable dropout, etc.
        opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

        def lr_at(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return lr * (step + 1) / warmup_steps
            return lr

        it = iter(batches)
        running = 0.0
        for step in range(steps):
            try:
                input_ids = next(it)
            except StopIteration:
                it = iter(batches)
                input_ids = next(it)

            self.targets.clear()
            self.predictions.clear()

            # Forward — populates targets (no_grad) & predictions (with grad)
            _ = self.model(input_ids, use_cache=False)

            # Loss (cast to fp32 inside MSE for numerical stability)
            losses = []
            for li, tgt in self.targets.items():
                pred = self.predictions[li]
                losses.append(F.mse_loss(pred.float(), tgt.float()))
            loss = torch.stack(losses).mean()

            # Apply linear warmup
            current_lr = lr_at(step)
            for pg in opt.param_groups:
                pg['lr'] = current_lr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()

            running = 0.9 * running + 0.1 * loss.item() if step > 0 else loss.item()
            if step % log_every == 0 or step == steps - 1:
                print(f"  step {step:4d} | mse={loss.item():.5f} | "
                      f"ema={running:.5f}")

        # 3. Restore freeze state
        self.training_mode = False
        self.targets.clear()
        self.predictions.clear()
        for i, p in enumerate(self.model.parameters()):
            p.requires_grad = original_grad.get(i, True)
        self.model.train()
        print("✓ Attention transfer done.")

    # ---------- save / load trainable weights ----------

    def save_weights(self, filepath: str, keep_heads: Optional[Dict[int, List[int]]] = None):
        """
        Save the LoLCATs trainable modules (phi_q, phi_k, g_proj, mix_logit)
        plus enough config to re-construct the patcher on load.

        ``keep_heads`` — the dict you passed to ``patch_model``.  Stored so
        ``load_weights`` can verify the patch was applied with the same
        head split.
        """
        state: Dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if not hasattr(attn, "phi_q"):
                continue
            prefix = f"layer.{layer_idx}"
            state[f"{prefix}.phi_q.weight"] = attn.phi_q.weight.detach().cpu()
            state[f"{prefix}.phi_k.weight"] = attn.phi_k.weight.detach().cpu()
            state[f"{prefix}.g_proj.weight"] = attn.g_proj.weight.detach().cpu()
            state[f"{prefix}.g_proj.bias"] = attn.g_proj.bias.detach().cpu()
            state[f"{prefix}.mix_logit"] = attn.mix_logit.detach().cpu()

        payload = {
            "config": {
                "window_size": self.window_size,
                "gate_init_bias": self.gate_init_bias,
                "mix_init_logit": self.mix_init_logit,
                "layer_count": self.layer_count,
                "num_attention_heads": self.model.config.num_attention_heads,
                "head_dim": getattr(
                    self.model.config, "head_dim",
                    self.model.config.hidden_size // self.model.config.num_attention_heads,
                ),
            },
            "weights": state,
            "keep_heads": (
                {str(k): list(v) for k, v in keep_heads.items()}
                if keep_heads is not None else None
            ),
        }
        torch.save(payload, filepath)
        print(f"✓ Saved LoLCATs weights → {filepath} "
              f"({len(state)} tensors, {sum(t.numel() for t in state.values()):,} params)")

    def load_weights(self, filepath: str) -> Optional[Dict[int, List[int]]]:
        """
        Load LoLCATs trainable params into a model that has already been
        patched.  Returns the saved ``keep_heads`` dict (or None) so the
        caller can sanity-check it against what they passed to ``patch_model``.
        """
        payload = torch.load(filepath, weights_only=False, map_location="cpu")
        state = payload["weights"]
        cfg = payload.get("config", {})

        if cfg.get("layer_count", self.layer_count) != self.layer_count:
            print(f"  Warning: saved layer_count={cfg.get('layer_count')} vs "
                  f"current {self.layer_count}")

        loaded = 0
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if not hasattr(attn, "phi_q"):
                continue
            prefix = f"layer.{layer_idx}"
            if f"{prefix}.phi_q.weight" not in state:
                continue
            ref = attn.q_proj.weight
            dev, dt = ref.device, ref.dtype
            # phi_q/phi_k stay in fp32 (matches _attach_lc_modules)
            attn.phi_q.weight.data.copy_(state[f"{prefix}.phi_q.weight"].to(device=dev))
            attn.phi_k.weight.data.copy_(state[f"{prefix}.phi_k.weight"].to(device=dev))
            attn.g_proj.weight.data.copy_(
                state[f"{prefix}.g_proj.weight"].to(device=dev, dtype=dt))
            attn.g_proj.bias.data.copy_(
                state[f"{prefix}.g_proj.bias"].to(device=dev, dtype=dt))
            attn.mix_logit.data.copy_(
                state[f"{prefix}.mix_logit"].to(device=dev, dtype=dt))
            loaded += 1

        print(f"✓ Loaded LoLCATs weights from {filepath} ({loaded} layers)")
        saved_keep = payload.get("keep_heads")
        if saved_keep is not None:
            saved_keep = {int(k): list(v) for k, v in saved_keep.items()}
        return saved_keep

    # ---------- cleanup ----------

    def unpatch_model(self):
        for layer_idx, original_forward in self._original_forwards.items():
            attn = self.model.model.layers[layer_idx].self_attn
            attn.forward = original_forward
            for name in ("phi_q", "phi_k", "g_proj", "mix_logit",
                         "_lc_keep_heads", "_lc_window_size"):
                if hasattr(attn, name):
                    delattr(attn, name)
        print("✓ Model restored to original state")


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 80)
    print("Qwen3 LoLCATs-style Linearization Test")
    print("=" * 80)

    MODEL_PATH = "Qwen/Qwen3-4B"
    DEVICE = "cuda"
    SEQ_LEN = 512
    BATCH_SIZE = 2
    STEPS = 200

    print(f"\nLoading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )

    test_prompts = ["The capital of France is", "2 + 2 ="]

    def run_prompts(m, label):
        print(f"\n--- {label} ---")
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = m.generate(**inputs, max_new_tokens=10, do_sample=False)
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            print(f"  {prompt!r} → {text!r}")

    # 1. Baseline
    run_prompts(model, "Original Model")

    # 2. Patch
    print("\n" + "─" * 60)
    print("Patch: retrieval heads full attention, others → LoLCATs hybrid")
    print("─" * 60)
    patcher = Qwen3LoLCATsPatcher(
        model, window_size=256, gate_init_bias=5.0, mix_init_logit=5.0,
    )

    heads_file = "compressible_heads_0.pt"
    if not os.path.exists(heads_file):
        print(f"  {heads_file} not found — using all heads as retrieval.")
        num_layers = len(model.model.layers)
        num_heads = model.config.num_attention_heads
        keep_heads = {l: list(range(num_heads)) for l in range(num_layers)}
    else:
        keep_heads = patcher.load_compressible_heads(heads_file)

    patcher.patch_model(keep_heads)

    # 3. Before training — sanity (should be coherent because mix≈SW at init)
    run_prompts(model, "Patched, pre-train (mix ≈ pure SW at init)")

    # 4. Attention transfer on synthetic batches
    # NOTE: For real use, replace with a proper text dataloader (FineWeb-Edu,
    # SlimPajama, …) tokenised to SEQ_LEN.  Random IDs work for a smoke test
    # because the *target* is the original model's own attention output —
    # nothing else needs to be "right" in the input.
    print("\n" + "─" * 60)
    print(f"Attention transfer ({STEPS} steps, seq_len={SEQ_LEN}, bs={BATCH_SIZE})")
    print("─" * 60)
    vocab = model.config.vocab_size

    def random_batches():
        while True:
            yield torch.randint(0, vocab, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    patcher.attention_transfer_train(
        random_batches(),
        steps=STEPS,
        lr=1e-3,
        log_every=25,
    )

    # 5. After training
    run_prompts(model, "Patched, post-train")

    patcher.unpatch_model()
