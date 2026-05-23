# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import random

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from security.type import check_type
from security.path import SafeWriteUmask, get_valid_write_path
from security.hook import FunctionReplace
from ra_rope_config import RARopeCompressConfig
from utils.logging import logger

DUMMY_INPUT_LENGTH = 2500   # 10001 tokens; efficient path uses flash_attn, no O(n²) GPU alloc
REPET_TIMES = 4
INPUT_IDS = "input_ids"


class RARopeCompressor(object):

    def __init__(self, model, tokenizer, cfg: RARopeCompressConfig):
        check_type(model, nn.Module, param_name="model")
        check_type(cfg, RARopeCompressConfig, param_name="cfg")
        check_type(tokenizer, PreTrainedTokenizerBase, param_name="tokenizer")

        if not hasattr(model, "config"):
            raise ValueError("Model does not have attribute `config`. \
                              Model must be a huggingface model.")
        if not hasattr(model.config, 'hidden_size'):
            raise ValueError("Model must have a `config` attribute with a `hidden_size` property. \
                              Model must be a huggingface model.")

        # Try multiple parameter names
        num_attention_heads = None
        num_attention_head_names = ['num_attention_heads', 'n_head', 'num_heads']
        for name in num_attention_head_names:
            if hasattr(model.config, name):
                num_attention_heads = getattr(model.config, name)
                break
        if not num_attention_heads:
            raise ValueError(f"Model must have num_attention_heads in config. Tried: {num_attention_head_names}")

        num_key_value_heads = None
        num_key_value_heads_names = ['num_key_value_heads', 'multi_query_group_num']
        for name in num_key_value_heads_names:
            if hasattr(model.config, name):
                num_key_value_heads = getattr(model.config, name)
                break
        if not num_key_value_heads:
            # For models without GQA, use num_attention_heads
            num_key_value_heads = num_attention_heads

        self.model = model
        self.cfg = cfg
        self.hidden_size = self.model.config.hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.num_kv_per_group = self.num_attention_heads / self.num_key_value_heads

        # Debug info
        print(f"DEBUG: Model config - Layers: {getattr(model.config, 'num_hidden_layers', 'N/A')}, "
              f"Attention heads: {num_attention_heads}, KV heads: {num_key_value_heads}, "
              f"Group size: {self.num_kv_per_group}")

        self.tokenizer = tokenizer

    def _sample_vocab_tokens(self, num_tokens):
        """Sample uniformly from the full tokenizer vocabulary (spec §1: vocab-sampled).

        Excludes special tokens (BOS/EOS/PAD/etc.) so the random block carries no
        structural meaning. Returns a plain Python list of token IDs.
        """
        vocab_size = len(self.tokenizer)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        tokens = []
        while len(tokens) < num_tokens:
            t = random.randint(0, vocab_size - 1)
            if t not in special_ids:
                tokens.append(t)
        return tokens

    def select_top_heads_all(self, data, ratio):
        """Select top ratio% of KV groups and return all Q-heads in those groups.

        Matches the "Total retrieval heads" column of spec §4 (Table 2):
        14%/1%/15%, 30%/2%/32%, 46%/4%/50%. The literal text of §4
        ("rank Q-heads, take top ratio × num_heads, then propagate") would
        inflate by up to a factor of `num_kv_per_group` on GQA models
        (e.g. Qwen3-4B group size 4 → 46% target → ~65% after propagate),
        which contradicts the table. We honour the table by selecting at
        KV-group level, which yields exactly `ratio × num_Q_heads` Q-heads.

        For GQA models (num_kv_per_group > 1):
          1. Aggregate Q-head scores to KV-group level (max per group).
          2. Rank KV groups globally, select top ratio%.
          3. Return all Q-heads belonging to selected groups.

        For MHA (num_kv_per_group == 1) the per-head and per-group views coincide.
        """
        nrep = int(self.num_kv_per_group)

        if nrep <= 1:
            # MHA — select Q-heads directly (group == head)
            all_entries = []
            for layer_idx, values in data.items():
                for head_idx, value in enumerate(values):
                    score = value.item() if isinstance(value, torch.Tensor) else float(value)
                    all_entries.append((layer_idx, head_idx, score))
            all_entries.sort(key=lambda x: x[2], reverse=True)
            total = sum(len(v) for v in data.values())
            num_to_select = max(1, round(total * ratio))
            result = {}
            for layer_idx, head_idx, _ in all_entries[:num_to_select]:
                result.setdefault(layer_idx, []).append(head_idx)
            return result

        # GQA — aggregate per KV group (max of Q-heads in the group)
        group_entries = []
        for layer_idx, values in data.items():
            num_q  = len(values)
            num_kv = num_q // nrep
            for kv_idx in range(num_kv):
                q_slice = values[kv_idx * nrep:(kv_idx + 1) * nrep]
                group_score = max(
                    (v.item() if isinstance(v, torch.Tensor) else float(v))
                    for v in q_slice
                )
                group_entries.append((layer_idx, kv_idx, group_score))

        group_entries.sort(key=lambda x: x[2], reverse=True)
        total_groups  = len(group_entries)
        num_to_select = max(1, round(total_groups * ratio))

        result = {}
        for layer_idx, kv_idx, _ in group_entries[:num_to_select]:
            q_heads = list(range(kv_idx * nrep, (kv_idx + 1) * nrep))
            result.setdefault(layer_idx, []).extend(q_heads)

        for layer_idx in result:
            result[layer_idx] = sorted(result[layer_idx])

        total_q = sum(len(v) for v in result.values())
        total_all_q = sum(len(v) for v in data.values())
        print(f"  Selected {num_to_select}/{total_groups} KV groups "
              f"→ {total_q}/{total_all_q} Q-heads "
              f"({100 * total_q / total_all_q:.1f}%)")
        return result

    def _propagate_kv_groups(self, head_dict):
        """Propagate retrieval status to the entire KV group (spec §4, GQA handling).

        If any Q-head in a KV group is retrieval, all Q-heads in that group become retrieval.
        For Qwen3-4B: 32 Q-heads / 8 KV-heads → group size = 4.
        """
        nrep = int(self.num_kv_per_group)
        if nrep == 1:
            return head_dict  # No GQA, nothing to propagate
        propagated = {}
        for layer_idx, heads in head_dict.items():
            expanded = set()
            for h in heads:
                kv_idx = h // nrep          # which KV group this Q-head belongs to
                for qi in range(nrep):      # all Q-heads in that KV group
                    expanded.add(kv_idx * nrep + qi)
            propagated[layer_idx] = sorted(expanded)
        return propagated

    def remove_empty_list_keys(self, dictionary):
        dictionary = {k: v for k, v in dictionary.items() if v != []}
        return dictionary

    def get_compress_heads(self, save_path, efficient=True):
        check_type(save_path, str, param_name="save_path")

        total_tokens = DUMMY_INPUT_LENGTH * REPET_TIMES + 1
        use_efficient = efficient and total_tokens > 6000
        if use_efficient:
            print(f"Using efficient scoring (flash_attn forward + CPU): {total_tokens} tokens")
            prefix_matching_score, copying_matching_score = self.get_attention_score_efficient()
        else:
            print(f"Using standard scoring (eager + output_attentions): {total_tokens} tokens")
            prefix_matching_score, copying_matching_score = self.get_attention_score()

        # Debug: print sizes
        print(f"DEBUG: Raw data - prefix layers: {len(prefix_matching_score)}, copying layers: {len(copying_matching_score)}")
        for layer in sorted(prefix_matching_score.keys())[:3]:
            print(f"  Layer {layer}: {len(prefix_matching_score[layer])} heads")

        # Select at KV-group level, returns full groups already expanded to Q-heads (spec §4)
        print(f"  Induction heads (ratio={self.cfg.induction_head_ratio}):")
        selected_heads_prefix = self.select_top_heads_all(prefix_matching_score, self.cfg.induction_head_ratio)
        print(f"  Echo heads (ratio={self.cfg.echo_head_ratio}):")
        selected_heads_copying = self.select_top_heads_all(copying_matching_score, self.cfg.echo_head_ratio)

        head_dict = {
            'prefix_matching': self.remove_empty_list_keys(selected_heads_prefix),
            'copying': self.remove_empty_list_keys(selected_heads_copying)
        }

        print(f"DEBUG: Final result - prefix layers: {len(head_dict['prefix_matching'])}, copying layers: {len(head_dict['copying'])}")

        with SafeWriteUmask():
            output_model_path = get_valid_write_path(save_path, extensions=".pt")
            torch.save(head_dict, output_model_path)
        logger.info("heads file is stored in %r ", output_model_path)

    def select_top_heads(self, data, ratio):
        # 将所有列表里的值汇总
        all_values = [
            value
            for key in data
            for value in data[key]
        ]
        # 对汇总后的值进行排序
        sorted_values = sorted(all_values, reverse=True)
        # 计算前%的索引
        percent_index = round(len(sorted_values) * ratio)
        # 获取前%的值
        percent_values = sorted_values[:percent_index]

        # Debug info
        print(f"DEBUG: Total values: {len(all_values)}, Top {ratio*100}% index: {percent_index}")
        print(f"DEBUG: Top values: {percent_values[:10]}...")  # Show first 10

        # 创建一个新字典
        result = {}
        for key in data:
            # 获取前%的值在原列表中的索引
            percent_index_in_original_list = [i for i, value in enumerate(data[key]) if value in percent_values]
            result[key] = percent_index_in_original_list
            print(f"DEBUG: Layer {key}: {len(data[key])} values, selected {len(percent_index_in_original_list)} heads")
        return result

    def get_attention_score(self):
        # Новый подход: глобальный сборщик и хуки для каждого слоя
        rand_tokens = torch.tensor(self._sample_vocab_tokens(DUMMY_INPUT_LENGTH) * REPET_TIMES)
        model_tokens = self.tokenizer('', return_tensors="pt")
        if not model_tokens[INPUT_IDS].tolist()[0]:
            model_tokens = self.tokenizer('A', return_tensors="pt")
        model_tokens = {key: model_tokens[key] for key in [INPUT_IDS, 'attention_mask']}

        model_tokens[INPUT_IDS] = torch.tensor(torch.cat((model_tokens[INPUT_IDS][0][-1].unsqueeze(0),
                                                          rand_tokens), dim=0)).reshape(1, -1)
        model_tokens['attention_mask'] = torch.ones(1, DUMMY_INPUT_LENGTH*REPET_TIMES + 1)
        for key in model_tokens:
            model_tokens[key] = model_tokens[key].to(self.model.device)

        # Глобальные словари для сбора данных
        gather_data_prefix = {}
        gather_data_copying = {}

        # Создаём и регистрируем хуки для каждого слоя
        hook_handles = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = AttentionHook(layer_idx, self.num_attention_heads, self.hidden_size,
                                gather_data_prefix, gather_data_copying)
            handle = layer.self_attn.register_forward_hook(hook)
            hook_handles.append((handle, hook))

        print(f"DEBUG: Hooked {len(hook_handles)} attention modules")

        # Запускаем модель
        with torch.no_grad():
            self.model(**model_tokens, output_attentions=True)

        # Удаляем хуки
        for handle, _ in hook_handles:
            handle.remove()

        print(f"DEBUG: Layers with data: prefix={list(gather_data_prefix.keys())}, copying={list(gather_data_copying.keys())}")
        print(f"DEBUG: Total heads collected: prefix={sum(len(v) for v in gather_data_prefix.values())}, "
              f"copying={sum(len(v) for v in gather_data_copying.values())}")

        return gather_data_prefix, gather_data_copying

    def get_attention_score_efficient(self):
        """Flash-attn forward + GPU-vectorized scoring (no O(seq²) CPU work).

        For each layer: compute [H, n_score, seq_len] scores on GPU via batched bmm,
        extract periodic positions with advanced indexing, store only [H] scalars.

        GPU peak: ~8 GB (model) + ~3.2 GB (scores) + ~3.2 GB (attn) ≈ 15 GB at 10001 tokens.
        Speed: ~1 second per forward pass instead of hours on CPU.
        """
        try:
            from flash_attn import flash_attn_func
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        except ImportError as e:
            raise ImportError(f"flash_attn and Qwen3 transformers are required: {e}")

        rand_tokens = torch.tensor(
            self._sample_vocab_tokens(DUMMY_INPUT_LENGTH) * REPET_TIMES
        )
        model_tokens = self.tokenizer('', return_tensors="pt")
        if not model_tokens[INPUT_IDS].tolist()[0]:
            model_tokens = self.tokenizer('A', return_tensors="pt")
        model_tokens = {key: model_tokens[key] for key in [INPUT_IDS, 'attention_mask']}
        model_tokens[INPUT_IDS] = torch.cat(
            (model_tokens[INPUT_IDS][0][-1:], rand_tokens), dim=0
        ).reshape(1, -1).to(self.model.device)
        model_tokens['attention_mask'] = torch.ones(
            1, DUMMY_INPUT_LENGTH * REPET_TIMES + 1, device=self.model.device
        )

        gather_prefix = {}
        gather_copying = {}
        original_forwards = {}
        L, R = DUMMY_INPUT_LENGTH, REPET_TIMES
        # Scoring rows: original positions [L+1 .. seq_len-1] (block_num >= 1, spec §2-3).
        # For scoring row r (0-indexed): most recent prior occurrence is always at
        # original key position r+1 (copying) and r+2 (prefix/induction).
        score_start = L + 1
        n_score = (R - 1) * L

        for layer_idx, layer in enumerate(self.model.model.layers):
            attn_mod = layer.self_attn
            original_forwards[layer_idx] = attn_mod.forward

            def make_forward(self_attn, lidx):
                def fwd(hidden_states, position_embeddings, attention_mask=None, **kwargs):
                    shape = hidden_states.shape[:-1]
                    hshape = (*shape, -1, self_attn.head_dim)
                    q = self_attn.q_norm(self_attn.q_proj(hidden_states).view(hshape)).transpose(1, 2)
                    k = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hshape)).transpose(1, 2)
                    v = self_attn.v_proj(hidden_states).view(hshape).transpose(1, 2)
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)

                    nrep = self_attn.num_key_value_groups
                    bsz, nkv, slen, d = k.shape
                    k_exp = k[:, :, None].expand(bsz, nkv, nrep, slen, d).reshape(bsz, nkv * nrep, slen, d)
                    v_exp = v[:, :, None].expand(bsz, nkv, nrep, slen, d).reshape(bsz, nkv * nrep, slen, d)

                    # Actual output via flash_attn — O(n) GPU memory
                    out = flash_attn_func(
                        q.to(torch.bfloat16).transpose(1, 2),
                        k_exp.to(torch.bfloat16).transpose(1, 2),
                        v_exp.to(torch.bfloat16).transpose(1, 2),
                        softmax_scale=self_attn.scaling,
                        causal=True,
                    )
                    out = out.to(hidden_states.dtype).reshape(*shape, -1).contiguous()
                    out = self_attn.o_proj(out)

                    # GPU-vectorized scoring — no CPU transfer, no Python head loop
                    with torch.no_grad():
                        H, seq_len = q.shape[1], q.shape[2]
                        dev = q.device

                        # [H, n_score, seq_len]: only scoring rows, not the full matrix
                        q_sc = q[0, :, score_start:, :]            # [H, n_score, d]
                        scores = torch.bmm(
                            q_sc.to(torch.bfloat16),
                            k_exp[0].to(torch.bfloat16).transpose(1, 2),  # [H, d, seq_len]
                        ) * self_attn.scaling                         # [H, n_score, seq_len]

                        # Causal mask: query at pos (r + score_start) may only attend to pos <= that
                        q_pos = torch.arange(n_score, device=dev) + score_start   # [n_score]
                        k_pos = torch.arange(seq_len, device=dev)                  # [seq_len]
                        causal = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)          # [n_score, seq_len]
                        scores.masked_fill_(~causal.unsqueeze(0), float('-inf'))
                        del causal

                        attn_s = scores.softmax(dim=-1)   # [H, n_score, seq_len], bf16
                        del scores

                        # Vectorized extraction (spec §2-3, most recent prior only).
                        # For scoring row r: most recent prior is always at key col r+1.
                        # Copying (echo): attn_s[:, r, r+1]
                        # Prefix (induction): attn_s[:, r, r+2]
                        r = torch.arange(n_score, device=dev)
                        copy_h = attn_s[:, r, r + 1].float().mean(dim=1).cpu()
                        # Spec §3: induction excludes last-of-block positions.
                        # Absolute m = r + score_start; m mod L == 0 ⇔ r mod L == L-1.
                        induct_mask = (r % L) != (L - 1)
                        r_ind = r[induct_mask]
                        pref_h = attn_s[:, r_ind, r_ind + 2].float().mean(dim=1).cpu()
                        del attn_s
                        for h in range(H):
                            gather_prefix.setdefault(lidx, []).append(pref_h[h])
                            gather_copying.setdefault(lidx, []).append(copy_h[h])

                    return out, None
                return fwd

            attn_mod.forward = make_forward(attn_mod, layer_idx)

        print(f"DEBUG: Flash-attn + GPU scoring forward ({len(original_forwards)} layers)...")
        with torch.no_grad():
            self.model(**model_tokens, use_cache=False)

        for layer_idx, orig in original_forwards.items():
            self.model.model.layers[layer_idx].self_attn.forward = orig

        print(f"DEBUG: Done. prefix={sum(len(v) for v in gather_prefix.values())}, "
              f"copying={sum(len(v) for v in gather_copying.values())} heads collected")
        return gather_prefix, gather_copying


class AttentionHook:
    """Hook for Qwen3 attention module to extract scores"""

    def __init__(self, layer_idx, num_heads, hidden_size, gather_data_prefix, gather_data_copying):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.gather_data_prefix = gather_data_prefix
        self.gather_data_copying = gather_data_copying

    def __call__(self, module, input, output):
        """Extract attention scores from Qwen3 attention module"""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
        else:
            attn_weights = getattr(module, 'attn_weights', None)

        if attn_weights is None:
            return

        # Move to CPU immediately to free GPU memory
        attn_weights_cpu = attn_weights.float().cpu()
        # Return None for attn_weights so the GPU tensor is released
        modified_output = (output[0],) + (None,) + output[2:]

        _, num_heads, _, _ = attn_weights_cpu.shape

        for head_idx in range(num_heads):
            head_scores = attn_weights_cpu[0, head_idx]  # [seq_len, seq_len]
            prefix_score = self._get_prefix_matching_score(head_scores)
            copying_score = self._get_copying_matching_score(head_scores)
            self.gather_data_prefix.setdefault(self.layer_idx, []).append(prefix_score)
            self.gather_data_copying.setdefault(self.layer_idx, []).append(copying_score)

        return modified_output

    @staticmethod
    def _get_prefix_matching_score(out):
        """Induction score: attention to the token *after* the most recent prior occurrence.

        Spec §3: induction_attn_{h,m} = A_{h, m, n+1}
        where n = max{j < m : x_j = x_m} (most recent prior, not all priors).
        Excludes last-token-of-block positions (m mod K == K-1) per spec §3.
        """
        L = DUMMY_INPUT_LENGTH
        score = []
        for i, token_attn in enumerate(out[1:, 1:]):
            block_num = i // L
            if block_num < 1:
                continue
            start_idx = i % L
            if start_idx == L - 1:                # spec §3: skip last-of-block
                continue
            j = start_idx + (block_num - 1) * L   # most recent prior occurrence
            if j + 1 < len(token_attn):
                score.append(token_attn[j + 1])
        return torch.mean(torch.stack(score)) if score else torch.tensor(0.0)

    @staticmethod
    def _get_copying_matching_score(out):
        """Echo score: attention to the most recent prior occurrence of the same token.

        Spec §2: echo_attn_{h,m} = A_{h, m, n}
        where n = max{j < m : x_j = x_m} (most recent prior).
        Includes all positions with at least one prior occurrence (block_num >= 1).
        """
        L = DUMMY_INPUT_LENGTH
        score = []
        for i, token_attn in enumerate(out[1:, 1:]):
            block_num = i // L
            if block_num < 1:
                continue
            start_idx = i % L
            j = start_idx + (block_num - 1) * L   # most recent prior occurrence
            if j < len(token_attn):
                score.append(token_attn[j])
        return torch.mean(torch.stack(score)) if score else torch.tensor(0.0)


class SoftmaxDumpOutput:

    def __init__(self, num_attention_heads, hidden_size):
        self.head_num = 0
        self.call_count = 0
        self.torch_softmax = torch.nn.functional.softmax
        self.gather_data_prefix = {}
        self.gather_data_copying = {}
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def __call__(self, inputs, **kwargs):
        out = self.torch_softmax(inputs, **kwargs) # torch.Size([64, 10001, 10001])
        if self.num_attention_heads == 0:
            raise ValueError("Num attention heads can not be zero.")
        cur_layer = self.head_num // self.num_attention_heads
        cur_head = self.head_num % self.num_attention_heads
        logger.info(f"The {cur_layer}-th layer {cur_head}-th head attention score has been obtained")
        self.gather_data_prefix.setdefault(cur_layer, []).append(SoftmaxDumpOutput._get_prefix_matching_score(out[0]))
        self.gather_data_copying.setdefault(cur_layer, []).append(SoftmaxDumpOutput._get_copying_matching_score(out[0]))
        self.head_num += 1
        self.call_count += 1
        return out
    
    @staticmethod
    def _get_prefix_matching_score(out):
        L = DUMMY_INPUT_LENGTH
        score = []
        for i, token_attn in enumerate(out[1:, 1:]):
            block_num = i // L
            if block_num < 1:
                continue
            start_idx = i % L
            if start_idx == L - 1:                # spec §3: skip last-of-block
                continue
            j = start_idx + (block_num - 1) * L   # most recent prior occurrence
            if j + 1 < len(token_attn):
                score.append(token_attn[j + 1])
        return torch.mean(torch.Tensor(score)) if score else torch.tensor(0.0)

    @staticmethod
    def _get_copying_matching_score(out):
        L = DUMMY_INPUT_LENGTH
        score = []
        for i, token_attn in enumerate(out[1:, 1:]):
            block_num = i // L
            if block_num < 1:
                continue
            j = i % L + (block_num - 1) * L   # most recent prior occurrence
            if j < len(token_attn):
                score.append(token_attn[j])
        return torch.mean(torch.Tensor(score)) if score else torch.tensor(0.0)

