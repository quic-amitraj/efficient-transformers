# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
from transformers.cache_utils import Cache

from QEfficient.blocking.blocked_attention_forwards import (
    blocked_bhqkv_attention_forward,
    blocked_h_attention_forward,
    blocked_h_mla_attention_forward,
    blocked_hqkv_attention_forward,
    blocked_kv_attention_forward,
    blocked_kv_mla_attention_forward,
    blocked_q_attention_forward,
    blocked_qkv_attention_forward,
)


class BlockingMode(str, Enum):
    NONE = ""
    KV = "kv"
    Q = "q"
    H = "h"
    QKV = "qkv"
    HQ = "hq"
    HKV = "hkv"
    HQKV = "hqkv"
    BHQKV = "bhqkv"


@dataclass
class AttentionBlockingConfig:
    mode: BlockingMode = BlockingMode.NONE
    num_kv_blocks: Optional[int] = None
    num_q_blocks: Optional[int] = None
    head_block_size: Optional[int] = None
    skip_kv: Optional[bool] = True
    num_batch_blocks: Optional[int] = None
    # ── Skip-softmax (BLASST) threshold scale factors ────────────────────
    # Threshold formula: λ = scale_factor / context_length
    # The scale factor is calibrated per model and target sparsity.
    # Use the split prefill/decode fields for production (the attention
    # sparsity pattern differs between the two phases, so optimal thresholds
    # differ too).  The combined field is a convenience fallback.
    #
    # Reference values for Qwen3-30B-A3B-Instruct-2507 (NVIDIA TRT-LLM):
    #   50 % sparsity → prefill=587,  decode=16.5
    #   70 % sparsity → prefill=3293, decode=119
    skip_softmax_scale_factor: Optional[float] = None           # used when prefill/decode not split
    skip_softmax_scale_factor_prefill: Optional[float] = None   # seq_len > 1
    skip_softmax_scale_factor_decode: Optional[float] = None    # seq_len == 1 (single decode step)


def supports_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_blockedKV")


_STRATEGIES: Dict[BlockingMode, Callable] = {
    BlockingMode.KV: blocked_kv_attention_forward,
    BlockingMode.Q: blocked_q_attention_forward,
    BlockingMode.H: blocked_h_attention_forward,
    BlockingMode.QKV: blocked_qkv_attention_forward,
    BlockingMode.HQ: blocked_hqkv_attention_forward,
    BlockingMode.HKV: blocked_hqkv_attention_forward,
    BlockingMode.HQKV: blocked_hqkv_attention_forward,
    BlockingMode.BHQKV: blocked_bhqkv_attention_forward,
}

_STRATEGIES_MLA: Dict[BlockingMode, Callable] = {
    BlockingMode.KV: blocked_kv_mla_attention_forward,
    BlockingMode.H: blocked_h_mla_attention_forward,
}


# helper function needed both in generic blocked approach and in other modeling files for non-blocked approach
def past_key_value_update(
    module,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_value: Cache,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    sliding_window: Optional[int] = None,
):
    if past_key_value is not None:
        cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
        if sliding_window is not None:
            cache_kwargs.update(
                {
                    "is_sliding": sliding_window is not None,
                    "sliding_window": past_key_value.sliding_window_len,
                }
            )
        if comp_ctx_lengths is not None:
            attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
            cache_kwargs["CCL"] = attention_mask.shape[-1]
        key, value = past_key_value.update(key, value, module.layer_idx, cache_kwargs)
    return key, value, attention_mask, cache_kwargs


def generic_blocked_attention_interface(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    layer_idx: int,
    past_key_value: Cache,
    blocking_config: AttentionBlockingConfig,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_seen_tokens: Optional[int] = None,
    non_blocked_forward: Callable = None,
    score_mod: Optional[Callable] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    # ── Dynamic skip-softmax inputs (optional) ───────────────────────────
    # When provided (shape [1], float32), these tensor inputs override the
    # static float values baked in AttentionBlockingConfig, allowing the
    # threshold to be varied at inference time without recompiling.
    # The ONNX graph uses torch.where + shape_as_tensor to select between
    # them dynamically so both become live ONNX nodes (not constants).
    skip_softmax_scale_factor_prefill_t: Optional[torch.Tensor] = None,
    skip_softmax_scale_factor_decode_t: Optional[torch.Tensor] = None,
    **kwargs,
):
    use_kv_blocked = (
        blocking_config is not None and "kv" in blocking_config.mode and supports_blocked_kv(past_key_value)
    )

    if past_key_value is not None:
        if use_kv_blocked and sliding_window is None:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
                "past_seen_tokens": past_seen_tokens,
            }
            if sliding_window is not None:
                cache_kwargs.update(
                    {
                        "is_sliding": sliding_window is not None,
                        "sliding_window": past_key_value.sliding_window_len,
                    }
                )
            past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
        else:
            key, value, attention_mask, cache_kwargs = past_key_value_update(
                module=module,
                key=key,
                value=value,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
                sliding_window=sliding_window,
            )

    strategy = _STRATEGIES.get(blocking_config.mode)

    # ── Resolve skip-softmax scale factor ────────────────────────────────
    # Single-tensor path: use the tensor directly (no Where/shape_as_tensor).
    # The caller seeds the buffer with the correct value for the current phase.
    # No .squeeze() — keeps the [1] shape so the ONNX has a direct edge from
    # the named input to the log computation without an intermediate Squeeze node.
    if skip_softmax_scale_factor_prefill_t is not None:
        skip_softmax_scale_factor = skip_softmax_scale_factor_prefill_t
    elif skip_softmax_scale_factor_decode_t is not None:
        skip_softmax_scale_factor = skip_softmax_scale_factor_decode_t
    else:
        # Static fallback — baked floats from blocking_config.
        seq_len = query.shape[2]
        sf_static = (
            (blocking_config.skip_softmax_scale_factor_decode or blocking_config.skip_softmax_scale_factor)
            if seq_len == 1
            else (blocking_config.skip_softmax_scale_factor_prefill or blocking_config.skip_softmax_scale_factor)
        )
        skip_softmax_scale_factor = (
            torch.tensor([sf_static], dtype=torch.float32, device=query.device)
            if sf_static is not None
            else None
        )

    attn_output, attn_weights, log_thresh_out, skip_blocks = strategy(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        scaling=scaling,
        cache_kwargs=cache_kwargs,
        layer_idx=layer_idx,
        past_key_value=past_key_value,
        num_kv_blocks=blocking_config.num_kv_blocks,
        num_q_blocks=blocking_config.num_q_blocks,
        head_block_size=blocking_config.head_block_size,
        num_batch_blocks=blocking_config.num_batch_blocks,
        score_mod=score_mod,
        position_bias=position_bias,
        sinks=sinks,
        skip_softmax_scale_factor=skip_softmax_scale_factor,
        # Only collect skip_blocks when debug mode is active.
        # Without this gate the dead Stack nodes in the ONNX cause compiler
        # failures (Invalid index) for certain num_kv_blocks values.
        collect_debug=getattr(module, "_debug_output", False),
    )

    return attn_output, attn_weights, log_thresh_out, skip_blocks


def generic_blocked_mla_attention_interface(
    module,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    mla_absorption: Dict[str, Any],
    blocking_config: AttentionBlockingConfig,
    query: Optional[torch.Tensor] = None,
    q_a_proj_out: Optional[torch.Tensor] = None,
    fusedqk: Optional[torch.Tensor] = None,
    q_nope: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    kva: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
    per_head_q_up: Optional[torch.Tensor] = None,
    per_head_k_up: Optional[torch.Tensor] = None,
    per_head_v_up: Optional[torch.Tensor] = None,
    per_head_k_up_normal: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
    compressed_kvs: Optional[torch.Tensor] = None,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_seen_tokens: Optional[int] = None,
    non_blocked_forward: Callable = None,
    score_mod: Optional[Callable] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
    mla_blocking_strategy = _STRATEGIES_MLA.get(blocking_config.mode)
    attn_output, attn_weights = mla_blocking_strategy(
        module=module,
        query=query,
        q_a_proj_out=q_a_proj_out,
        fusedqk=fusedqk,
        q_nope=q_nope,
        q_pe=q_pe,
        kva=kva,
        k_pe=k_pe,
        per_head_q_up=per_head_q_up,
        per_head_k_up=per_head_k_up,
        per_head_v_up=per_head_v_up,
        per_head_k_up_normal=per_head_k_up_normal,
        attention_mask=attention_mask,
        scaling=scaling,
        cache_kwargs=cache_kwargs,
        layer_idx=layer_idx,
        compressed_kvs=compressed_kvs,
        mla_absorption=mla_absorption,
        num_kv_blocks=blocking_config.num_kv_blocks,
        num_q_blocks=blocking_config.num_q_blocks,
        head_block_size=blocking_config.head_block_size,
        num_batch_blocks=blocking_config.num_batch_blocks,
        score_mod=score_mod,
        position_bias=position_bias,
        sinks=sinks,
    )

    return attn_output, attn_weights
