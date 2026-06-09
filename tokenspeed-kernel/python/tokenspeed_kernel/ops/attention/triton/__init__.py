# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math
from typing import Tuple

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.mha_decode import decode_attention_fwd
from tokenspeed_kernel.ops.attention.triton.mha_prefill import prefill_attention_fwd
from tokenspeed_kernel.ops.attention.triton.mla_prefill import mla_prefill_attention_fwd
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def mha_merge_state_kernel(
    OutA,
    LseA,
    OutB,
    LseB,
    Out,
    Lse,
    head_dim: tl.constexpr,
    lse_scale_log2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    value_offsets = row * head_dim + offs_d

    lse_a = tl.load(LseA + row).to(tl.float32)
    lse_b = tl.load(LseB + row).to(tl.float32)
    lse_a_log2 = lse_a * lse_scale_log2
    lse_b_log2 = lse_b * lse_scale_log2
    lse_max_log2 = tl.maximum(lse_a_log2, lse_b_log2)

    weight_a = tl.exp2(lse_a_log2 - lse_max_log2)
    weight_b = tl.exp2(lse_b_log2 - lse_max_log2)
    denom = weight_a + weight_b

    out_a = tl.load(OutA + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out_b = tl.load(OutB + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out = (out_a * weight_a + out_b * weight_b) / denom
    merged_lse = (lse_max_log2 + tl.log2(denom)) / lse_scale_log2

    tl.store(Out + value_offsets, out, mask=mask_d)
    tl.store(Lse + row, merged_lse)

LSE_LOG2 = 1.0
LSE_LN = math.log2(math.e)


@triton.jit
def _merge_state_kernel(
    v_a,
    s_a,
    v_b,
    s_b,
    v_merged,
    s_merged,
    lse_scale_log2,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < HEAD_DIM

    s_a_log2 = tl.load(s_a + token_head_idx).to(tl.float32) * lse_scale_log2
    s_b_log2 = tl.load(s_b + token_head_idx).to(tl.float32) * lse_scale_log2
    s_max = tl.maximum(s_a_log2, s_b_log2)
    w_a = tl.exp2(s_a_log2 - s_max)
    w_b = tl.exp2(s_b_log2 - s_max)
    sum_w = w_a + w_b

    v_offsets = token_head_idx * HEAD_DIM + offsets
    v_a_values = tl.load(v_a + v_offsets, mask=mask, other=0.0).to(tl.float32)
    v_b_values = tl.load(v_b + v_offsets, mask=mask, other=0.0).to(tl.float32)
    v_values = (w_a * v_a_values + w_b * v_b_values) / sum_w
    s_value = (tl.log2(sum_w) + s_max) / lse_scale_log2

    tl.store(v_merged + v_offsets, v_values, mask=mask)
    tl.store(s_merged + token_head_idx, s_value)


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    *,
    lse_scale_log2: float = LSE_LN,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert v_a.is_contiguous() and v_b.is_contiguous()
    assert s_a.is_contiguous() and s_b.is_contiguous()
    assert v_a.shape == v_b.shape
    assert s_a.shape == s_b.shape
    assert v_a.shape[:2] == s_a.shape
    assert v_a.dtype == v_b.dtype
    assert v_a.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"merge_state V must be bf16/fp16, got {v_a.dtype}"
    assert (
        s_a.dtype == torch.float32 and s_b.dtype == torch.float32
    ), f"merge_state expects fp32 LSE, got s_a={s_a.dtype} s_b={s_b.dtype}"

    seq_len, num_heads, head_dim = v_a.shape
    v_out = torch.empty_like(v_a)
    s_out = torch.empty(seq_len, num_heads, dtype=torch.float32, device=v_a.device)
    if seq_len == 0 or num_heads == 0:
        return v_out, s_out

    block_d = triton.next_power_of_2(head_dim)
    _merge_state_kernel[(seq_len * num_heads,)](
        v_a,
        s_a,
        v_b,
        s_b,
        v_out,
        s_out,
        float(lse_scale_log2),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=1 if block_d <= 64 else 4,
    )
    return v_out, s_out


def normalize_mla_cache(cache: torch.Tensor) -> torch.Tensor:
    if cache.dim() == 3:
        return cache
    if cache.dim() != 4:
        raise ValueError(
            "MLA cache must have shape [pages, page_size, dim], "
            "[pages, page_size, 1, dim], or [pages, 1, page_size, dim]"
        )
    if cache.shape[1] == 1:
        return cache.squeeze(1)
    if cache.shape[2] == 1:
        return cache.squeeze(2)
    raise ValueError("MLA cache must have a singleton head dimension for 4D layouts")


def normalize_page_size(cache: torch.Tensor) -> int:
    if cache.dim() == 3:
        return cache.shape[1]
    if cache.shape[1] == 1:
        return cache.shape[2]
    return cache.shape[1]


def flatten_mla_cache_view(cache: torch.Tensor) -> torch.Tensor:
    cache = normalize_mla_cache(cache)
    return cache.as_strided(
        (cache.shape[0] * cache.shape[1], 1, cache.shape[2]),
        (cache.stride(1), cache.stride(0), cache.stride(2)),
    )


def pack_mla_cache_view(
    ckv_cache: torch.Tensor, kpe_cache: torch.Tensor
) -> torch.Tensor:
    ckv_cache = normalize_mla_cache(ckv_cache)
    kpe_cache = normalize_mla_cache(kpe_cache)
    packed_dim = ckv_cache.shape[-1] + kpe_cache.shape[-1]
    can_view_as_packed = (
        ckv_cache.untyped_storage().data_ptr() == kpe_cache.untyped_storage().data_ptr()
        and ckv_cache.shape[:-1] == kpe_cache.shape[:-1]
        and ckv_cache.stride() == kpe_cache.stride()
        and kpe_cache.storage_offset()
        == ckv_cache.storage_offset() + ckv_cache.shape[-1] * ckv_cache.stride(-1)
    )
    if can_view_as_packed:
        return ckv_cache.as_strided(
            (*ckv_cache.shape[:-1], packed_dim),
            ckv_cache.stride(),
        )

    kv_cache = torch.empty(
        (*ckv_cache.shape[:-1], packed_dim),
        dtype=ckv_cache.dtype,
        device=ckv_cache.device,
    )
    kv_cache[..., : ckv_cache.shape[-1]] = ckv_cache
    kv_cache[..., ckv_cache.shape[-1] :] = kpe_cache
    return kv_cache


# ------------------------------------------------------------------------------
# Kernel registration
# ------------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_prefill",
    name="triton_mha_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    cache_seqlens = torch.empty((0,), dtype=torch.int32, device=q.device)
    empty_k = torch.empty((0, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    empty_v = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
    prefill_attention_fwd(
        q,
        k,
        v,
        out,
        empty_k,
        empty_v,
        cu_seqlens,
        cache_seqlens,
        None,
        True,
        max_seqlen,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=sinks,
        has_kv_cache=False,
        lse_extend=lse,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="triton_mha_extend_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "is_causal": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    k = torch.empty(
        (0, k_cache.shape[2], k_cache.shape[3]),
        dtype=k_cache.dtype,
        device=k_cache.device,
    )
    v = torch.empty(
        (0, v_cache.shape[2], v_cache.shape[3]),
        dtype=v_cache.dtype,
        device=v_cache.device,
    )

    out = torch.empty_like(q)
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    prefill_attention_fwd(
        q,
        k,
        v,
        out,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        cu_seqlens_q,
        cache_seqlens,
        None,
        is_causal,
        max_seqlen_q,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=sinks,
        page_table=page_table,
        page_table_stride_b=page_table.stride(0),
        page_size=k_cache.shape[1],
        has_kv_cache=True,
        lse_extend=lse,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="triton_mha_decode_with_kvcache_cached",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    out = torch.empty_like(q)
    max_kv_splits = 4
    attn_logits = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        dtype=torch.float32,
        device=q.device,
    )
    num_kv_splits = torch.ones(
        (cache_seqlens.shape[0],), dtype=torch.int32, device=q.device
    )
    decode_attention_fwd(
        q,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        out,
        page_table,
        cache_seqlens,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        page_table.stride(0),
        k_cache.shape[1],
        window_left,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=logit_cap,
        sinks=sinks,
    )
    return out


@register_kernel(
    "attention",
    "mha_merge_state",
    name="triton_mha_merge_state",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("out_a", "out_b"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={},
    tags={"portability"},
)
def triton_mha_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(out_a)
    lse = torch.empty_like(lse_a)
    total_rows = out_a.shape[0] * out_a.shape[1]
    head_dim = out_a.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    mha_merge_state_kernel[(total_rows,)](
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        lse,
        head_dim,
        float(lse_scale_log2),
        BLOCK_D=block_d,
    )
    return out, lse


@register_kernel(
    "attention",
    "mla_prefill",
    name="triton_mla_prefill",
    features={"mla"},
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q_nope", "q_pe", "k_nope", "k_pe", "v"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    priority=Priority.PERFORMANT,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({True, False}),
    },
    tags={"portability"},
)
def triton_mla_prefill(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = q_nope.new_empty((*q_nope.shape[:-1], v.shape[-1]))
    lse = (
        torch.empty(q_nope.shape[:-1], dtype=torch.float32, device=q_nope.device)
        if return_lse
        else None
    )
    sm_scale = (
        softmax_scale
        if softmax_scale is not None
        else 1.0 / math.sqrt(q_nope.shape[-1] + q_pe.shape[-1])
    )
    q = torch.cat((q_nope, q_pe), dim=-1)
    k = torch.cat((k_nope, k_pe), dim=-1)
    empty_k = torch.empty((0, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    empty_v = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
    cache_seqlens = torch.empty((0,), dtype=torch.int32, device=q.device)
    mla_prefill_attention_fwd(
        q,
        k,
        v,
        out,
        empty_k,
        empty_v,
        cu_seqlens_q,
        cu_seqlens_kv,
        cache_seqlens,
        None,
        is_causal,
        max_seqlen_q,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=None,
        lse=lse,
        has_kv_cache=False,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "mla_prefill_with_kvcache",
    name="triton_mla_prefill_with_kvcache_cached",
    features={"mla", "paged"},
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q_nope", "q_pe", "kv_cache"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    priority=Priority.PERFORMANT,
    traits={
        "kv_cache_mode": frozenset({"cached"}),
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mla_prefill_with_kvcache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    qk_nope_head_dim: int | None = None,
) -> torch.Tensor:
    out = torch.empty_like(q_nope)
    ckv_cache = kv_cache[..., : q_nope.shape[-1]]
    kpe_cache = kv_cache[..., q_nope.shape[-1] :]
    sm_scale = (
        softmax_scale
        if softmax_scale is not None
        else 1.0 / math.sqrt(q_nope.shape[-1] + q_pe.shape[-1])
    )
    q = torch.cat((q_nope, q_pe), dim=-1)
    kv_cache = pack_mla_cache_view(ckv_cache, kpe_cache)
    k_cache = flatten_mla_cache_view(kv_cache)
    v_cache = flatten_mla_cache_view(kv_cache[..., : out.shape[-1]])
    dummy_k = q.new_empty((0, 1, q.shape[-1]))
    dummy_v = q.new_empty((0, 1, out.shape[-1]))
    mla_prefill_attention_fwd(
        q,
        dummy_k,
        dummy_v,
        out,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_q,
        cache_seqlens,
        None,
        is_causal,
        max_seqlen_q,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=None,
        page_table=page_table,
        page_table_stride_b=page_table.stride(0),
        page_size=normalize_page_size(kv_cache),
        extend_from_cache=True,
        has_kv_cache=True,
    )
    return out


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="triton_mla_decode_with_kvcache_cached",
    features={"mla", "paged"},
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q_nope", "q_pe", "kv_cache"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    priority=Priority.PERFORMANT,
    traits={
        "kv_cache_mode": frozenset({"cached"}),
        "uniform_query_len": frozenset({True}),
        "query_len": frozenset({1}),
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mla_decode_with_kvcache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    qk_nope_head_dim: int | None = None,
) -> torch.Tensor:
    out = torch.empty_like(q_nope)
    ckv_cache = kv_cache[..., : q_nope.shape[-1]]
    kpe_cache = kv_cache[..., q_nope.shape[-1] :]
    q = torch.cat((q_nope, q_pe), dim=-1)
    packed_cache = pack_mla_cache_view(ckv_cache, kpe_cache)
    k_cache = flatten_mla_cache_view(packed_cache)
    v_cache = flatten_mla_cache_view(packed_cache[..., : out.shape[-1]])
    max_kv_splits = 4
    attn_logits = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        out.shape[-1],
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        q.shape[0], q.shape[1], max_kv_splits, dtype=torch.float32, device=q.device
    )
    num_kv_splits = torch.ones(
        (cache_seqlens.shape[0],), dtype=torch.int32, device=q.device
    )
    sm_scale = (
        softmax_scale
        if softmax_scale is not None
        else 1.0 / math.sqrt(q_nope.shape[-1] + q_pe.shape[-1])
    )
    decode_attention_fwd(
        q,
        k_cache,
        v_cache,
        out,
        page_table,
        cache_seqlens,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        page_table.stride(0),
        normalize_page_size(packed_cache),
        window_left,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sinks=None,
    )
    return out


# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------

__all__ = ["merge_state"]
