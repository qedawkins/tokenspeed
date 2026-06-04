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

"""Paged MLA decode Gluon kernel for AMD GFX950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.attention.gluon.utils import _INV_LN2_VALUE, maximum
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


_MLA_DECODE_SIGNATURES = frozenset(
    {
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            kv_cache=dense_tensor_format(torch.bfloat16),
        ),
        format_signature(
            q=dense_tensor_format(torch.float16),
            kv_cache=dense_tensor_format(torch.float16),
        ),
        format_signature(
            q=dense_tensor_format(torch.float8_e4m3fn),
            kv_cache=dense_tensor_format(torch.float8_e4m3fn),
        ),
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            kv_cache=dense_tensor_format(torch.float8_e4m3fn),
        ),
        format_signature(
            q=dense_tensor_format(torch.float16),
            kv_cache=dense_tensor_format(torch.float8_e4m3fn),
        ),
    }
)


@gluon.aggregate
class MLADecodeConfig:
    Q_STRIDE_M: gl.constexpr
    Q_STRIDE_H: gl.constexpr
    Q_STRIDE_D: gl.constexpr
    KV_STRIDE_P: gl.constexpr
    KV_STRIDE_T: gl.constexpr
    KV_STRIDE_D: gl.constexpr
    PAGE_TABLE_STRIDE_M: gl.constexpr
    OUT_STRIDE_M: gl.constexpr
    OUT_STRIDE_H: gl.constexpr
    OUT_STRIDE_D: gl.constexpr
    SOFTMAX_SCALE_LOG2: gl.constexpr
    PAGE_SIZE: gl.constexpr
    QUERY_DIM: gl.constexpr
    VALUE_HEAD_DIM: gl.constexpr
    BLOCK_QK: gl.constexpr
    BLOCK_D: gl.constexpr
    qk_layout: gl.constexpr
    value_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        Q_STRIDE_M,
        Q_STRIDE_H,
        Q_STRIDE_D,
        KV_STRIDE_P,
        KV_STRIDE_T,
        KV_STRIDE_D,
        PAGE_TABLE_STRIDE_M,
        OUT_STRIDE_M,
        OUT_STRIDE_H,
        OUT_STRIDE_D,
        SOFTMAX_SCALE_LOG2,
        PAGE_SIZE,
        QUERY_DIM,
        VALUE_HEAD_DIM,
        BLOCK_QK,
        BLOCK_D,
    ):
        assert PAGE_SIZE == 64
        assert QUERY_DIM > 0
        assert VALUE_HEAD_DIM > 0
        assert BLOCK_QK == 64
        assert BLOCK_D == 64

        qk_layout = gl.BlockedLayout([1], [BLOCK_QK], [1], [0])
        value_layout = gl.BlockedLayout([1], [BLOCK_D], [1], [0])

        self.Q_STRIDE_M = gl.constexpr(Q_STRIDE_M)
        self.Q_STRIDE_H = gl.constexpr(Q_STRIDE_H)
        self.Q_STRIDE_D = gl.constexpr(Q_STRIDE_D)
        self.KV_STRIDE_P = gl.constexpr(KV_STRIDE_P)
        self.KV_STRIDE_T = gl.constexpr(KV_STRIDE_T)
        self.KV_STRIDE_D = gl.constexpr(KV_STRIDE_D)
        self.PAGE_TABLE_STRIDE_M = gl.constexpr(PAGE_TABLE_STRIDE_M)
        self.OUT_STRIDE_M = gl.constexpr(OUT_STRIDE_M)
        self.OUT_STRIDE_H = gl.constexpr(OUT_STRIDE_H)
        self.OUT_STRIDE_D = gl.constexpr(OUT_STRIDE_D)
        self.SOFTMAX_SCALE_LOG2 = gl.constexpr(SOFTMAX_SCALE_LOG2)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.QUERY_DIM = gl.constexpr(QUERY_DIM)
        self.VALUE_HEAD_DIM = gl.constexpr(VALUE_HEAD_DIM)
        self.BLOCK_QK = gl.constexpr(BLOCK_QK)
        self.BLOCK_D = gl.constexpr(BLOCK_D)
        self.qk_layout = gl.constexpr(qk_layout)
        self.value_layout = gl.constexpr(value_layout)

    @gluon.jit
    def q_offsets(self, row, head, offs_d):
        return (
            row * self.Q_STRIDE_M
            + head * self.Q_STRIDE_H
            + offs_d * self.Q_STRIDE_D
        ).to(gl.int32)

    @gluon.jit
    def kv_offsets(self, physical_page, token_slot, offs_d):
        return (
            physical_page * self.KV_STRIDE_P
            + token_slot * self.KV_STRIDE_T
            + offs_d * self.KV_STRIDE_D
        ).to(gl.int32)

    @gluon.jit
    def out_offsets(self, row, head, offs_d):
        return (
            row * self.OUT_STRIDE_M
            + head * self.OUT_STRIDE_H
            + offs_d * self.OUT_STRIDE_D
        ).to(gl.int32)


@gluon.jit
def _mla_decode_paged(
    q_ptr,
    kv_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    out_ptr,
    Q_STRIDE_M: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    KV_STRIDE_P: gl.constexpr,
    KV_STRIDE_T: gl.constexpr,
    KV_STRIDE_D: gl.constexpr,
    PAGE_TABLE_STRIDE_M: gl.constexpr,
    OUT_STRIDE_M: gl.constexpr,
    OUT_STRIDE_H: gl.constexpr,
    OUT_STRIDE_D: gl.constexpr,
    SOFTMAX_SCALE_LOG2: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    QUERY_DIM: gl.constexpr,
    VALUE_HEAD_DIM: gl.constexpr,
    BLOCK_QK: gl.constexpr,
    BLOCK_D: gl.constexpr,
):
    cfg = MLADecodeConfig(
        Q_STRIDE_M,
        Q_STRIDE_H,
        Q_STRIDE_D,
        KV_STRIDE_P,
        KV_STRIDE_T,
        KV_STRIDE_D,
        PAGE_TABLE_STRIDE_M,
        OUT_STRIDE_M,
        OUT_STRIDE_H,
        OUT_STRIDE_D,
        SOFTMAX_SCALE_LOG2,
        PAGE_SIZE,
        QUERY_DIM,
        VALUE_HEAD_DIM,
        BLOCK_QK,
        BLOCK_D,
    )
    row = gl.program_id(0)
    head = gl.program_id(1)
    value_block = gl.program_id(2)
    value_base = value_block * cfg.BLOCK_D
    value_offsets = value_base + gl.arange(
        0, cfg.BLOCK_D, layout=cfg.value_layout
    )
    valid_value = value_offsets < cfg.VALUE_HEAD_DIM
    cache_len = gl.load(cache_seqlens_ptr + row)

    max_score = gl.full((), value=-float("inf"), dtype=gl.float32)
    for token in range(0, cache_len):
        page_id = token // cfg.PAGE_SIZE
        token_slot = token - page_id * cfg.PAGE_SIZE
        physical_page = gl.load(page_table_ptr + row * cfg.PAGE_TABLE_STRIDE_M + page_id)
        score = gl.full((), value=0.0, dtype=gl.float32)
        for query_base in range(0, cfg.QUERY_DIM, cfg.BLOCK_QK):
            query_offsets = query_base + gl.arange(
                0, cfg.BLOCK_QK, layout=cfg.qk_layout
            )
            valid_query = query_offsets < cfg.QUERY_DIM
            q = gl.load(
                q_ptr + cfg.q_offsets(row, head, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            k = gl.load(
                kv_cache_ptr + cfg.kv_offsets(physical_page, token_slot, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            score += gl.sum(q * k, axis=0)
        score *= cfg.SOFTMAX_SCALE_LOG2
        max_score = maximum(max_score, score)

    denom = gl.full((), value=0.0, dtype=gl.float32)
    acc = gl.full(
        [cfg.BLOCK_D], value=0.0, dtype=gl.float32, layout=cfg.value_layout
    )
    for token in range(0, cache_len):
        page_id = token // cfg.PAGE_SIZE
        token_slot = token - page_id * cfg.PAGE_SIZE
        physical_page = gl.load(page_table_ptr + row * cfg.PAGE_TABLE_STRIDE_M + page_id)
        score = gl.full((), value=0.0, dtype=gl.float32)
        for query_base in range(0, cfg.QUERY_DIM, cfg.BLOCK_QK):
            query_offsets = query_base + gl.arange(
                0, cfg.BLOCK_QK, layout=cfg.qk_layout
            )
            valid_query = query_offsets < cfg.QUERY_DIM
            q = gl.load(
                q_ptr + cfg.q_offsets(row, head, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            k = gl.load(
                kv_cache_ptr + cfg.kv_offsets(physical_page, token_slot, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            score += gl.sum(q * k, axis=0)
        score *= cfg.SOFTMAX_SCALE_LOG2
        prob = gl.exp2(score - max_score)
        value = gl.load(
            kv_cache_ptr + cfg.kv_offsets(physical_page, token_slot, value_offsets),
            mask=valid_value,
            other=0.0,
        ).to(gl.float32)
        acc += prob * value
        denom += prob

    output = gl.where(cache_len > 0, acc / denom, 0.0)
    output = output.to(out_ptr.dtype.element_ty)
    gl.store(
        out_ptr + cfg.out_offsets(row, head, value_offsets),
        output,
        mask=valid_value,
    )


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="gluon_mla_decode_gfx950",
    features={"paged", "mla"},
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_MLA_DECODE_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "page_size": frozenset({64}),
    },
)
def gluon_mla_decode_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    value_head_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    del max_seqlen_k

    out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    output = torch.empty(
        (q.shape[0], q.shape[1], value_head_dim),
        device=q.device,
        dtype=out_dtype,
    )
    if q.shape[0] == 0:
        return output

    block_qk = 64
    block_d = 64
    grid = (q.shape[0], q.shape[1], math.ceil(value_head_dim / block_d))
    _mla_decode_paged[grid](
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        page_table.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        softmax_scale * _INV_LN2_VALUE,
        kv_cache.shape[1],
        kv_lora_rank + qk_rope_head_dim,
        value_head_dim,
        block_qk,
        block_d,
        num_warps=1,
    )
    return output
