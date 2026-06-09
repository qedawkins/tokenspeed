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

"""Ragged MLA prefill Gluon kernel for AMD GFX950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.attention.gluon.utils import _INV_LN2_VALUE, maximum
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


_MLA_PREFILL_SIGNATURES = frozenset(
    {
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k=dense_tensor_format(torch.bfloat16),
            v=dense_tensor_format(torch.bfloat16),
        ),
        format_signature(
            q=dense_tensor_format(torch.float16),
            k=dense_tensor_format(torch.float16),
            v=dense_tensor_format(torch.float16),
        ),
        format_signature(
            q=dense_tensor_format(torch.float8_e4m3fn),
            k=dense_tensor_format(torch.float8_e4m3fn),
            v=dense_tensor_format(torch.float8_e4m3fn),
        ),
    }
)


@gluon.aggregate
class MLAPrefillConfig:
    Q_STRIDE_T: gl.constexpr
    Q_STRIDE_H: gl.constexpr
    Q_STRIDE_D: gl.constexpr
    K_STRIDE_T: gl.constexpr
    K_STRIDE_H: gl.constexpr
    K_STRIDE_D: gl.constexpr
    V_STRIDE_T: gl.constexpr
    V_STRIDE_H: gl.constexpr
    V_STRIDE_D: gl.constexpr
    OUT_STRIDE_T: gl.constexpr
    OUT_STRIDE_H: gl.constexpr
    OUT_STRIDE_D: gl.constexpr
    SOFTMAX_SCALE_LOG2: gl.constexpr
    BATCH_SIZE: gl.constexpr
    NUM_Q_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    GROUP_SIZE: gl.constexpr
    QUERY_DIM: gl.constexpr
    VALUE_HEAD_DIM: gl.constexpr
    BLOCK_QK: gl.constexpr
    BLOCK_D: gl.constexpr
    IS_CAUSAL: gl.constexpr
    HAS_LSE: gl.constexpr
    qk_layout: gl.constexpr
    value_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        Q_STRIDE_T,
        Q_STRIDE_H,
        Q_STRIDE_D,
        K_STRIDE_T,
        K_STRIDE_H,
        K_STRIDE_D,
        V_STRIDE_T,
        V_STRIDE_H,
        V_STRIDE_D,
        OUT_STRIDE_T,
        OUT_STRIDE_H,
        OUT_STRIDE_D,
        SOFTMAX_SCALE_LOG2,
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        QUERY_DIM,
        VALUE_HEAD_DIM,
        BLOCK_QK,
        BLOCK_D,
        IS_CAUSAL,
        HAS_LSE,
    ):
        assert BATCH_SIZE > 0
        assert NUM_Q_HEADS % NUM_KV_HEADS == 0
        assert QUERY_DIM > 0
        assert VALUE_HEAD_DIM > 0
        assert BLOCK_QK == 64
        assert BLOCK_D == 64

        qk_layout = gl.BlockedLayout([1], [BLOCK_QK], [1], [0])
        value_layout = gl.BlockedLayout([1], [BLOCK_D], [1], [0])

        self.Q_STRIDE_T = gl.constexpr(Q_STRIDE_T)
        self.Q_STRIDE_H = gl.constexpr(Q_STRIDE_H)
        self.Q_STRIDE_D = gl.constexpr(Q_STRIDE_D)
        self.K_STRIDE_T = gl.constexpr(K_STRIDE_T)
        self.K_STRIDE_H = gl.constexpr(K_STRIDE_H)
        self.K_STRIDE_D = gl.constexpr(K_STRIDE_D)
        self.V_STRIDE_T = gl.constexpr(V_STRIDE_T)
        self.V_STRIDE_H = gl.constexpr(V_STRIDE_H)
        self.V_STRIDE_D = gl.constexpr(V_STRIDE_D)
        self.OUT_STRIDE_T = gl.constexpr(OUT_STRIDE_T)
        self.OUT_STRIDE_H = gl.constexpr(OUT_STRIDE_H)
        self.OUT_STRIDE_D = gl.constexpr(OUT_STRIDE_D)
        self.SOFTMAX_SCALE_LOG2 = gl.constexpr(SOFTMAX_SCALE_LOG2)
        self.BATCH_SIZE = gl.constexpr(BATCH_SIZE)
        self.NUM_Q_HEADS = gl.constexpr(NUM_Q_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
        self.GROUP_SIZE = gl.constexpr(NUM_Q_HEADS // NUM_KV_HEADS)
        self.QUERY_DIM = gl.constexpr(QUERY_DIM)
        self.VALUE_HEAD_DIM = gl.constexpr(VALUE_HEAD_DIM)
        self.BLOCK_QK = gl.constexpr(BLOCK_QK)
        self.BLOCK_D = gl.constexpr(BLOCK_D)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.qk_layout = gl.constexpr(qk_layout)
        self.value_layout = gl.constexpr(value_layout)

    @gluon.jit
    def q_offsets(self, token, head, offs_d):
        return (
            token * self.Q_STRIDE_T
            + head * self.Q_STRIDE_H
            + offs_d * self.Q_STRIDE_D
        ).to(gl.int32)

    @gluon.jit
    def k_offsets(self, token, head, offs_d):
        return (
            token * self.K_STRIDE_T
            + head * self.K_STRIDE_H
            + offs_d * self.K_STRIDE_D
        ).to(gl.int32)

    @gluon.jit
    def v_offsets(self, token, head, offs_d):
        return (
            token * self.V_STRIDE_T
            + head * self.V_STRIDE_H
            + offs_d * self.V_STRIDE_D
        ).to(gl.int32)

    @gluon.jit
    def out_offsets(self, token, head, offs_d):
        return (
            token * self.OUT_STRIDE_T
            + head * self.OUT_STRIDE_H
            + offs_d * self.OUT_STRIDE_D
        ).to(gl.int32)


@gluon.jit
def _mla_prefill_ragged(
    q_ptr,
    k_ptr,
    v_ptr,
    cum_seq_lens_q_ptr,
    cum_seq_lens_kv_ptr,
    out_ptr,
    lse_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    K_STRIDE_T: gl.constexpr,
    K_STRIDE_H: gl.constexpr,
    K_STRIDE_D: gl.constexpr,
    V_STRIDE_T: gl.constexpr,
    V_STRIDE_H: gl.constexpr,
    V_STRIDE_D: gl.constexpr,
    OUT_STRIDE_T: gl.constexpr,
    OUT_STRIDE_H: gl.constexpr,
    OUT_STRIDE_D: gl.constexpr,
    SOFTMAX_SCALE_LOG2: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    QUERY_DIM: gl.constexpr,
    VALUE_HEAD_DIM: gl.constexpr,
    BLOCK_QK: gl.constexpr,
    BLOCK_D: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    HAS_LSE: gl.constexpr,
):
    cfg = MLAPrefillConfig(
        Q_STRIDE_T,
        Q_STRIDE_H,
        Q_STRIDE_D,
        K_STRIDE_T,
        K_STRIDE_H,
        K_STRIDE_D,
        V_STRIDE_T,
        V_STRIDE_H,
        V_STRIDE_D,
        OUT_STRIDE_T,
        OUT_STRIDE_H,
        OUT_STRIDE_D,
        SOFTMAX_SCALE_LOG2,
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        QUERY_DIM,
        VALUE_HEAD_DIM,
        BLOCK_QK,
        BLOCK_D,
        IS_CAUSAL,
        HAS_LSE,
    )
    q_token = gl.program_id(0)
    q_head = gl.program_id(1)
    value_block = gl.program_id(2)
    kv_head = q_head // cfg.GROUP_SIZE
    value_base = value_block * cfg.BLOCK_D
    value_offsets = value_base + gl.arange(
        0, cfg.BLOCK_D, layout=cfg.value_layout
    )
    valid_value = value_offsets < cfg.VALUE_HEAD_DIM

    batch = gl.full((), value=0, dtype=gl.int32)
    q_start = gl.full((), value=0, dtype=gl.int32)
    q_end = gl.full((), value=0, dtype=gl.int32)
    for candidate in range(0, cfg.BATCH_SIZE):
        candidate_start = gl.load(cum_seq_lens_q_ptr + candidate)
        candidate_end = gl.load(cum_seq_lens_q_ptr + candidate + 1)
        is_owner = (q_token >= candidate_start) & (q_token < candidate_end)
        batch = gl.where(is_owner, candidate, batch)
        q_start = gl.where(is_owner, candidate_start, q_start)
        q_end = gl.where(is_owner, candidate_end, q_end)

    kv_start = gl.load(cum_seq_lens_kv_ptr + batch)
    kv_end = gl.load(cum_seq_lens_kv_ptr + batch + 1)
    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    q_pos = q_token - q_start

    max_score = gl.full((), value=-float("inf"), dtype=gl.float32)
    for kv_pos in range(0, kv_len):
        score = gl.full((), value=0.0, dtype=gl.float32)
        for query_base in range(0, cfg.QUERY_DIM, cfg.BLOCK_QK):
            query_offsets = query_base + gl.arange(
                0, cfg.BLOCK_QK, layout=cfg.qk_layout
            )
            valid_query = query_offsets < cfg.QUERY_DIM
            q = gl.load(
                q_ptr + cfg.q_offsets(q_token, q_head, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            k = gl.load(
                k_ptr + cfg.k_offsets(kv_start + kv_pos, kv_head, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            score += gl.sum(q * k, axis=0)
        score *= cfg.SOFTMAX_SCALE_LOG2
        if cfg.IS_CAUSAL:
            causal_valid = kv_pos <= q_pos + (kv_len - q_len)
            score = gl.where(causal_valid, score, -float("inf"))
        max_score = maximum(max_score, score)

    denom = gl.full((), value=0.0, dtype=gl.float32)
    acc = gl.full(
        [cfg.BLOCK_D], value=0.0, dtype=gl.float32, layout=cfg.value_layout
    )
    for kv_pos in range(0, kv_len):
        score = gl.full((), value=0.0, dtype=gl.float32)
        for query_base in range(0, cfg.QUERY_DIM, cfg.BLOCK_QK):
            query_offsets = query_base + gl.arange(
                0, cfg.BLOCK_QK, layout=cfg.qk_layout
            )
            valid_query = query_offsets < cfg.QUERY_DIM
            q = gl.load(
                q_ptr + cfg.q_offsets(q_token, q_head, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            k = gl.load(
                k_ptr + cfg.k_offsets(kv_start + kv_pos, kv_head, query_offsets),
                mask=valid_query,
                other=0.0,
            ).to(gl.float32)
            score += gl.sum(q * k, axis=0)
        score *= cfg.SOFTMAX_SCALE_LOG2
        if cfg.IS_CAUSAL:
            causal_valid = kv_pos <= q_pos + (kv_len - q_len)
            score = gl.where(causal_valid, score, -float("inf"))
        prob = gl.exp2(score - max_score)
        if cfg.IS_CAUSAL:
            prob = gl.where(causal_valid, prob, 0.0)
        value = gl.load(
            v_ptr + cfg.v_offsets(kv_start + kv_pos, kv_head, value_offsets),
            mask=valid_value,
            other=0.0,
        ).to(gl.float32)
        acc += prob * value
        denom += prob

    output = gl.where(denom > 0.0, acc / denom, 0.0)
    output = output.to(out_ptr.dtype.element_ty)
    gl.store(
        out_ptr + cfg.out_offsets(q_token, q_head, value_offsets),
        output,
        mask=valid_value,
    )

    if cfg.HAS_LSE:
        lse = gl.where(denom > 0.0, max_score + gl.log2(denom), -float("inf"))
        gl.store(lse_ptr + q_token * cfg.NUM_Q_HEADS + q_head, lse)


@register_kernel(
    "attention",
    "mla_prefill",
    name="gluon_mla_prefill_gfx950",
    features={"mla"},
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_MLA_PREFILL_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "is_causal": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
)
def gluon_mla_prefill_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cum_seq_lens: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    softmax_scale: float,
    *,
    is_causal: bool,
    return_lse: bool,
    cum_seq_lens_q: torch.Tensor,
    max_seq_len_q: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del max_seq_len, max_seq_len_q

    output_shape = (q.shape[0], q.shape[1], v.shape[-1])
    if out is None:
        output = torch.empty(output_shape, device=q.device, dtype=torch.bfloat16)
    else:
        if out.shape != output_shape:
            raise ValueError(f"out shape must be {output_shape}, got {tuple(out.shape)}")
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out dtype must be torch.bfloat16, got {out.dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        output = out
    lse = (
        torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    if q.shape[0] == 0:
        if return_lse:
            return output, lse
        return output

    block_qk = 64
    block_d = 64
    lse_arg = lse if lse is not None else output
    grid = (q.shape[0], q.shape[1], math.ceil(v.shape[-1] / block_d))
    _mla_prefill_ragged[grid](
        q,
        k,
        v,
        cum_seq_lens_q,
        cum_seq_lens,
        output,
        lse_arg,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        softmax_scale * _INV_LN2_VALUE,
        batch_size,
        q.shape[1],
        k.shape[1],
        q.shape[-1],
        v.shape[-1],
        block_qk,
        block_d,
        is_causal,
        return_lse,
        num_warps=1,
    )
    if return_lse:
        return output, lse
    return output
