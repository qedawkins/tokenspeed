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

"""Local MoE weighted-slot combine Gluon kernel for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


_COMBINE_SIGNATURES = format_signatures(
    "x", "dense", {torch.float16, torch.bfloat16}
)


@gluon.jit
def _local_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    INPUT_STRIDE_M: gl.constexpr,
    INPUT_STRIDE_K: gl.constexpr,
    INPUT_STRIDE_H: gl.constexpr,
    OUTPUT_STRIDE_M: gl.constexpr,
    OUTPUT_STRIDE_H: gl.constexpr,
    NUM_TOKENS: gl.constexpr,
    TOPK: gl.constexpr,
    HIDDEN: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_H_ELEMS_PER_THREAD: gl.constexpr,
):
    token = gl.program_id(0)
    hidden_block = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_H_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offs_h = hidden_block * BLOCK_H + gl.arange(0, BLOCK_H, layout=layout)
    mask = (token < NUM_TOKENS) & (offs_h < HIDDEN)
    acc = gl.full([BLOCK_H], value=0.0, dtype=gl.float32, layout=layout)

    for topk_idx in range(0, TOPK):
        values = gl.load(
            input_ptr
            + token * INPUT_STRIDE_M
            + topk_idx * INPUT_STRIDE_K
            + offs_h * INPUT_STRIDE_H,
            mask=mask,
            other=0.0,
        ).to(gl.float32)
        acc += values

    acc *= ROUTED_SCALING_FACTOR
    gl.store(
        output_ptr + token * OUTPUT_STRIDE_M + offs_h * OUTPUT_STRIDE_H,
        acc.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@register_kernel(
    "moe",
    "combine",
    name="gluon_local_sum_reduce_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_COMBINE_SIGNATURES,
    traits={"comm_strategy": frozenset({None})},
    priority=Priority.SPECIALIZED,
    tags={"latency"},
)
def gluon_local_sum_reduce_gfx950(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    if input.ndim != 3:
        raise ValueError(f"input must be rank-3 [M, K, H], got {tuple(input.shape)}")
    if output.ndim != 2:
        raise ValueError(f"output must be rank-2 [M, H], got {tuple(output.shape)}")

    num_tokens, topk, hidden = input.shape
    if output.shape != (num_tokens, hidden):
        raise ValueError(
            f"output shape {tuple(output.shape)} does not match input [M,H] "
            f"{(num_tokens, hidden)}"
        )
    if num_tokens == 0 or hidden == 0:
        return

    block_h = min(1024, max(64, 1 << (hidden - 1).bit_length()))
    grid = (num_tokens, (hidden + block_h - 1) // block_h)
    _local_sum_reduce_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        num_tokens,
        topk,
        hidden,
        float(routed_scaling_factor),
        block_h,
        block_h // 64,
        num_warps=1,
    )
