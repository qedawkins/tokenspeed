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

"""Local MoE sorted-dispatch metadata Gluon kernel for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


_DISPATCH_SIGNATURES = format_signatures("indices", "dense", {torch.int32})


@gluon.jit
def _count_dispatch_slots(
    topk_ids_ptr,
    counts_ptr,
    TOPK_IDS_STRIDE_M: gl.constexpr,
    TOPK_IDS_STRIDE_K: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_N_ELEMS_PER_THREAD: gl.constexpr,
):
    expert_id = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    total = gl.full((), 0, dtype=gl.int32)

    for base in range(0, NUM_SLOTS, BLOCK_N):
        flat_slot = base + offsets
        active = flat_slot < NUM_SLOTS
        row = flat_slot // TOPK
        col = flat_slot - row * TOPK
        topk_id = gl.load(
            topk_ids_ptr + row * TOPK_IDS_STRIDE_M + col * TOPK_IDS_STRIDE_K,
            mask=active,
            other=-1,
        ).to(gl.int32)
        invalid = (topk_id < 0) | (topk_id >= NUM_EXPERTS)
        match = gl.where(expert_id == NUM_EXPERTS, invalid, topk_id == expert_id)
        total += gl.sum((match & active).to(gl.int32), axis=0)

    gl.store(counts_ptr + expert_id, total)


@gluon.jit
def _build_dispatch_offsets(
    counts_ptr,
    offsets_ptr,
    write_offsets_ptr,
    num_tokens_post_pad_ptr,
    BLOCK_SIZE: gl.constexpr,
    NUM_EXPERT_BUCKETS: gl.constexpr,
):
    write_pos = gl.full((), 0, dtype=gl.int32)
    for expert_id in range(0, NUM_EXPERT_BUCKETS):
        gl.store(offsets_ptr + expert_id, write_pos)
        gl.store(write_offsets_ptr + expert_id, write_pos)
        count = gl.load(counts_ptr + expert_id).to(gl.int32)
        num_blocks = (count + BLOCK_SIZE - 1) // BLOCK_SIZE
        write_pos += num_blocks * BLOCK_SIZE

    gl.store(offsets_ptr + NUM_EXPERT_BUCKETS, write_pos)
    gl.store(num_tokens_post_pad_ptr, write_pos)


@gluon.jit
def _fill_dispatch_expert_ids(
    offsets_ptr,
    num_tokens_post_pad_ptr,
    expert_ids_ptr,
    BLOCK_SIZE: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    NUM_EXPERT_BUCKETS: gl.constexpr,
):
    block_id = gl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    num_tokens_post_pad = gl.load(num_tokens_post_pad_ptr).to(gl.int32)
    stored_expert = gl.full((), 0, dtype=gl.int32)

    if block_start < num_tokens_post_pad:
        for expert_id in range(0, NUM_EXPERT_BUCKETS):
            start = gl.load(offsets_ptr + expert_id).to(gl.int32)
            end = gl.load(offsets_ptr + expert_id + 1).to(gl.int32)
            owns_block = (block_start >= start) & (block_start < end)
            expert_value = gl.where(expert_id == NUM_EXPERTS, -1, expert_id)
            stored_expert = gl.where(owns_block, expert_value, stored_expert)

    gl.store(expert_ids_ptr + block_id, stored_expert)


@gluon.jit
def _scatter_dispatch_slots(
    topk_ids_ptr,
    write_offsets_ptr,
    sorted_ids_ptr,
    TOPK_IDS_STRIDE_M: gl.constexpr,
    TOPK_IDS_STRIDE_K: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
):
    flat_slot = gl.program_id(0)
    row = flat_slot // TOPK
    col = flat_slot - row * TOPK
    topk_id = gl.load(
        topk_ids_ptr + row * TOPK_IDS_STRIDE_M + col * TOPK_IDS_STRIDE_K,
    ).to(gl.int32)
    invalid = (topk_id < 0) | (topk_id >= NUM_EXPERTS)
    bucket = gl.where(invalid, NUM_EXPERTS, topk_id)
    write_pos = gl.atomic_add(
        write_offsets_ptr + bucket,
        1,
        sem="relaxed",
    )
    gl.store(sorted_ids_ptr + write_pos, flat_slot.to(gl.int32))


@register_kernel(
    "moe",
    "dispatch",
    name="gluon_local_dispatch_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_DISPATCH_SIGNATURES,
    traits={"comm_strategy": frozenset({"local"})},
    priority=Priority.SPECIALIZED,
    tags={"latency"},
)
def gluon_local_dispatch_gfx950(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got shape {tuple(topk_ids.shape)}")
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be torch.int32, got {topk_ids.dtype}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    total_tokens, topk = topk_ids.shape
    pad_id = total_tokens * topk
    max_num_tokens_padded = pad_id + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_ids = torch.full(
        (max_num_tokens_padded,),
        pad_id,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.zeros(
        (max_num_m_blocks,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    counts = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    offsets = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    write_offsets = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )

    block_n = 256
    _count_dispatch_slots[(num_experts + 1,)](
        topk_ids,
        counts,
        topk_ids.stride(0),
        topk_ids.stride(1),
        pad_id,
        topk,
        num_experts,
        block_n,
        block_n // 64,
        num_warps=1,
    )
    _build_dispatch_offsets[(1,)](
        counts,
        offsets,
        write_offsets,
        num_tokens_post_pad,
        block_size,
        num_experts + 1,
        num_warps=1,
    )
    if max_num_m_blocks > 0:
        _fill_dispatch_expert_ids[(max_num_m_blocks,)](
            offsets,
            num_tokens_post_pad,
            expert_ids,
            block_size,
            num_experts,
            num_experts + 1,
            num_warps=1,
    )
    if pad_id > 0:
        grid = (pad_id,)
        _scatter_dispatch_slots[grid](
            topk_ids,
            write_offsets,
            sorted_ids,
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk,
            num_experts,
            num_warps=1,
        )

    return sorted_ids, expert_ids, num_tokens_post_pad
