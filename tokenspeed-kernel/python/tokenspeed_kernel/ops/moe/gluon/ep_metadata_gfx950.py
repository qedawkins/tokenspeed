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

"""Expert-parallel routing metadata builder for AMD GFX950."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


_EP_METADATA_SIGNATURES = format_signatures("indices", "dense", {torch.int32})


@dataclass(frozen=True)
class EPDispatchMetadata:
    owner_counts: torch.Tensor
    owner_offsets: torch.Tensor
    owner_expert_counts: torch.Tensor
    owner_expert_offsets: torch.Tensor
    local_expert_counts: torch.Tensor
    local_expert_offsets: torch.Tensor
    dispatch_offsets: torch.Tensor
    combine_offsets: torch.Tensor


@gluon.jit
def _count_ep_metadata_slots(
    topk_ids_ptr,
    expert_owner_ptr,
    local_expert_id_ptr,
    owner_counts_ptr,
    owner_expert_counts_ptr,
    TOPK_IDS_STRIDE_M: gl.constexpr,
    TOPK_IDS_STRIDE_K: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
):
    flat_slot = gl.program_id(0)
    row = flat_slot // TOPK
    col = flat_slot - row * TOPK
    expert = gl.load(
        topk_ids_ptr + row * TOPK_IDS_STRIDE_M + col * TOPK_IDS_STRIDE_K,
    ).to(gl.int32)
    expert_valid = (expert >= 0) & (expert < NUM_EXPERTS)
    owner = gl.load(expert_owner_ptr + expert, mask=expert_valid, other=-1).to(gl.int32)
    local_id = gl.load(
        local_expert_id_ptr + expert,
        mask=expert_valid,
        other=-1,
    ).to(gl.int32)
    valid = (
        expert_valid
        & (owner >= 0)
        & (owner < WORLD_SIZE)
        & (local_id >= 0)
        & (local_id < NUM_LOCAL_EXPERTS)
    )
    gl.atomic_add(owner_counts_ptr + owner, 1, sem="relaxed", mask=valid)
    gl.atomic_add(
        owner_expert_counts_ptr + owner * NUM_LOCAL_EXPERTS + local_id,
        1,
        sem="relaxed",
        mask=valid,
    )


@gluon.jit
def _build_ep_metadata_offsets(
    owner_counts_ptr,
    owner_offsets_ptr,
    owner_expert_counts_ptr,
    owner_expert_offsets_ptr,
    write_offsets_ptr,
    WORLD_SIZE: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
):
    owner_pos = gl.full((), 0, dtype=gl.int32)
    for owner in range(0, WORLD_SIZE):
        gl.store(owner_offsets_ptr + owner, owner_pos)
        local_pos = gl.full((), 0, dtype=gl.int32)
        for local_id in range(0, NUM_LOCAL_EXPERTS):
            gl.store(
                owner_expert_offsets_ptr + owner * (NUM_LOCAL_EXPERTS + 1) + local_id,
                local_pos,
            )
            gl.store(
                write_offsets_ptr + owner * NUM_LOCAL_EXPERTS + local_id,
                local_pos,
            )
            count = gl.load(
                owner_expert_counts_ptr + owner * NUM_LOCAL_EXPERTS + local_id,
            ).to(gl.int32)
            local_pos += count
        gl.store(
            owner_expert_offsets_ptr
            + owner * (NUM_LOCAL_EXPERTS + 1)
            + NUM_LOCAL_EXPERTS,
            local_pos,
        )
        owner_pos += gl.load(owner_counts_ptr + owner).to(gl.int32)
    gl.store(owner_offsets_ptr + WORLD_SIZE, owner_pos)


@gluon.jit
def _sum_i32(a, b):
    return a + b


@gluon.jit
def _scatter_ep_metadata_offsets(
    topk_ids_ptr,
    expert_owner_ptr,
    local_expert_id_ptr,
    owner_expert_offsets_ptr,
    dispatch_offsets_ptr,
    combine_offsets_ptr,
    TOPK_IDS_STRIDE_M: gl.constexpr,
    TOPK_IDS_STRIDE_K: gl.constexpr,
    DISPATCH_OFFSETS_STRIDE_M: gl.constexpr,
    DISPATCH_OFFSETS_STRIDE_K: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_N_ELEMS_PER_THREAD: gl.constexpr,
):
    bucket = gl.program_id(0)
    bucket_owner = bucket // NUM_LOCAL_EXPERTS
    bucket_local_id = bucket - bucket_owner * NUM_LOCAL_EXPERTS
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    write_pos = gl.load(
        owner_expert_offsets_ptr
        + bucket_owner * (NUM_LOCAL_EXPERTS + 1)
        + bucket_local_id,
    ).to(gl.int32)

    for base in range(0, NUM_SLOTS, BLOCK_N):
        flat_slot = base + offsets
        active = flat_slot < NUM_SLOTS
        row = flat_slot // TOPK
        col = flat_slot - row * TOPK
        expert = gl.load(
            topk_ids_ptr + row * TOPK_IDS_STRIDE_M + col * TOPK_IDS_STRIDE_K,
            mask=active,
            other=-1,
        ).to(gl.int32)
        expert_valid = (expert >= 0) & (expert < NUM_EXPERTS)
        owner = gl.load(expert_owner_ptr + expert, mask=expert_valid, other=-1).to(
            gl.int32
        )
        local_id = gl.load(
            local_expert_id_ptr + expert,
            mask=expert_valid,
            other=-1,
        ).to(gl.int32)
        match = (
            active
            & expert_valid
            & (owner == bucket_owner)
            & (local_id == bucket_local_id)
        )
        match_i32 = match.to(gl.int32)
        local_rows = (
            gl.associative_scan(match_i32, axis=0, combine_fn=_sum_i32) - match_i32
        )
        owner_rows = write_pos + local_rows
        gl.store(
            dispatch_offsets_ptr
            + row * DISPATCH_OFFSETS_STRIDE_M
            + col * DISPATCH_OFFSETS_STRIDE_K,
            owner_rows,
            mask=match,
        )
        gl.store(
            combine_offsets_ptr + bucket_owner * NUM_SLOTS + owner_rows,
            flat_slot.to(gl.int32),
            mask=match,
        )
        write_pos += gl.sum(match_i32, axis=0)


@gluon.jit
def _copy_rank_local_metadata(
    owner_expert_counts_ptr,
    owner_expert_offsets_ptr,
    local_expert_counts_ptr,
    local_expert_offsets_ptr,
    RANK: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_N_ELEMS_PER_THREAD: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    active_counts = offsets < NUM_LOCAL_EXPERTS
    gl.store(
        local_expert_counts_ptr + offsets,
        gl.load(
            owner_expert_counts_ptr + RANK * NUM_LOCAL_EXPERTS + offsets,
            mask=active_counts,
            other=0,
        ).to(gl.int32),
        mask=active_counts,
    )
    active_offsets = offsets < NUM_LOCAL_EXPERTS + 1
    gl.store(
        local_expert_offsets_ptr + offsets,
        gl.load(
            owner_expert_offsets_ptr + RANK * (NUM_LOCAL_EXPERTS + 1) + offsets,
            mask=active_offsets,
            other=0,
        ).to(gl.int32),
        mask=active_offsets,
    )


def _infer_num_local_experts(
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    rank: int,
) -> int:
    owner_cpu = expert_owner.detach().cpu()
    valid_local_ids = local_expert_id.detach().cpu()
    valid_local_ids = valid_local_ids[(owner_cpu == rank) & (valid_local_ids >= 0)]
    if valid_local_ids.numel() == 0:
        raise ValueError(f"rank {rank} must own at least one non-negative local expert")
    return int(valid_local_ids.max().item()) + 1


@register_kernel(
    "moe",
    "dispatch",
    name="gluon_ep_metadata_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_EP_METADATA_SIGNATURES,
    traits={"comm_strategy": frozenset({"ep_metadata"})},
    priority=Priority.SPECIALIZED,
    tags={"latency"},
)
def gluon_ep_metadata_gfx950(
    topk_ids: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: Optional[int] = None,
) -> EPDispatchMetadata:
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got shape {tuple(topk_ids.shape)}")
    if expert_owner.ndim != 1 or local_expert_id.ndim != 1:
        raise ValueError("expert_owner and local_expert_id must be rank-1 tensors")
    if expert_owner.shape != local_expert_id.shape:
        raise ValueError(
            "expert_owner and local_expert_id must have matching shapes, got "
            f"{tuple(expert_owner.shape)} and {tuple(local_expert_id.shape)}"
        )
    for name, tensor in (
        ("topk_ids", topk_ids),
        ("expert_owner", expert_owner),
        ("local_expert_id", local_expert_id),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")
        if tensor.device != topk_ids.device:
            raise ValueError(f"{name} must be on the same device as topk_ids")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if num_local_experts is None:
        num_local_experts = _infer_num_local_experts(
            expert_owner,
            local_expert_id,
            rank,
        )
    if num_local_experts <= 0:
        raise ValueError(
            f"num_local_experts must be positive, got {num_local_experts}"
        )

    num_tokens, topk = topk_ids.shape
    num_slots = num_tokens * topk
    num_experts = expert_owner.numel()
    device = topk_ids.device
    owner_counts = torch.zeros((world_size,), dtype=torch.int32, device=device)
    owner_offsets = torch.empty((world_size + 1,), dtype=torch.int32, device=device)
    owner_expert_counts = torch.zeros(
        (world_size, num_local_experts),
        dtype=torch.int32,
        device=device,
    )
    owner_expert_offsets = torch.empty(
        (world_size, num_local_experts + 1),
        dtype=torch.int32,
        device=device,
    )
    local_expert_counts = torch.empty(
        (num_local_experts,),
        dtype=torch.int32,
        device=device,
    )
    local_expert_offsets = torch.empty(
        (num_local_experts + 1,),
        dtype=torch.int32,
        device=device,
    )
    write_offsets = torch.empty(
        (world_size, num_local_experts),
        dtype=torch.int32,
        device=device,
    )
    dispatch_offsets = torch.full(
        topk_ids.shape,
        -1,
        dtype=torch.int32,
        device=device,
    )
    combine_offsets = torch.full(
        (world_size, num_slots),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_slots > 0:
        _count_ep_metadata_slots[(num_slots,)](
            topk_ids,
            expert_owner,
            local_expert_id,
            owner_counts,
            owner_expert_counts,
            topk_ids.stride(0),
            topk_ids.stride(1),
            num_slots,
            topk,
            num_experts,
            world_size,
            num_local_experts,
            num_warps=1,
        )
    _build_ep_metadata_offsets[(1,)](
        owner_counts,
        owner_offsets,
        owner_expert_counts,
        owner_expert_offsets,
        write_offsets,
        world_size,
        num_local_experts,
        num_warps=1,
    )
    if num_slots > 0:
        block_n = 256
        _scatter_ep_metadata_offsets[(world_size * num_local_experts,)](
            topk_ids,
            expert_owner,
            local_expert_id,
            owner_expert_offsets,
            dispatch_offsets,
            combine_offsets,
            topk_ids.stride(0),
            topk_ids.stride(1),
            dispatch_offsets.stride(0),
            dispatch_offsets.stride(1),
            num_slots,
            topk,
            num_experts,
            world_size,
            num_local_experts,
            block_n,
            block_n // 64,
            num_warps=1,
        )
    _copy_rank_local_metadata[(1,)](
        owner_expert_counts,
        owner_expert_offsets,
        local_expert_counts,
        local_expert_offsets,
        rank,
        num_local_experts,
        max(64, num_local_experts + 1),
        1,
        num_warps=1,
    )

    return EPDispatchMetadata(
        owner_counts=owner_counts,
        owner_offsets=owner_offsets,
        owner_expert_counts=owner_expert_counts,
        owner_expert_offsets=owner_expert_offsets,
        local_expert_counts=local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        dispatch_offsets=dispatch_offsets,
        combine_offsets=combine_offsets,
    )
