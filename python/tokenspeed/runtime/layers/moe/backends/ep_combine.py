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

"""Internal owner-output return for expert-parallel MoE.

Source ranks use their local slot metadata to gather owner-rank expert outputs
back into ``[tokens, top_k, hidden]`` slot order. The returned tensor remains
unweighted so the existing weighted reduce/finalize path can consume it.
"""

from __future__ import annotations

from typing import Any

import torch

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import EPOwnerDispatchPlan
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
    EPWorkspaceUnavailable,
)
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

try:
    with redirect_triton_to_tokenspeed_triton():
        import triton
        from iris.gluon import IrisDeviceCtx
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
except ImportError:
    triton = None
    IrisDeviceCtx = None
    gluon = None
    gl = None


if gluon is not None:

    @gluon.jit
    def _owner_directed_combine_kernel(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        dst_ptr,
        owner_outputs_ptr,
        topk_ids_ptr,
        expert_owner_ptr,
        local_expert_id_ptr,
        dispatch_offsets_ptr,
        source_owner_expert_offsets_ptr,
        aggregate_owner_expert_offsets_ptr,
        owner_expert_base_offsets_ptr,
        DST_STRIDE_M: gl.constexpr,
        DST_STRIDE_K: gl.constexpr,
        DST_STRIDE_H: gl.constexpr,
        OWNER_OUTPUTS_STRIDE_M: gl.constexpr,
        TOPK_IDS_STRIDE_M: gl.constexpr,
        TOPK_IDS_STRIDE_K: gl.constexpr,
        DISPATCH_OFFSETS_STRIDE_M: gl.constexpr,
        DISPATCH_OFFSETS_STRIDE_K: gl.constexpr,
        SOURCE_OWNER_EXPERT_STRIDE_OWNER: gl.constexpr,
        SOURCE_OWNER_EXPERT_STRIDE_LOCAL: gl.constexpr,
        AGGREGATE_OWNER_EXPERT_STRIDE_OWNER: gl.constexpr,
        AGGREGATE_OWNER_EXPERT_STRIDE_LOCAL: gl.constexpr,
        OWNER_EXPERT_BASE_STRIDE_OWNER: gl.constexpr,
        OWNER_EXPERT_BASE_STRIDE_LOCAL: gl.constexpr,
        NUM_TOKENS: gl.constexpr,
        TOPK: gl.constexpr,
        HIDDEN_SIZE: gl.constexpr,
        NUM_EXPERTS: gl.constexpr,
        WORLD_SIZE: gl.constexpr,
        NUM_LOCAL_EXPERTS: gl.constexpr,
        MAX_OWNER_ROWS: gl.constexpr,
        BLOCK_H: gl.constexpr,
        BLOCK_H_ELEMS_PER_THREAD: gl.constexpr,
    ):
        ctx = IrisDeviceCtx.initialize(context_tensor)
        token = gl.program_id(0)
        topk_idx = gl.program_id(1)
        hidden_block = gl.program_id(2)

        layout: gl.constexpr = gl.BlockedLayout(
            [BLOCK_H_ELEMS_PER_THREAD], [64], [1], [0]
        )
        offsets = hidden_block * BLOCK_H + gl.arange(0, BLOCK_H, layout=layout)
        mask_h = offsets < HIDDEN_SIZE
        active = (token < NUM_TOKENS) & (topk_idx < TOPK)

        expert = gl.load(
            topk_ids_ptr
            + token * TOPK_IDS_STRIDE_M
            + topk_idx * TOPK_IDS_STRIDE_K,
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
        mapping_valid = (
            expert_valid
            & (owner >= 0)
            & (owner < WORLD_SIZE)
            & (local_id >= 0)
            & (local_id < NUM_LOCAL_EXPERTS)
        )
        source_owner_row = gl.load(
            dispatch_offsets_ptr
            + token * DISPATCH_OFFSETS_STRIDE_M
            + topk_idx * DISPATCH_OFFSETS_STRIDE_K,
            mask=active,
            other=-1,
        ).to(gl.int32)
        local_start = gl.load(
            source_owner_expert_offsets_ptr
            + owner * SOURCE_OWNER_EXPERT_STRIDE_OWNER
            + local_id * SOURCE_OWNER_EXPERT_STRIDE_LOCAL,
            mask=mapping_valid,
            other=0,
        ).to(gl.int32)
        aggregate_start = gl.load(
            aggregate_owner_expert_offsets_ptr
            + owner * AGGREGATE_OWNER_EXPERT_STRIDE_OWNER
            + local_id * AGGREGATE_OWNER_EXPERT_STRIDE_LOCAL,
            mask=mapping_valid,
            other=0,
        ).to(gl.int32)
        source_base = gl.load(
            owner_expert_base_offsets_ptr
            + owner * OWNER_EXPERT_BASE_STRIDE_OWNER
            + local_id * OWNER_EXPERT_BASE_STRIDE_LOCAL,
            mask=mapping_valid,
            other=0,
        ).to(gl.int32)
        owner_row = aggregate_start + source_base + source_owner_row - local_start
        row_valid = (
            mapping_valid
            & (source_owner_row >= 0)
            & (owner_row >= 0)
            & (owner_row < MAX_OWNER_ROWS)
        )
        values = ctx.load(
            owner_outputs_ptr + owner_row * OWNER_OUTPUTS_STRIDE_M + offsets,
            owner,
            mask=row_valid & mask_h,
            other=0.0,
        )
        gl.store(
            dst_ptr
            + token * DST_STRIDE_M
            + topk_idx * DST_STRIDE_K
            + offsets * DST_STRIDE_H,
            values,
            mask=row_valid & mask_h,
        )


def owner_directed_combine(
    owner_outputs: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    synchronize: bool = True,
) -> torch.Tensor:
    _validate_combine_inputs(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
        out,
    )
    if out is None:
        out = torch.empty(
            (*topk_ids.shape, owner_outputs.shape[1]),
            dtype=owner_outputs.dtype,
            device=owner_outputs.device,
        )
    out.zero_()

    if topk_ids.numel() == 0:
        if workspace.backend == "iris" and synchronize:
            workspace.barrier()
            workspace.barrier()
        return out

    if workspace.backend == "iris":
        if synchronize:
            workspace.barrier()
        _combine_with_iris_gluon(
            out,
            topk_ids,
            metadata,
            dispatch_plan,
            workspace,
            expert_owner,
            local_expert_id,
        )
        if synchronize:
            workspace.barrier()
    else:
        _combine_local_torch(
            owner_outputs,
            out,
            topk_ids,
            metadata,
            dispatch_plan,
            workspace,
            expert_owner,
            local_expert_id,
        )
    return out


def _combine_with_iris_gluon(
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
) -> None:
    if _owner_directed_combine_kernel is None or IrisDeviceCtx is None or triton is None:
        raise EPWorkspaceUnavailable("Iris Gluon combine support is not importable")
    if workspace.handle.device_context is None:
        raise EPWorkspaceUnavailable("Iris-backed workspace has no device context")
    hidden_size = out.shape[-1]
    block_h = min(triton.next_power_of_2(hidden_size), 512)
    grid = (
        topk_ids.shape[0],
        topk_ids.shape[1],
        triton.cdiv(hidden_size, block_h),
    )
    _owner_directed_combine_kernel[grid](
        IrisDeviceCtx,
        workspace.handle.device_context,
        out,
        workspace.combine_buffer,
        topk_ids,
        expert_owner,
        local_expert_id,
        metadata.dispatch_offsets,
        metadata.owner_expert_offsets,
        dispatch_plan.aggregate_owner_expert_offsets,
        dispatch_plan.owner_expert_base_offsets,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        workspace.combine_buffer.stride(0),
        topk_ids.stride(0),
        topk_ids.stride(1),
        metadata.dispatch_offsets.stride(0),
        metadata.dispatch_offsets.stride(1),
        metadata.owner_expert_offsets.stride(0),
        metadata.owner_expert_offsets.stride(1),
        dispatch_plan.aggregate_owner_expert_offsets.stride(0),
        dispatch_plan.aggregate_owner_expert_offsets.stride(1),
        dispatch_plan.owner_expert_base_offsets.stride(0),
        dispatch_plan.owner_expert_base_offsets.stride(1),
        topk_ids.shape[0],
        topk_ids.shape[1],
        hidden_size,
        expert_owner.numel(),
        workspace.world_size,
        dispatch_plan.owner_expert_base_offsets.shape[1],
        workspace.max_dispatch_rows,
        block_h,
        max(block_h // 64, 1),
        num_warps=1,
    )


def _combine_local_torch(
    owner_outputs: torch.Tensor,
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
) -> None:
    flat_topk = topk_ids.reshape(-1)
    flat_dispatch = metadata.dispatch_offsets.reshape(-1)
    flat_out = out.reshape(-1, out.shape[-1])
    for flat_slot, expert_tensor in enumerate(flat_topk):
        expert = int(expert_tensor.item())
        if expert < 0 or expert >= expert_owner.numel():
            continue
        owner = int(expert_owner[expert].item())
        local_id = int(local_expert_id[expert].item())
        if owner < 0 or owner >= workspace.world_size or local_id < 0:
            continue
        if owner != workspace.rank:
            raise EPWorkspaceUnavailable(
                "torch EP combine can only return local-owner rows; use Iris for "
                "remote owner rows"
            )
        source_owner_row = int(flat_dispatch[flat_slot].item())
        if source_owner_row < 0:
            continue
        owner_row = _owner_output_row(
            source_owner_row,
            owner,
            local_id,
            metadata,
            dispatch_plan,
        )
        if owner_row < 0 or owner_row >= owner_outputs.shape[0]:
            raise EPWorkspaceError(
                f"owner output row {owner_row} is outside owner_outputs rows "
                f"{owner_outputs.shape[0]}"
            )
        flat_out[flat_slot].copy_(owner_outputs[owner_row])


def _owner_output_row(
    source_owner_row: int,
    owner: int,
    local_id: int,
    metadata: Any,
    dispatch_plan: EPOwnerDispatchPlan,
) -> int:
    local_start = int(metadata.owner_expert_offsets[owner, local_id].item())
    aggregate_start = int(
        dispatch_plan.aggregate_owner_expert_offsets[owner, local_id].item()
    )
    source_base = int(dispatch_plan.owner_expert_base_offsets[owner, local_id].item())
    return aggregate_start + source_base + source_owner_row - local_start


def _validate_combine_inputs(
    owner_outputs: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if owner_outputs.ndim != 2:
        raise EPWorkspaceError(
            f"owner_outputs must be rank-2, got {tuple(owner_outputs.shape)}"
        )
    if owner_outputs.dtype != workspace.dtype:
        raise EPWorkspaceError(
            f"owner_outputs dtype {owner_outputs.dtype} != workspace dtype "
            f"{workspace.dtype}"
        )
    if owner_outputs.device != workspace.device:
        raise EPWorkspaceError(
            f"owner_outputs device {owner_outputs.device} != workspace device "
            f"{workspace.device}"
        )
    if (
        workspace.backend == "iris"
        and owner_outputs.data_ptr() != workspace.combine_buffer.data_ptr()
    ):
        raise EPWorkspaceError(
            "Iris EP combine requires owner_outputs to start at workspace.combine_buffer"
        )
    workspace.check_capacity(owner_outputs.shape[0], owner_outputs.shape[1])
    if topk_ids.ndim != 2:
        raise EPWorkspaceError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_ids.dtype != torch.int32:
        raise EPWorkspaceError(f"topk_ids must be torch.int32, got {topk_ids.dtype}")
    if topk_ids.device != workspace.device:
        raise EPWorkspaceError(
            f"topk_ids device {topk_ids.device} != workspace device {workspace.device}"
        )
    if metadata.dispatch_offsets.shape != topk_ids.shape:
        raise EPWorkspaceError(
            f"dispatch_offsets shape {tuple(metadata.dispatch_offsets.shape)} != "
            f"{tuple(topk_ids.shape)}"
        )
    if metadata.dispatch_offsets.dtype != torch.int32:
        raise EPWorkspaceError(
            f"dispatch_offsets must be torch.int32, got {metadata.dispatch_offsets.dtype}"
        )
    if metadata.dispatch_offsets.device != workspace.device:
        raise EPWorkspaceError(
            "dispatch_offsets must be on the same device as the EP workspace"
        )
    _validate_owner_expert_offsets(metadata.owner_expert_offsets, dispatch_plan, workspace)
    for name, tensor in (
        ("expert_owner", expert_owner),
        ("local_expert_id", local_expert_id),
    ):
        if tensor.ndim != 1:
            raise EPWorkspaceError(f"{name} must be rank-1, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise EPWorkspaceError(f"{name} must be torch.int32, got {tensor.dtype}")
        if tensor.device != workspace.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != workspace device {workspace.device}"
            )
    if expert_owner.shape != local_expert_id.shape:
        raise EPWorkspaceError(
            f"expert_owner shape {tuple(expert_owner.shape)} != local_expert_id "
            f"shape {tuple(local_expert_id.shape)}"
        )
    if out is not None:
        expected_out_shape = (*topk_ids.shape, owner_outputs.shape[1])
        if out.shape != expected_out_shape:
            raise EPWorkspaceError(f"out shape {tuple(out.shape)} != {expected_out_shape}")
        if out.dtype != owner_outputs.dtype:
            raise EPWorkspaceError(f"out dtype {out.dtype} != {owner_outputs.dtype}")
        if out.device != workspace.device:
            raise EPWorkspaceError(
                f"out device {out.device} != workspace device {workspace.device}"
            )


def _validate_owner_expert_offsets(
    owner_expert_offsets: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
) -> None:
    expected_shape = dispatch_plan.aggregate_owner_expert_offsets.shape
    if owner_expert_offsets.shape != expected_shape:
        raise EPWorkspaceError(
            f"owner_expert_offsets shape {tuple(owner_expert_offsets.shape)} != "
            f"{tuple(expected_shape)}"
        )
    for name, tensor in (
        ("owner_expert_offsets", owner_expert_offsets),
        ("owner_expert_base_offsets", dispatch_plan.owner_expert_base_offsets),
        (
            "aggregate_owner_expert_offsets",
            dispatch_plan.aggregate_owner_expert_offsets,
        ),
    ):
        if tensor.dtype != torch.int32:
            raise EPWorkspaceError(f"{name} must be torch.int32, got {tensor.dtype}")
        if tensor.device != workspace.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != workspace device {workspace.device}"
            )


__all__ = ["owner_directed_combine"]
