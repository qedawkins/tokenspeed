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

"""Internal owner-directed EP dispatch.

This is not a public MoE backend mode. It is the first internal consumer of the
S4 EP metadata and workspace ABI: source ranks copy selected token rows to the
expert-owner rank's dispatch buffer while preserving the metadata row order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
    EPWorkspaceStep,
    EPWorkspaceUnavailable,
)
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton


@dataclass(frozen=True)
class EPOwnerDispatchPlan:
    owner_base_offsets: torch.Tensor
    owner_expert_base_offsets: torch.Tensor
    aggregate_owner_expert_counts: torch.Tensor
    aggregate_owner_expert_offsets: torch.Tensor
    local_expert_counts: torch.Tensor
    local_expert_offsets: torch.Tensor


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
    def _owner_directed_dispatch_kernel(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        dst_ptr,
        src_ptr,
        combine_offsets_ptr,
        owner_counts_ptr,
        owner_expert_offsets_ptr,
        owner_expert_base_offsets_ptr,
        aggregate_owner_expert_offsets_ptr,
        DST_STRIDE_M: gl.constexpr,
        SRC_STRIDE_M: gl.constexpr,
        COMBINE_STRIDE_OWNER: gl.constexpr,
        COMBINE_STRIDE_ROW: gl.constexpr,
        OWNER_EXPERT_STRIDE_OWNER: gl.constexpr,
        OWNER_EXPERT_STRIDE_LOCAL: gl.constexpr,
        OWNER_EXPERT_BASE_STRIDE_OWNER: gl.constexpr,
        OWNER_EXPERT_BASE_STRIDE_LOCAL: gl.constexpr,
        AGGREGATE_OWNER_EXPERT_STRIDE_OWNER: gl.constexpr,
        AGGREGATE_OWNER_EXPERT_STRIDE_LOCAL: gl.constexpr,
        HIDDEN_SIZE: gl.constexpr,
        TOPK: gl.constexpr,
        MAX_OWNER_ROWS: gl.constexpr,
        NUM_LOCAL_EXPERTS: gl.constexpr,
        BLOCK_N: gl.constexpr,
        BLOCK_N_ELEMS_PER_THREAD: gl.constexpr,
    ):
        ctx = IrisDeviceCtx.initialize(context_tensor)
        owner = gl.program_id(0)
        owner_row = gl.program_id(1)
        hidden_block = gl.program_id(2)

        layout: gl.constexpr = gl.BlockedLayout(
            [BLOCK_N_ELEMS_PER_THREAD], [64], [1], [0]
        )
        offsets = hidden_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
        mask_n = offsets < HIDDEN_SIZE
        owner_count = gl.load(owner_counts_ptr + owner).to(gl.int32)

        if owner_row < owner_count:
            flat_slot = gl.load(
                combine_offsets_ptr
                + owner * COMBINE_STRIDE_OWNER
                + owner_row * COMBINE_STRIDE_ROW,
            ).to(gl.int32)
            if flat_slot >= 0:
                local_id = gl.full((), 0, dtype=gl.int32)
                local_start = gl.full((), 0, dtype=gl.int32)
                for candidate in range(0, NUM_LOCAL_EXPERTS):
                    start = gl.load(
                        owner_expert_offsets_ptr
                        + owner * OWNER_EXPERT_STRIDE_OWNER
                        + candidate * OWNER_EXPERT_STRIDE_LOCAL,
                    ).to(gl.int32)
                    end = gl.load(
                        owner_expert_offsets_ptr
                        + owner * OWNER_EXPERT_STRIDE_OWNER
                        + (candidate + 1) * OWNER_EXPERT_STRIDE_LOCAL,
                    ).to(gl.int32)
                    if (owner_row >= start) & (owner_row < end):
                        local_id = candidate
                        local_start = start
                token = flat_slot // TOPK
                expert_base = gl.load(
                    aggregate_owner_expert_offsets_ptr
                    + owner * AGGREGATE_OWNER_EXPERT_STRIDE_OWNER
                    + local_id * AGGREGATE_OWNER_EXPERT_STRIDE_LOCAL,
                ).to(gl.int32)
                source_expert_base = gl.load(
                    owner_expert_base_offsets_ptr
                    + owner * OWNER_EXPERT_BASE_STRIDE_OWNER
                    + local_id * OWNER_EXPERT_BASE_STRIDE_LOCAL,
                ).to(gl.int32)
                dst_row = expert_base + source_expert_base + owner_row - local_start
                values = gl.load(
                    src_ptr + token * SRC_STRIDE_M + offsets,
                    mask=mask_n,
                    other=0.0,
                )
                ctx.store(
                    dst_ptr + dst_row * DST_STRIDE_M + offsets,
                    values,
                    owner,
                    mask=mask_n,
                )


def owner_directed_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
    *,
    dispatch_plan: EPOwnerDispatchPlan | None = None,
    owner_base_offsets: torch.Tensor | None = None,
    synchronize: bool = True,
) -> EPWorkspaceStep:
    _validate_dispatch_inputs(hidden_states, topk_ids, metadata, workspace)
    if dispatch_plan is not None and owner_base_offsets is not None:
        raise EPWorkspaceError("provide either dispatch_plan or owner_base_offsets, not both")
    if dispatch_plan is None and owner_base_offsets is None:
        dispatch_plan = prepare_owner_directed_dispatch(metadata, workspace)
    elif dispatch_plan is None:
        _validate_owner_base_offsets(owner_base_offsets, workspace)
        dispatch_plan = _source_major_compat_plan(
            owner_base_offsets,
            metadata,
            workspace,
        )
    else:
        _validate_dispatch_plan(dispatch_plan, metadata, workspace)

    if workspace.backend == "iris":
        _dispatch_with_iris_gluon(
            hidden_states,
            topk_ids,
            metadata,
            workspace,
            dispatch_plan,
        )
        if _should_synchronize_workspace(synchronize):
            workspace.barrier()
    else:
        _dispatch_local_torch(
            hidden_states,
            topk_ids,
            metadata,
            workspace,
            dispatch_plan,
        )

    return workspace.view(_dispatch_view_rows(dispatch_plan, workspace))


def prepare_owner_directed_dispatch(
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> EPOwnerDispatchPlan:
    owner_counts = metadata.owner_counts
    _validate_owner_counts(owner_counts, workspace)
    _validate_owner_expert_metadata(metadata, workspace)

    if workspace.world_size > 1 and dist.is_available() and dist.is_initialized():
        dist_world_size = dist.get_world_size()
        context_ranks = _workspace_context_ranks(workspace, dist_world_size)
        gathered = [torch.empty_like(owner_counts) for _ in range(dist_world_size)]
        dist.all_gather(gathered, owner_counts.contiguous())
        all_counts = torch.stack(gathered, dim=0).index_select(0, context_ranks)
        recv_counts = all_counts[:, workspace.rank].contiguous()
        owner_base_offsets = all_counts[: workspace.rank, :].sum(dim=0).to(torch.int32)

        owner_expert_counts = metadata.owner_expert_counts.contiguous()
        expert_gathered = [
            torch.empty_like(owner_expert_counts) for _ in range(dist_world_size)
        ]
        dist.all_gather(expert_gathered, owner_expert_counts)
        all_expert_counts = torch.stack(expert_gathered, dim=0).index_select(
            0,
            context_ranks,
        )
        owner_expert_base_offsets = all_expert_counts[: workspace.rank].sum(dim=0).to(
            torch.int32
        )
        local_expert_counts = all_expert_counts[:, workspace.rank, :].sum(dim=0).to(
            torch.int32
        )
        aggregate_owner_expert_counts = all_expert_counts.sum(dim=0).to(torch.int32)
    else:
        recv_counts = torch.zeros_like(owner_counts)
        recv_counts[workspace.rank] = owner_counts[workspace.rank]
        owner_base_offsets = torch.zeros_like(owner_counts)
        owner_expert_base_offsets = torch.zeros_like(metadata.owner_expert_counts)
        aggregate_owner_expert_counts = torch.zeros_like(metadata.owner_expert_counts)
        aggregate_owner_expert_counts[workspace.rank] = metadata.owner_expert_counts[
            workspace.rank
        ]
        local_expert_counts = metadata.owner_expert_counts[workspace.rank].contiguous()

    recv_offsets = _offsets_from_counts(recv_counts)
    workspace.prepare_step(
        recv_counts,
        recv_offsets,
        num_rows=_capture_view_rows(workspace),
    )
    return EPOwnerDispatchPlan(
        owner_base_offsets=owner_base_offsets,
        owner_expert_base_offsets=owner_expert_base_offsets,
        aggregate_owner_expert_counts=aggregate_owner_expert_counts,
        aggregate_owner_expert_offsets=_owner_offsets_from_counts(
            aggregate_owner_expert_counts
        ),
        local_expert_counts=local_expert_counts,
        local_expert_offsets=_offsets_from_counts(local_expert_counts),
    )


def _dispatch_with_iris_gluon(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
    dispatch_plan: EPOwnerDispatchPlan,
) -> None:
    if _owner_directed_dispatch_kernel is None or IrisDeviceCtx is None or triton is None:
        raise EPWorkspaceUnavailable("Iris Gluon dispatch support is not importable")
    if workspace.handle.device_context is None:
        raise EPWorkspaceUnavailable("Iris-backed workspace has no device context")

    max_owner_rows = _dispatch_grid_owner_rows(metadata, workspace)
    if max_owner_rows == 0:
        return

    hidden_size = hidden_states.shape[1]
    block_n = min(triton.next_power_of_2(hidden_size), 512)
    grid = (
        workspace.world_size,
        max_owner_rows,
        triton.cdiv(hidden_size, block_n),
    )
    _owner_directed_dispatch_kernel[grid](
        IrisDeviceCtx,
        workspace.handle.device_context,
        workspace.dispatch_buffer,
        hidden_states,
        metadata.combine_offsets,
        metadata.owner_counts,
        metadata.owner_expert_offsets,
        dispatch_plan.owner_expert_base_offsets,
        dispatch_plan.aggregate_owner_expert_offsets,
        workspace.dispatch_buffer.stride(0),
        hidden_states.stride(0),
        metadata.combine_offsets.stride(0),
        metadata.combine_offsets.stride(1),
        metadata.owner_expert_offsets.stride(0),
        metadata.owner_expert_offsets.stride(1),
        dispatch_plan.owner_expert_base_offsets.stride(0),
        dispatch_plan.owner_expert_base_offsets.stride(1),
        dispatch_plan.aggregate_owner_expert_offsets.stride(0),
        dispatch_plan.aggregate_owner_expert_offsets.stride(1),
        hidden_size,
        topk_ids.shape[1],
        max_owner_rows,
        metadata.owner_expert_counts.shape[1],
        block_n,
        max(block_n // 64, 1),
        num_warps=1,
    )


def _workspace_context_ranks(
    workspace: EPCommunicationWorkspace,
    dist_world_size: int,
) -> torch.Tensor:
    context_ranks = (
        torch.arange(workspace.world_size, device=workspace.device, dtype=torch.long)
        * workspace.handle.context_rank_stride
        + workspace.handle.context_rank_start
    )
    if bool(context_ranks.ge(dist_world_size).any().item()):
        raise EPWorkspaceError(
            "EP workspace subgroup ranks exceed distributed world size: "
            f"start={workspace.handle.context_rank_start}, "
            f"stride={workspace.handle.context_rank_stride}, "
            f"world_size={workspace.world_size}, "
            f"distributed_world_size={dist_world_size}"
        )
    return context_ranks


def _dispatch_local_torch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
    dispatch_plan: EPOwnerDispatchPlan,
) -> None:
    top_k = topk_ids.shape[1]
    owner = workspace.rank
    owner_count = int(metadata.owner_counts[owner].item())
    workspace.check_capacity(
        int(dispatch_plan.local_expert_offsets[-1].item()),
        hidden_states.shape[1],
    )
    for owner_row in range(owner_count):
        flat_slot = int(metadata.combine_offsets[owner, owner_row].item())
        if flat_slot < 0:
            continue
        local_id, local_start = _local_expert_for_owner_row(metadata, owner, owner_row)
        dst_row = (
            int(dispatch_plan.aggregate_owner_expert_offsets[owner, local_id].item())
            + int(dispatch_plan.owner_expert_base_offsets[owner, local_id].item())
            + owner_row
            - local_start
        )
        token = flat_slot // top_k
        workspace.dispatch_buffer[dst_row].copy_(hidden_states[token])


def _validate_dispatch_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> None:
    workspace.validate_payload(hidden_states, name="hidden_states")
    if topk_ids.ndim != 2:
        raise EPWorkspaceError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise EPWorkspaceError(
            f"topk_ids rows {topk_ids.shape[0]} != hidden rows {hidden_states.shape[0]}"
        )
    if topk_ids.dtype != torch.int32:
        raise EPWorkspaceError(f"topk_ids must be torch.int32, got {topk_ids.dtype}")
    if topk_ids.device != hidden_states.device:
        raise EPWorkspaceError(
            f"topk_ids device {topk_ids.device} != hidden device {hidden_states.device}"
        )
    _validate_owner_counts(metadata.owner_counts, workspace)
    combine_offsets = metadata.combine_offsets
    if combine_offsets.shape[0] != workspace.world_size:
        raise EPWorkspaceError(
            f"combine_offsets owner dimension {combine_offsets.shape[0]} != "
            f"{workspace.world_size}"
        )
    if combine_offsets.dtype != torch.int32:
        raise EPWorkspaceError(
            f"combine_offsets must be torch.int32, got {combine_offsets.dtype}"
        )
    if combine_offsets.device != hidden_states.device:
        raise EPWorkspaceError(
            f"combine_offsets device {combine_offsets.device} != hidden device "
            f"{hidden_states.device}"
        )


def _validate_owner_counts(
    owner_counts: torch.Tensor,
    workspace: EPCommunicationWorkspace,
) -> None:
    if owner_counts.shape != (workspace.world_size,):
        raise EPWorkspaceError(
            f"owner_counts shape {tuple(owner_counts.shape)} != "
            f"{(workspace.world_size,)}"
        )
    if owner_counts.dtype != torch.int32:
        raise EPWorkspaceError(f"owner_counts must be torch.int32, got {owner_counts.dtype}")
    if owner_counts.device != workspace.device:
        raise EPWorkspaceError(
            f"owner_counts device {owner_counts.device} != workspace device "
            f"{workspace.device}"
        )


def _validate_owner_base_offsets(
    owner_base_offsets: torch.Tensor,
    workspace: EPCommunicationWorkspace,
) -> None:
    if owner_base_offsets.shape != (workspace.world_size,):
        raise EPWorkspaceError(
            f"owner_base_offsets shape {tuple(owner_base_offsets.shape)} != "
            f"{(workspace.world_size,)}"
        )
    if owner_base_offsets.dtype != torch.int32:
        raise EPWorkspaceError(
            f"owner_base_offsets must be torch.int32, got {owner_base_offsets.dtype}"
        )
    if owner_base_offsets.device != workspace.device:
        raise EPWorkspaceError(
            f"owner_base_offsets device {owner_base_offsets.device} != "
            f"{workspace.device}"
        )


def _validate_owner_expert_metadata(
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> None:
    owner_expert_counts = metadata.owner_expert_counts
    if owner_expert_counts.ndim != 2:
        raise EPWorkspaceError(
            "owner_expert_counts must be rank-2, got "
            f"{tuple(owner_expert_counts.shape)}"
        )
    if owner_expert_counts.shape[0] != workspace.world_size:
        raise EPWorkspaceError(
            f"owner_expert_counts owner dimension {owner_expert_counts.shape[0]} != "
            f"{workspace.world_size}"
        )
    if owner_expert_counts.dtype != torch.int32:
        raise EPWorkspaceError(
            f"owner_expert_counts must be torch.int32, got {owner_expert_counts.dtype}"
        )
    if owner_expert_counts.device != workspace.device:
        raise EPWorkspaceError(
            f"owner_expert_counts device {owner_expert_counts.device} != "
            f"workspace device {workspace.device}"
        )
    owner_expert_offsets = metadata.owner_expert_offsets
    expected_offsets_shape = (
        workspace.world_size,
        owner_expert_counts.shape[1] + 1,
    )
    if owner_expert_offsets.shape != expected_offsets_shape:
        raise EPWorkspaceError(
            f"owner_expert_offsets shape {tuple(owner_expert_offsets.shape)} != "
            f"{expected_offsets_shape}"
        )
    if owner_expert_offsets.dtype != torch.int32:
        raise EPWorkspaceError(
            f"owner_expert_offsets must be torch.int32, got "
            f"{owner_expert_offsets.dtype}"
        )
    if owner_expert_offsets.device != workspace.device:
        raise EPWorkspaceError(
            f"owner_expert_offsets device {owner_expert_offsets.device} != "
            f"workspace device {workspace.device}"
        )
    if get_is_capture_mode():
        return
    if bool(owner_expert_offsets[:, 0].ne(0).any().item()):
        raise EPWorkspaceError("owner_expert_offsets must start at zero for each owner")
    if bool(
        (owner_expert_offsets[:, 1:] - owner_expert_offsets[:, :-1])
        .ne(owner_expert_counts)
        .any()
        .item()
    ):
        raise EPWorkspaceError(
            "owner_expert_offsets must match owner_expert_counts prefix sums"
        )
    if bool(owner_expert_offsets[:, -1].ne(metadata.owner_counts).any().item()):
        raise EPWorkspaceError(
            "owner_expert_offsets final column must match owner_counts"
        )


def _validate_dispatch_plan(
    dispatch_plan: EPOwnerDispatchPlan,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> None:
    _validate_owner_base_offsets(dispatch_plan.owner_base_offsets, workspace)
    num_local_experts = metadata.owner_expert_counts.shape[1]
    expected_expert_base_shape = (workspace.world_size, num_local_experts)
    expected_expert_offsets_shape = (workspace.world_size, num_local_experts + 1)
    if dispatch_plan.owner_expert_base_offsets.shape != expected_expert_base_shape:
        raise EPWorkspaceError(
            "owner_expert_base_offsets shape "
            f"{tuple(dispatch_plan.owner_expert_base_offsets.shape)} != "
            f"{expected_expert_base_shape}"
        )
    for name, tensor, expected_shape in (
        (
            "owner_expert_base_offsets",
            dispatch_plan.owner_expert_base_offsets,
            expected_expert_base_shape,
        ),
        (
            "aggregate_owner_expert_counts",
            dispatch_plan.aggregate_owner_expert_counts,
            expected_expert_base_shape,
        ),
        (
            "aggregate_owner_expert_offsets",
            dispatch_plan.aggregate_owner_expert_offsets,
            expected_expert_offsets_shape,
        ),
        ("local_expert_counts", dispatch_plan.local_expert_counts, (num_local_experts,)),
        (
            "local_expert_offsets",
            dispatch_plan.local_expert_offsets,
            (num_local_experts + 1,),
        ),
    ):
        if tensor.shape != expected_shape:
            raise EPWorkspaceError(f"{name} shape {tuple(tensor.shape)} != {expected_shape}")
        if tensor.dtype != torch.int32:
            raise EPWorkspaceError(f"{name} must be torch.int32, got {tensor.dtype}")
        if tensor.device != workspace.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != workspace device {workspace.device}"
            )
    if get_is_capture_mode():
        return
    if bool(
        (dispatch_plan.local_expert_offsets[1:] - dispatch_plan.local_expert_offsets[:-1])
        .ne(dispatch_plan.local_expert_counts)
        .any()
        .item()
    ):
        raise EPWorkspaceError(
            "local_expert_offsets must match local_expert_counts prefix sums"
        )
    if bool(
        (
            dispatch_plan.aggregate_owner_expert_offsets[:, 1:]
            - dispatch_plan.aggregate_owner_expert_offsets[:, :-1]
        )
        .ne(dispatch_plan.aggregate_owner_expert_counts)
        .any()
        .item()
    ):
        raise EPWorkspaceError(
            "aggregate_owner_expert_offsets must match "
            "aggregate_owner_expert_counts prefix sums"
        )


def _source_major_compat_plan(
    owner_base_offsets: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> EPOwnerDispatchPlan:
    _validate_owner_expert_metadata(metadata, workspace)
    local_expert_counts = metadata.owner_expert_counts[workspace.rank].contiguous()
    plan = EPOwnerDispatchPlan(
        owner_base_offsets=owner_base_offsets,
        owner_expert_base_offsets=torch.zeros_like(metadata.owner_expert_counts),
        aggregate_owner_expert_counts=metadata.owner_expert_counts,
        aggregate_owner_expert_offsets=_owner_offsets_from_counts(
            metadata.owner_expert_counts
        ),
        local_expert_counts=local_expert_counts,
        local_expert_offsets=_offsets_from_counts(local_expert_counts),
    )
    _validate_dispatch_plan(plan, metadata, workspace)
    return plan


def _offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        (counts.numel() + 1,),
        dtype=torch.int32,
        device=counts.device,
    )
    offsets[:1].zero_()
    offsets[1:].copy_(torch.cumsum(counts, dim=0, dtype=torch.int32))
    return offsets


def _dispatch_view_rows(
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
) -> int:
    if get_is_capture_mode():
        return workspace.max_dispatch_rows
    return int(dispatch_plan.local_expert_offsets[-1].item())


def _dispatch_grid_owner_rows(
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> int:
    if get_is_capture_mode():
        return workspace.max_dispatch_rows
    return int(metadata.owner_counts.max().item())


def _should_synchronize_workspace(synchronize: bool) -> bool:
    return synchronize and not get_is_capture_mode()


def _capture_view_rows(workspace: EPCommunicationWorkspace) -> int | None:
    if get_is_capture_mode():
        return workspace.max_dispatch_rows
    return None


def _owner_offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        (counts.shape[0], counts.shape[1] + 1),
        dtype=torch.int32,
        device=counts.device,
    )
    offsets[:, :1].zero_()
    offsets[:, 1:].copy_(torch.cumsum(counts, dim=1, dtype=torch.int32))
    return offsets


def _local_expert_for_owner_row(
    metadata: Any,
    owner: int,
    owner_row: int,
) -> tuple[int, int]:
    offsets = metadata.owner_expert_offsets[owner]
    for local_id in range(offsets.numel() - 1):
        start = int(offsets[local_id].item())
        end = int(offsets[local_id + 1].item())
        if start <= owner_row < end:
            return local_id, start
    raise EPWorkspaceError(
        f"owner row {owner_row} is outside owner {owner} expert offsets"
    )


__all__ = [
    "EPOwnerDispatchPlan",
    "owner_directed_dispatch",
    "prepare_owner_directed_dispatch",
]
