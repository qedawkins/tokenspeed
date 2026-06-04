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

from typing import Any

import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
    EPWorkspaceStep,
    EPWorkspaceUnavailable,
)

try:
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
        owner_base_offsets_ptr,
        DST_STRIDE_M: gl.constexpr,
        SRC_STRIDE_M: gl.constexpr,
        COMBINE_STRIDE_OWNER: gl.constexpr,
        COMBINE_STRIDE_ROW: gl.constexpr,
        HIDDEN_SIZE: gl.constexpr,
        TOPK: gl.constexpr,
        MAX_OWNER_ROWS: gl.constexpr,
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
                token = flat_slot // TOPK
                owner_base = gl.load(owner_base_offsets_ptr + owner).to(gl.int32)
                dst_row = owner_base + owner_row
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
    owner_base_offsets: torch.Tensor | None = None,
    synchronize: bool = True,
) -> EPWorkspaceStep:
    _validate_dispatch_inputs(hidden_states, topk_ids, metadata, workspace)
    if owner_base_offsets is None:
        owner_base_offsets = prepare_owner_directed_dispatch(metadata, workspace)
    else:
        _validate_owner_base_offsets(owner_base_offsets, workspace)

    if workspace.backend == "iris":
        _dispatch_with_iris_gluon(
            hidden_states,
            topk_ids,
            metadata,
            workspace,
            owner_base_offsets,
        )
        if synchronize:
            workspace.barrier()
    else:
        _dispatch_local_torch(
            hidden_states,
            topk_ids,
            metadata,
            workspace,
            owner_base_offsets,
        )

    return workspace.rank_local_view()


def prepare_owner_directed_dispatch(
    metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> torch.Tensor:
    owner_counts = metadata.owner_counts
    _validate_owner_counts(owner_counts, workspace)

    if workspace.world_size > 1 and dist.is_available() and dist.is_initialized():
        gathered = [torch.empty_like(owner_counts) for _ in range(workspace.world_size)]
        dist.all_gather(gathered, owner_counts.contiguous())
        all_counts = torch.stack(gathered, dim=0)
        recv_counts = all_counts[:, workspace.rank].contiguous()
        owner_base_offsets = all_counts[: workspace.rank, :].sum(dim=0).to(torch.int32)
    else:
        recv_counts = torch.zeros_like(owner_counts)
        recv_counts[workspace.rank] = owner_counts[workspace.rank]
        owner_base_offsets = torch.zeros_like(owner_counts)

    recv_offsets = torch.empty(
        (workspace.world_size + 1,),
        dtype=torch.int32,
        device=owner_counts.device,
    )
    recv_offsets[0] = 0
    recv_offsets[1:] = torch.cumsum(recv_counts, dim=0, dtype=torch.int32)
    workspace.prepare_step(recv_counts, recv_offsets)
    return owner_base_offsets


def _dispatch_with_iris_gluon(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
    owner_base_offsets: torch.Tensor,
) -> None:
    if _owner_directed_dispatch_kernel is None or IrisDeviceCtx is None or triton is None:
        raise EPWorkspaceUnavailable("Iris Gluon dispatch support is not importable")
    if workspace.handle.device_context is None:
        raise EPWorkspaceUnavailable("Iris-backed workspace has no device context")

    max_owner_rows = int(metadata.owner_counts.max().item())
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
        owner_base_offsets,
        workspace.dispatch_buffer.stride(0),
        hidden_states.stride(0),
        metadata.combine_offsets.stride(0),
        metadata.combine_offsets.stride(1),
        hidden_size,
        topk_ids.shape[1],
        max_owner_rows,
        block_n,
        max(block_n // 64, 1),
        num_warps=1,
    )


def _dispatch_local_torch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: Any,
    workspace: EPCommunicationWorkspace,
    owner_base_offsets: torch.Tensor,
) -> None:
    top_k = topk_ids.shape[1]
    owner = workspace.rank
    owner_count = int(metadata.owner_counts[owner].item())
    owner_base = int(owner_base_offsets[owner].item())
    workspace.check_capacity(owner_base + owner_count, hidden_states.shape[1])
    for owner_row in range(owner_count):
        flat_slot = int(metadata.combine_offsets[owner, owner_row].item())
        if flat_slot < 0:
            continue
        token = flat_slot // top_k
        workspace.dispatch_buffer[owner_base + owner_row].copy_(hidden_states[token])


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


__all__ = ["owner_directed_dispatch", "prepare_owner_directed_dispatch"]
