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

"""Internal pre-routed fused EP metadata.

This module only rearranges routing that already exists in TopK and S4 EP
metadata. It does not run routing, move token rows, or define a public fused
feature beyond the existing ``moe.fused`` ``pre_routed`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    EPOwnerDispatchPlan,
    prepare_owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)


@dataclass(frozen=True)
class PreRoutedFusedEPMetadata:
    source_rank: int
    world_size: int
    num_tokens: int
    hidden_size: int
    top_k: int
    source_flat_slots: torch.Tensor
    source_ranks: torch.Tensor
    source_token_ids: torch.Tensor
    source_topk_slots: torch.Tensor
    valid_slot_mask: torch.Tensor
    owner_ranks: torch.Tensor
    local_expert_ids: torch.Tensor
    source_owner_rows: torch.Tensor
    owner_rows: torch.Tensor
    topk_weights: torch.Tensor
    owner_row_to_source_flat_slot: torch.Tensor
    owner_row_to_source: torch.Tensor
    owner_counts: torch.Tensor
    owner_offsets: torch.Tensor
    owner_expert_counts: torch.Tensor
    owner_expert_offsets: torch.Tensor
    owner_expert_base_offsets: torch.Tensor
    aggregate_owner_expert_counts: torch.Tensor
    aggregate_owner_expert_offsets: torch.Tensor
    local_expert_counts: torch.Tensor
    local_expert_offsets: torch.Tensor

    @property
    def num_slots(self) -> int:
        return self.source_flat_slots.numel()

    @property
    def num_valid_slots(self) -> int:
        return int(self.valid_slot_mask.sum().item())


def build_pre_routed_fused_ep_metadata(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    *,
    dispatch_plan: EPOwnerDispatchPlan | None = None,
) -> PreRoutedFusedEPMetadata:
    _validate_inputs(hidden_states, topk_ids, topk_weights, ep_metadata, workspace)
    if dispatch_plan is None:
        dispatch_plan = _source_local_dispatch_plan(ep_metadata, workspace)
    else:
        _validate_dispatch_plan(dispatch_plan, ep_metadata, workspace)

    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    device = topk_ids.device
    flat_slots = torch.arange(num_slots, dtype=torch.int32, device=device)
    source_ranks = torch.full_like(flat_slots, workspace.rank)
    source_token_ids = (
        flat_slots // top_k if top_k > 0 else torch.empty_like(flat_slots)
    )
    source_topk_slots = (
        flat_slots - source_token_ids * top_k
        if top_k > 0
        else torch.empty_like(flat_slots)
    )
    source_owner_rows = ep_metadata.dispatch_offsets.reshape(-1).to(torch.int32)
    owner_ranks, local_expert_ids = _derive_slot_owner_metadata(
        ep_metadata,
        num_slots=num_slots,
        workspace=workspace,
    )
    valid_slot_mask = (
        (source_owner_rows >= 0)
        & (owner_ranks >= 0)
        & (local_expert_ids >= 0)
    )
    owner_rows = _aggregate_owner_rows(
        source_owner_rows,
        owner_ranks,
        local_expert_ids,
        valid_slot_mask,
        ep_metadata,
        dispatch_plan,
    )
    (
        owner_row_to_source_flat_slot,
        owner_row_to_source,
    ) = _build_owner_row_identity(
        flat_slots,
        source_ranks,
        source_token_ids,
        source_topk_slots,
        owner_ranks,
        owner_rows,
        valid_slot_mask,
        dispatch_plan,
        workspace,
    )

    return PreRoutedFusedEPMetadata(
        source_rank=workspace.rank,
        world_size=workspace.world_size,
        num_tokens=num_tokens,
        hidden_size=hidden_states.shape[1],
        top_k=top_k,
        source_flat_slots=flat_slots,
        source_ranks=source_ranks,
        source_token_ids=source_token_ids,
        source_topk_slots=source_topk_slots,
        valid_slot_mask=valid_slot_mask,
        owner_ranks=owner_ranks,
        local_expert_ids=local_expert_ids,
        source_owner_rows=source_owner_rows,
        owner_rows=owner_rows,
        topk_weights=topk_weights.reshape(-1),
        owner_row_to_source_flat_slot=owner_row_to_source_flat_slot,
        owner_row_to_source=owner_row_to_source,
        owner_counts=ep_metadata.owner_counts,
        owner_offsets=ep_metadata.owner_offsets,
        owner_expert_counts=ep_metadata.owner_expert_counts,
        owner_expert_offsets=ep_metadata.owner_expert_offsets,
        owner_expert_base_offsets=dispatch_plan.owner_expert_base_offsets,
        aggregate_owner_expert_counts=dispatch_plan.aggregate_owner_expert_counts,
        aggregate_owner_expert_offsets=dispatch_plan.aggregate_owner_expert_offsets,
        local_expert_counts=dispatch_plan.local_expert_counts,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
    )


def prepare_pre_routed_fused_ep_metadata(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> PreRoutedFusedEPMetadata:
    dispatch_plan = None
    if workspace.world_size == 1 or _has_distributed_process_group():
        dispatch_plan = prepare_owner_directed_dispatch(ep_metadata, workspace)
    return build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        dispatch_plan=dispatch_plan,
    )


def _has_distributed_process_group() -> bool:
    return dist.is_available() and dist.is_initialized()


def _validate_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> None:
    workspace.validate_payload(hidden_states, name="hidden_states")
    if topk_ids.ndim != 2:
        raise EPWorkspaceError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise EPWorkspaceError(
            f"topk_weights shape {tuple(topk_weights.shape)} != topk_ids shape "
            f"{tuple(topk_ids.shape)}"
        )
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise EPWorkspaceError(
            f"topk_ids rows {topk_ids.shape[0]} != hidden rows {hidden_states.shape[0]}"
        )
    if topk_ids.dtype != torch.int32:
        raise EPWorkspaceError(f"topk_ids must be torch.int32, got {topk_ids.dtype}")
    if not topk_weights.is_floating_point():
        raise EPWorkspaceError(
            f"topk_weights must be floating point, got {topk_weights.dtype}"
        )
    if topk_ids.shape[1] != workspace.top_k:
        raise EPWorkspaceError(
            f"top_k mismatch: topk_ids has {topk_ids.shape[1]}, "
            f"workspace expects {workspace.top_k}"
        )
    for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
        if tensor.device != workspace.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != workspace device {workspace.device}"
            )
    expected_slots = topk_ids.numel()
    _validate_metadata_tensor(
        ep_metadata.owner_counts,
        "owner_counts",
        (workspace.world_size,),
        workspace,
    )
    _validate_metadata_tensor(
        ep_metadata.owner_offsets,
        "owner_offsets",
        (workspace.world_size + 1,),
        workspace,
    )
    if ep_metadata.owner_expert_counts.ndim != 2:
        raise EPWorkspaceError(
            "owner_expert_counts must be rank-2, got "
            f"{tuple(ep_metadata.owner_expert_counts.shape)}"
        )
    if ep_metadata.owner_expert_counts.shape[0] != workspace.world_size:
        raise EPWorkspaceError(
            "owner_expert_counts owner dimension "
            f"{ep_metadata.owner_expert_counts.shape[0]} != {workspace.world_size}"
        )
    _validate_metadata_tensor(
        ep_metadata.owner_expert_counts,
        "owner_expert_counts",
        tuple(ep_metadata.owner_expert_counts.shape),
        workspace,
    )
    _validate_metadata_tensor(
        ep_metadata.owner_expert_offsets,
        "owner_expert_offsets",
        (
            workspace.world_size,
            ep_metadata.owner_expert_counts.shape[1] + 1,
        ),
        workspace,
    )
    _validate_metadata_tensor(
        ep_metadata.dispatch_offsets,
        "dispatch_offsets",
        topk_ids.shape,
        workspace,
    )
    _validate_metadata_tensor(
        ep_metadata.combine_offsets,
        "combine_offsets",
        (workspace.world_size, expected_slots),
        workspace,
    )
    _validate_s4_metadata_consistency(ep_metadata, expected_slots=expected_slots)


def _validate_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    workspace: EPCommunicationWorkspace,
) -> None:
    if tensor.shape != shape:
        raise EPWorkspaceError(f"{name} shape {tuple(tensor.shape)} != {shape}")
    if tensor.dtype != torch.int32:
        raise EPWorkspaceError(f"{name} must be torch.int32, got {tensor.dtype}")
    if tensor.device != workspace.device:
        raise EPWorkspaceError(
            f"{name} device {tensor.device} != workspace device {workspace.device}"
        )


def _validate_s4_metadata_consistency(ep_metadata: Any, *, expected_slots: int) -> None:
    if bool(ep_metadata.owner_offsets[0].ne(0).item()):
        raise EPWorkspaceError("owner_offsets must start at zero")
    if bool(
        (ep_metadata.owner_offsets[1:] - ep_metadata.owner_offsets[:-1])
        .ne(ep_metadata.owner_counts)
        .any()
        .item()
    ):
        raise EPWorkspaceError("owner_offsets must match owner_counts prefix sums")
    if bool(ep_metadata.owner_expert_offsets[:, 0].ne(0).any().item()):
        raise EPWorkspaceError("owner_expert_offsets must start at zero")
    if bool(
        (
            ep_metadata.owner_expert_offsets[:, 1:]
            - ep_metadata.owner_expert_offsets[:, :-1]
        )
        .ne(ep_metadata.owner_expert_counts)
        .any()
        .item()
    ):
        raise EPWorkspaceError(
            "owner_expert_offsets must match owner_expert_counts prefix sums"
        )
    if bool(
        ep_metadata.owner_expert_offsets[:, -1].ne(ep_metadata.owner_counts).any().item()
    ):
        raise EPWorkspaceError(
            "owner_expert_offsets final column must match owner_counts"
        )
    valid_combine = ep_metadata.combine_offsets >= 0
    valid_dispatch = ep_metadata.dispatch_offsets >= 0
    if bool(
        (
            ep_metadata.combine_offsets[valid_combine] >= expected_slots
        ).any().item()
    ):
        raise EPWorkspaceError("combine_offsets contain source slots out of range")
    if valid_combine.sum().item() != valid_dispatch.sum().item():
        raise EPWorkspaceError(
            "combine_offsets and dispatch_offsets must contain the same number "
            "of valid routed slots"
        )
    if bool(valid_combine.any().item()):
        slots = ep_metadata.combine_offsets[valid_combine]
        if torch.unique(slots).numel() != slots.numel():
            raise EPWorkspaceError("combine_offsets must not duplicate source slots")


def _source_local_dispatch_plan(
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> EPOwnerDispatchPlan:
    owner_expert_counts = ep_metadata.owner_expert_counts
    aggregate_owner_expert_offsets = _owner_offsets_from_counts(owner_expert_counts)
    local_expert_counts = owner_expert_counts[workspace.rank].contiguous()
    return EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros_like(ep_metadata.owner_counts),
        owner_expert_base_offsets=torch.zeros_like(owner_expert_counts),
        aggregate_owner_expert_counts=owner_expert_counts,
        aggregate_owner_expert_offsets=aggregate_owner_expert_offsets,
        local_expert_counts=local_expert_counts,
        local_expert_offsets=aggregate_owner_expert_offsets[workspace.rank].contiguous(),
    )


def _validate_dispatch_plan(
    dispatch_plan: EPOwnerDispatchPlan,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
) -> None:
    num_local_experts = ep_metadata.owner_expert_counts.shape[1]
    for name, tensor, shape in (
        ("owner_base_offsets", dispatch_plan.owner_base_offsets, (workspace.world_size,)),
        (
            "owner_expert_base_offsets",
            dispatch_plan.owner_expert_base_offsets,
            (workspace.world_size, num_local_experts),
        ),
        (
            "aggregate_owner_expert_counts",
            dispatch_plan.aggregate_owner_expert_counts,
            (workspace.world_size, num_local_experts),
        ),
        (
            "aggregate_owner_expert_offsets",
            dispatch_plan.aggregate_owner_expert_offsets,
            (workspace.world_size, num_local_experts + 1),
        ),
        (
            "local_expert_counts",
            dispatch_plan.local_expert_counts,
            (num_local_experts,),
        ),
        (
            "local_expert_offsets",
            dispatch_plan.local_expert_offsets,
            (num_local_experts + 1,),
        ),
    ):
        _validate_metadata_tensor(tensor, name, shape, workspace)
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
    if bool(
        (
            dispatch_plan.local_expert_offsets[1:]
            - dispatch_plan.local_expert_offsets[:-1]
        )
        .ne(dispatch_plan.local_expert_counts)
        .any()
        .item()
    ):
        raise EPWorkspaceError(
            "local_expert_offsets must match local_expert_counts prefix sums"
        )


def _owner_offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        (counts.shape[0], counts.shape[1] + 1),
        dtype=torch.int32,
        device=counts.device,
    )
    offsets[:, 0] = 0
    offsets[:, 1:] = torch.cumsum(counts, dim=1, dtype=torch.int32)
    return offsets


def _derive_slot_owner_metadata(
    ep_metadata: Any,
    *,
    num_slots: int,
    workspace: EPCommunicationWorkspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner_ranks = torch.full(
        (num_slots,),
        -1,
        dtype=torch.int32,
        device=workspace.device,
    )
    local_expert_ids = torch.full_like(owner_ranks, -1)
    for owner in range(workspace.world_size):
        owner_count = int(ep_metadata.owner_counts[owner].item())
        if owner_count == 0:
            continue
        owner_slots = ep_metadata.combine_offsets[owner, :owner_count]
        valid_owner_slots = owner_slots >= 0
        if bool(valid_owner_slots.any().item()):
            slot_indices = owner_slots[valid_owner_slots].to(torch.long)
            owner_ranks[slot_indices] = owner

        offsets = ep_metadata.owner_expert_offsets[owner]
        for local_id in range(offsets.numel() - 1):
            start = int(offsets[local_id].item())
            end = int(offsets[local_id + 1].item())
            if start == end:
                continue
            expert_slots = ep_metadata.combine_offsets[owner, start:end]
            valid_expert_slots = expert_slots >= 0
            if bool(valid_expert_slots.any().item()):
                local_expert_ids[expert_slots[valid_expert_slots].to(torch.long)] = (
                    local_id
                )
    return owner_ranks, local_expert_ids


def _aggregate_owner_rows(
    source_owner_rows: torch.Tensor,
    owner_ranks: torch.Tensor,
    local_expert_ids: torch.Tensor,
    valid_slot_mask: torch.Tensor,
    ep_metadata: Any,
    dispatch_plan: EPOwnerDispatchPlan,
) -> torch.Tensor:
    owner_rows = torch.full_like(source_owner_rows, -1)
    if not bool(valid_slot_mask.any().item()):
        return owner_rows

    owner_idx = owner_ranks[valid_slot_mask].to(torch.long)
    local_idx = local_expert_ids[valid_slot_mask].to(torch.long)
    source_rows = source_owner_rows[valid_slot_mask]
    local_starts = ep_metadata.owner_expert_offsets[owner_idx, local_idx]
    aggregate_starts = dispatch_plan.aggregate_owner_expert_offsets[
        owner_idx,
        local_idx,
    ]
    source_bases = dispatch_plan.owner_expert_base_offsets[owner_idx, local_idx]
    owner_rows[valid_slot_mask] = (
        aggregate_starts + source_bases + source_rows - local_starts
    ).to(torch.int32)
    return owner_rows


def _build_owner_row_identity(
    flat_slots: torch.Tensor,
    source_ranks: torch.Tensor,
    source_token_ids: torch.Tensor,
    source_topk_slots: torch.Tensor,
    owner_ranks: torch.Tensor,
    owner_rows: torch.Tensor,
    valid_slot_mask: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
    workspace: EPCommunicationWorkspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_owner_rows = int(dispatch_plan.aggregate_owner_expert_offsets[:, -1].max().item())
    owner_row_to_source_flat_slot = torch.full(
        (workspace.world_size, max_owner_rows),
        -1,
        dtype=torch.int32,
        device=workspace.device,
    )
    owner_row_to_source = torch.full(
        (workspace.world_size, max_owner_rows, 3),
        -1,
        dtype=torch.int32,
        device=workspace.device,
    )
    if max_owner_rows == 0 or not bool(valid_slot_mask.any().item()):
        if bool(valid_slot_mask.any().item()):
            raise EPWorkspaceError(
                "aggregate owner row capacity is zero for non-empty fused metadata"
            )
        return owner_row_to_source_flat_slot, owner_row_to_source

    owner_idx = owner_ranks[valid_slot_mask].to(torch.long)
    owner_row_idx = owner_rows[valid_slot_mask].to(torch.long)
    valid_rows = owner_row_idx >= 0
    if not bool(valid_rows.any().item()):
        return owner_row_to_source_flat_slot, owner_row_to_source
    if bool((owner_row_idx[valid_rows] >= max_owner_rows).any().item()):
        raise EPWorkspaceError(
            "owner_rows contain entries outside aggregate owner row capacity"
        )

    owner_idx = owner_idx[valid_rows]
    owner_row_idx = owner_row_idx[valid_rows]
    linear_owner_rows = owner_idx * max_owner_rows + owner_row_idx
    if torch.unique(linear_owner_rows).numel() != linear_owner_rows.numel():
        raise EPWorkspaceError("owner_rows must not duplicate aggregate owner rows")
    slot_idx = flat_slots[valid_slot_mask][valid_rows]
    owner_row_to_source_flat_slot[owner_idx, owner_row_idx] = slot_idx
    owner_row_to_source[owner_idx, owner_row_idx, 0] = source_ranks[
        valid_slot_mask
    ][valid_rows]
    owner_row_to_source[owner_idx, owner_row_idx, 1] = source_token_ids[
        valid_slot_mask
    ][valid_rows]
    owner_row_to_source[owner_idx, owner_row_idx, 2] = source_topk_slots[
        valid_slot_mask
    ][valid_rows]
    return owner_row_to_source_flat_slot, owner_row_to_source


__all__ = [
    "PreRoutedFusedEPMetadata",
    "build_pre_routed_fused_ep_metadata",
    "prepare_pre_routed_fused_ep_metadata",
]
