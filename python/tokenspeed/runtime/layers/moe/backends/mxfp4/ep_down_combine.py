# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.layers.moe.backends.ep_combine import owner_directed_combine
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import EPOwnerDispatchPlan
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    PreRoutedFusedEPMetadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_reduce import ep_weighted_reduce
from tokenspeed.runtime.layers.moe.backends.ep_workspace import EPCommunicationWorkspace
from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
    owner_rank_mxfp4_down_gemm,
)


@dataclass(frozen=True)
class Mxfp4DownCombineResult:
    output: torch.Tensor
    owner_outputs: torch.Tensor
    returned_slots: torch.Tensor
    dispatch_plan: EPOwnerDispatchPlan
    fused_metadata: PreRoutedFusedEPMetadata


def owner_rank_mxfp4_down_from_gate_up(
    gate_up: torch.Tensor,
    local_down_weight: torch.Tensor,
    local_down_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run owner-rank MXFP4 down GEMM over sorted gate/up rows."""

    return owner_rank_mxfp4_down_gemm(
        gate_up,
        local_down_weight,
        local_down_weight_scale,
        local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        bias=bias,
        output_dtype=output_dtype or gate_up.dtype,
        out=out,
    )


def mxfp4_ep_down_gemm_combine(
    gate_up: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    fused_metadata: PreRoutedFusedEPMetadata,
    dispatch_plan: EPOwnerDispatchPlan,
    local_down_weight: torch.Tensor,
    local_down_weight_scale: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    out: torch.Tensor | None = None,
    synchronize: bool = True,
    expected_reduce_kernel_name: str | None = None,
) -> Mxfp4DownCombineResult:
    """Run MXFP4 owner down GEMM, return slots, then apply top-k weights."""

    owner_outputs = owner_rank_mxfp4_down_from_gate_up(
        gate_up,
        local_down_weight,
        local_down_weight_scale,
        dispatch_plan.local_expert_counts,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
        bias=bias,
        output_dtype=workspace.dtype,
    )
    combine_rows = owner_outputs.shape[0]
    active_combine_buffer = workspace.combine_buffer[:combine_rows]
    if active_combine_buffer.data_ptr() != owner_outputs.data_ptr():
        active_combine_buffer.copy_(owner_outputs)
    returned_slots = owner_directed_combine(
        workspace.combine_buffer,
        topk_ids,
        ep_metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
        synchronize=synchronize,
    )
    output = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        out=out,
        expected_kernel_name=expected_reduce_kernel_name,
    )
    return Mxfp4DownCombineResult(
        output=output,
        owner_outputs=active_combine_buffer,
        returned_slots=returned_slots,
        dispatch_plan=dispatch_plan,
        fused_metadata=fused_metadata,
    )


__all__ = [
    "Mxfp4DownCombineResult",
    "mxfp4_ep_down_gemm_combine",
    "owner_rank_mxfp4_down_from_gate_up",
]
