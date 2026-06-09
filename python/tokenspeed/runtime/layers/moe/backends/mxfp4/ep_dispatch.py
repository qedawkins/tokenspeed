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

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    EPOwnerDispatchPlan,
    owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    PreRoutedFusedEPMetadata,
    build_pre_routed_fused_ep_metadata,
    prepare_pre_routed_fused_ep_metadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceStep,
)


@dataclass(frozen=True)
class Mxfp4OwnerDispatchResult:
    owner_tokens: torch.Tensor
    dispatch_step: EPWorkspaceStep
    dispatch_plan: EPOwnerDispatchPlan
    fused_metadata: PreRoutedFusedEPMetadata


def dispatch_mxfp4_hidden_states_to_owner_buffers(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    *,
    fused_metadata: PreRoutedFusedEPMetadata | None = None,
    dispatch_plan: EPOwnerDispatchPlan | None = None,
    synchronize: bool = True,
) -> Mxfp4OwnerDispatchResult:
    """Dispatch MXFP4 EP hidden rows to owner-rank buffers.

    The payload stays in the incoming BF16/FP16 hidden-state dtype. This helper
    only ties Kimi/MXFP4 call sites to the existing owner-directed dispatch
    contract; Iris-specific handles remain encapsulated in ``workspace``.
    """

    if fused_metadata is None:
        if dispatch_plan is None:
            fused_metadata = prepare_pre_routed_fused_ep_metadata(
                hidden_states,
                topk_ids,
                topk_weights,
                ep_metadata,
                workspace,
            )
            dispatch_plan = _dispatch_plan_from_fused_metadata(fused_metadata)
        else:
            fused_metadata = build_pre_routed_fused_ep_metadata(
                hidden_states,
                topk_ids,
                topk_weights,
                ep_metadata,
                workspace,
                dispatch_plan=dispatch_plan,
            )
    else:
        if dispatch_plan is not None:
            raise ValueError("provide fused_metadata or dispatch_plan, not both")
        dispatch_plan = _dispatch_plan_from_fused_metadata(fused_metadata)

    dispatch_step = owner_directed_dispatch(
        hidden_states,
        topk_ids,
        ep_metadata,
        workspace,
        dispatch_plan=dispatch_plan,
        synchronize=synchronize,
    )
    return Mxfp4OwnerDispatchResult(
        owner_tokens=dispatch_step.dispatch_buffer,
        dispatch_step=dispatch_step,
        dispatch_plan=dispatch_plan,
        fused_metadata=fused_metadata,
    )


def _dispatch_plan_from_fused_metadata(
    fused_metadata: PreRoutedFusedEPMetadata,
) -> EPOwnerDispatchPlan:
    return EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros_like(fused_metadata.owner_counts),
        owner_expert_base_offsets=fused_metadata.owner_expert_base_offsets,
        aggregate_owner_expert_counts=fused_metadata.aggregate_owner_expert_counts,
        aggregate_owner_expert_offsets=fused_metadata.aggregate_owner_expert_offsets,
        local_expert_counts=fused_metadata.local_expert_counts,
        local_expert_offsets=fused_metadata.local_expert_offsets,
    )


__all__ = [
    "Mxfp4OwnerDispatchResult",
    "dispatch_mxfp4_hidden_states_to_owner_buffers",
]
