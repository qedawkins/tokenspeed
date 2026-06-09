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

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import EPOwnerDispatchPlan
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    PreRoutedFusedEPMetadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceStep,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    quantize_mxfp4_activation,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_dispatch import (
    dispatch_mxfp4_hidden_states_to_owner_buffers,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
    owner_rank_mxfp4_gate_up_gemm,
)


@dataclass(frozen=True)
class Mxfp4OwnerGateUpResult:
    gate_up: torch.Tensor
    owner_tokens: torch.Tensor
    dispatch_step: EPWorkspaceStep
    dispatch_plan: EPOwnerDispatchPlan
    fused_metadata: PreRoutedFusedEPMetadata


def owner_rank_mxfp4_gate_up_from_owner_tokens(
    owner_tokens: torch.Tensor,
    local_gate_up_weight: torch.Tensor,
    local_gate_up_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float | None = None,
    swiglu_beta: float | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run MXFP4 gate/up over owner-local hidden rows sorted by local expert."""

    packed_owner_tokens, owner_token_scale = quantize_mxfp4_activation(owner_tokens)
    return owner_rank_mxfp4_gate_up_gemm(
        packed_owner_tokens,
        owner_token_scale,
        local_gate_up_weight,
        local_gate_up_weight_scale,
        local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        bias=bias,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        output_dtype=output_dtype,
        out=out,
    )


def dispatch_mxfp4_hidden_states_gate_up(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    local_gate_up_weight: torch.Tensor,
    local_gate_up_weight_scale: torch.Tensor,
    *,
    fused_metadata: PreRoutedFusedEPMetadata | None = None,
    dispatch_plan: EPOwnerDispatchPlan | None = None,
    bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float | None = None,
    swiglu_beta: float | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    synchronize: bool = True,
) -> Mxfp4OwnerGateUpResult:
    """Dispatch hidden rows to owners, then run owner-rank MXFP4 gate/up."""

    dispatch_result = dispatch_mxfp4_hidden_states_to_owner_buffers(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        fused_metadata=fused_metadata,
        dispatch_plan=dispatch_plan,
        synchronize=synchronize,
    )
    gate_up = owner_rank_mxfp4_gate_up_from_owner_tokens(
        dispatch_result.owner_tokens,
        local_gate_up_weight,
        local_gate_up_weight_scale,
        dispatch_result.dispatch_plan.local_expert_counts,
        local_expert_offsets=dispatch_result.dispatch_plan.local_expert_offsets,
        bias=bias,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
        output_dtype=output_dtype,
        out=out,
    )
    return Mxfp4OwnerGateUpResult(
        gate_up=gate_up,
        owner_tokens=dispatch_result.owner_tokens,
        dispatch_step=dispatch_result.dispatch_step,
        dispatch_plan=dispatch_result.dispatch_plan,
        fused_metadata=dispatch_result.fused_metadata,
    )


__all__ = [
    "Mxfp4OwnerGateUpResult",
    "dispatch_mxfp4_hidden_states_gate_up",
    "owner_rank_mxfp4_gate_up_from_owner_tokens",
]
