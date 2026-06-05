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

"""Internal self-routing top-k preparation for the existing fused MoE path."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
    select_experts,
)


def self_routing_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    hidden_states: torch.Tensor | None = None,
    routing_bias: torch.Tensor | None = None,
    n_group: int | None = None,
    topk_group: int | None = None,
    renormalize: bool = True,
    routed_scaling_factor: float | None = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    num_token_non_padded: torch.Tensor | None = None,
    expert_location_dispatch_info=None,
) -> StandardTopKOutput:
    """Compute the top-k payload a self-routing fused MoE kernel will consume.

    This is deliberately an internal preparation helper for
    ``moe.fused``/``self_routing``. It reuses the existing ``TopK``/``moe.route``
    semantics instead of inventing a second routing implementation.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must have shape [M_r, E], got {tuple(router_logits.shape)}"
        )
    routing_hidden_states = _routing_hidden_states(router_logits, hidden_states)
    if routing_hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError(
            "hidden_states and router_logits must have the same token dimension, "
            f"got {routing_hidden_states.shape[0]} and {router_logits.shape[0]}"
        )
    if (n_group is None) != (topk_group is None):
        raise ValueError("n_group and topk_group must be provided together")
    use_grouped_topk = n_group is not None

    topk_config = TopKConfig(
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        topk_group=topk_group,
        num_expert_group=n_group,
        renormalize=renormalize,
        correction_bias=routing_bias,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            apply_routed_scaling_factor_on_output
        ),
        topk_indices_dtype=torch.int32,
    )
    return select_experts(
        hidden_states=routing_hidden_states,
        router_logits=router_logits,
        topk_config=topk_config,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )


def self_routing_topk_from_bypassed(
    topk_output: BypassedTopKOutput,
) -> StandardTopKOutput:
    """Resolve the existing ``moe.fused`` self-routing TopK bypass contract."""

    topk_config = topk_output.topk_config
    n_group = topk_config.num_expert_group if topk_config.use_grouped_topk else None
    topk_group = topk_config.topk_group if topk_config.use_grouped_topk else None
    return self_routing_topk(
        topk_output.router_logits,
        top_k=topk_config.top_k,
        hidden_states=topk_output.hidden_states,
        routing_bias=topk_config.correction_bias,
        n_group=n_group,
        topk_group=topk_group,
        renormalize=topk_config.renormalize,
        routed_scaling_factor=topk_config.routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            topk_config.apply_routed_scaling_factor_on_output
        ),
        num_token_non_padded=topk_output.num_token_non_padded,
        expert_location_dispatch_info=topk_output.expert_location_dispatch_info,
    )


def _routing_hidden_states(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor | None,
) -> torch.Tensor:
    if hidden_states is not None:
        return hidden_states
    return router_logits.new_empty((router_logits.shape[0], 0))


__all__ = ["self_routing_topk", "self_routing_topk_from_bypassed"]
