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

from collections.abc import Callable
from typing import Any

import torch


def is_kimi_sigmoid_noaux_topk_config(topk_config: object) -> bool:
    """Return whether a bypassed top-k config needs Kimi noaux routing."""

    return (
        bool(getattr(topk_config, "use_grouped_topk", False))
        and getattr(topk_config, "num_expert_group", None) == 1
        and getattr(topk_config, "topk_group", None) == 1
        and getattr(topk_config, "correction_bias", None) is not None
        and getattr(topk_config, "num_fused_shared_experts", 0) == 0
        and getattr(topk_config, "custom_routing_function", None) is None
    )


def select_kimi_sigmoid_noaux_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    correction_bias: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float | None,
    apply_routed_scaling_factor_on_output: bool,
    topk_indices_dtype: torch.dtype | None = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi-K2.5 ``noaux_tc`` routing for ``n_group == topk_group == 1``.

    Selection uses ``sigmoid(router_logits) + correction_bias``. Returned
    weights are gathered from the un-biased sigmoid scores, then normalized and
    optionally scaled to match the existing grouped-biased top-k kernel
    contract.
    """

    if router_logits.dim() != 2:
        raise ValueError(
            "router_logits must have shape [num_tokens, num_experts], "
            f"got {tuple(router_logits.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    num_tokens, num_experts = router_logits.shape
    if top_k > num_experts:
        raise ValueError(f"top_k {top_k} exceeds num_experts {num_experts}")
    if correction_bias.dim() != 1 or correction_bias.shape[0] != num_experts:
        raise ValueError(
            "correction_bias must have shape [num_experts], got "
            f"{tuple(correction_bias.shape)} for {num_experts} experts"
        )

    id_dtype = topk_indices_dtype or torch.int32
    if num_tokens == 0:
        topk_weights = torch.empty(
            (0, top_k),
            device=router_logits.device,
            dtype=torch.float32,
        )
        topk_ids = torch.empty(
            (0, top_k),
            device=router_logits.device,
            dtype=id_dtype,
        )
        return topk_weights, topk_ids

    scores = router_logits.float().sigmoid()
    bias = correction_bias.to(device=scores.device, dtype=scores.dtype)
    choice_scores = scores + bias.unsqueeze(0)
    topk_ids = _stable_descending_topk_ids(choice_scores, top_k)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        denom = topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_weights.dtype).tiny
        )
        topk_weights = topk_weights / denom
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
            topk_weights = topk_weights * float(routed_scaling_factor)

    return topk_weights.to(torch.float32), topk_ids.to(id_dtype)


def mxfp4_kimi_sigmoid_ragged_route_from_bypassed(
    topk_output: object,
    *,
    num_experts: int,
    metadata_factory: Callable[[torch.Tensor, int], Any],
    gate_dtype: torch.dtype | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build MXFP4 ragged metadata from a bypassed Kimi sigmoid TopK output."""

    topk_config = topk_output.topk_config
    if not is_kimi_sigmoid_noaux_topk_config(topk_config):
        raise ValueError("topk_config is not a Kimi sigmoid/noaux top-k config")

    router_logits = topk_output.router_logits
    hidden_states = getattr(topk_output, "hidden_states", None)
    if hidden_states is not None and hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError(
            "hidden_states and router_logits must have the same token dimension, "
            f"got {hidden_states.shape[0]} and {router_logits.shape[0]}"
        )

    topk_weights, topk_ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=topk_config.top_k,
        correction_bias=topk_config.correction_bias,
        renormalize=topk_config.renormalize,
        routed_scaling_factor=topk_config.routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            topk_config.apply_routed_scaling_factor_on_output
        ),
        topk_indices_dtype=topk_config.topk_indices_dtype,
    )
    return topk_to_ragged_metadata(
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        metadata_factory=metadata_factory,
        gate_dtype=gate_dtype,
    )


def topk_to_ragged_metadata(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    metadata_factory: Callable[[torch.Tensor, int], Any],
    gate_dtype: torch.dtype | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert selected top-k ids/weights to the local MXFP4 ragged contract."""

    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be 2D, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    flat_ids = topk_ids.reshape(-1).to(torch.long)
    if flat_ids.numel() > 0:
        min_id = int(flat_ids.min().item())
        max_id = int(flat_ids.max().item())
        if min_id < 0 or max_id >= num_experts:
            raise ValueError(
                f"topk_ids must be in [0, {num_experts}), got range "
                f"[{min_id}, {max_id}]"
            )

    sort_order = torch.argsort(flat_ids, stable=True)
    top_k = topk_ids.shape[1]
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    gate_scal = topk_weights.reshape(-1)[sort_order]
    if gate_dtype is not None and gate_scal.dtype != gate_dtype:
        gate_scal = gate_scal.to(gate_dtype)

    col_sum = torch.bincount(flat_ids, minlength=num_experts).to(torch.int32)
    ragged_metadata = metadata_factory(col_sum, topk_ids.numel())
    return ragged_metadata, gather_indx, scatter_indx, gate_scal


def _stable_descending_topk_ids(
    choice_scores: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    num_tokens, num_experts = choice_scores.shape
    remaining = choice_scores.clone()
    expert_offsets = torch.arange(num_experts, device=choice_scores.device)
    fallback_ids = torch.full_like(expert_offsets, num_experts)
    ids = []
    for _ in range(top_k):
        best_score = remaining.max(dim=1, keepdim=True).values
        candidate_ids = torch.where(
            remaining == best_score,
            expert_offsets.unsqueeze(0),
            fallback_ids.unsqueeze(0),
        )
        best_expert = candidate_ids.min(dim=1).values
        ids.append(best_expert)
        remaining.scatter_(1, best_expert.unsqueeze(1), -float("inf"))
    return torch.stack(ids, dim=1)


__all__ = [
    "is_kimi_sigmoid_noaux_topk_config",
    "mxfp4_kimi_sigmoid_ragged_route_from_bypassed",
    "select_kimi_sigmoid_noaux_topk",
    "topk_to_ragged_metadata",
]
