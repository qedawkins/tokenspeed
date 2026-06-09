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

"""Grouped biased MoE top-k route Gluon kernel for AMD GFX950."""

from __future__ import annotations

from typing import Optional

import torch
from tokenspeed_kernel._triton import gl, gluon, tl
from tokenspeed_kernel.ops.moe.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


_ROUTE_TOPK_SIGNATURES = format_signatures(
    "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
)
_INV_LN2 = tl.constexpr(1.4426950408889634)
_COMMON_TOPK = frozenset(range(1, 17))
_COMMON_GROUPS = frozenset({1, 2, 4, 8, 16, 32})


@gluon.jit
def _maximum(a, b):
    return gl.maximum(a, b)


@gluon.jit
def _minimum(a, b):
    return gl.minimum(a, b)


@gluon.jit
def _max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, _maximum, keep_dims=keep_dims)


@gluon.jit
def _min(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, _minimum, keep_dims=keep_dims)


@gluon.jit
def _group_top2_sum(
    choice_scores,
    expert_offsets,
    expert_mask,
    group_id: gl.constexpr,
    experts_per_group: gl.constexpr,
    block_e: gl.constexpr,
):
    group_start = group_id * experts_per_group
    group_end = group_start + experts_per_group
    in_group = (expert_offsets >= group_start) & (expert_offsets < group_end)
    group_scores = gl.where(in_group & expert_mask, choice_scores, -float("inf"))

    top1_score = _max(group_scores, axis=0)
    top1_expert = _min(
        gl.where(group_scores == top1_score, expert_offsets, block_e),
        axis=0,
    )
    second_scores = gl.where(expert_offsets == top1_expert, -float("inf"), group_scores)
    top2_score = _max(second_scores, axis=0)
    return top1_score + top2_score


@gluon.jit
def _grouped_biased_topk_kernel(
    gating_output_ptr,
    correction_bias_ptr,
    num_token_non_padded_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    GATING_STRIDE_M: gl.constexpr,
    GATING_STRIDE_E: gl.constexpr,
    WEIGHTS_STRIDE_M: gl.constexpr,
    WEIGHTS_STRIDE_K: gl.constexpr,
    IDS_STRIDE_M: gl.constexpr,
    IDS_STRIDE_K: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    NUM_EXPERT_GROUP: gl.constexpr,
    EXPERTS_PER_GROUP: gl.constexpr,
    TOPK_GROUP: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_E_ELEMS_PER_THREAD: gl.constexpr,
    RENORMALIZE: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    HAS_NUM_TOKEN_NON_PADDED: gl.constexpr,
    APPLY_ROUTED_SCALING_FACTOR_ON_OUTPUT: gl.constexpr,
):
    token_id = gl.program_id(0)
    expert_offsets = gl.arange(
        0,
        BLOCK_E,
        layout=gl.BlockedLayout([BLOCK_E_ELEMS_PER_THREAD], [64], [1], [0]),
    )
    expert_mask = expert_offsets < NUM_EXPERTS

    logits = gl.load(
        gating_output_ptr
        + token_id * GATING_STRIDE_M
        + expert_offsets * GATING_STRIDE_E,
        mask=expert_mask,
        other=-float("inf"),
    ).to(gl.float32)
    bias = gl.load(
        correction_bias_ptr + expert_offsets,
        mask=expert_mask,
        other=0.0,
    ).to(gl.float32)

    scores = 1.0 / (1.0 + gl.exp2(-logits * _INV_LN2))
    choice_scores = gl.where(expert_mask, scores + bias, -float("inf"))
    expert_group = expert_offsets // EXPERTS_PER_GROUP

    own_group_score = gl.full(
        [BLOCK_E],
        -float("inf"),
        dtype=gl.float32,
        layout=gl.BlockedLayout([BLOCK_E_ELEMS_PER_THREAD], [64], [1], [0]),
    )
    for group_id in range(0, NUM_EXPERT_GROUP):
        group_score = _group_top2_sum(
            choice_scores,
            expert_offsets,
            expert_mask,
            group_id,
            EXPERTS_PER_GROUP,
            BLOCK_E,
        )
        own_group_score = gl.where(
            expert_group == group_id,
            group_score,
            own_group_score,
        )

    group_rank = gl.full(
        [BLOCK_E],
        0,
        dtype=gl.int32,
        layout=gl.BlockedLayout([BLOCK_E_ELEMS_PER_THREAD], [64], [1], [0]),
    )
    for group_id in range(0, NUM_EXPERT_GROUP):
        group_score = _group_top2_sum(
            choice_scores,
            expert_offsets,
            expert_mask,
            group_id,
            EXPERTS_PER_GROUP,
            BLOCK_E,
        )
        outranks = (group_score > own_group_score) | (
            (group_score == own_group_score) & (group_id < expert_group)
        )
        group_rank += outranks.to(gl.int32)

    selected_group = group_rank < TOPK_GROUP
    masked_choice_scores = gl.where(
        selected_group & expert_mask,
        choice_scores,
        -float("inf"),
    )

    weights_sum = gl.full((), 0.0, dtype=gl.float32)
    for topk_idx in range(0, TOPK):
        best_choice_score = _max(masked_choice_scores, axis=0)
        best_expert = _min(
            gl.where(masked_choice_scores == best_choice_score, expert_offsets, BLOCK_E),
            axis=0,
        )
        best_weight = _max(
            gl.where(expert_offsets == best_expert, scores, 0.0),
            axis=0,
        )
        weights_sum += best_weight

        gl.store(
            topk_ids_ptr + token_id * IDS_STRIDE_M + topk_idx * IDS_STRIDE_K,
            best_expert.to(gl.int32),
        )
        gl.store(
            topk_weights_ptr
            + token_id * WEIGHTS_STRIDE_M
            + topk_idx * WEIGHTS_STRIDE_K,
            best_weight,
        )
        masked_choice_scores = gl.where(
            expert_offsets == best_expert,
            -float("inf"),
            masked_choice_scores,
        )

    if RENORMALIZE:
        denom = gl.where(weights_sum != 0.0, weights_sum, 1.0)
        for topk_idx in range(0, TOPK):
            weight = gl.load(
                topk_weights_ptr
                + token_id * WEIGHTS_STRIDE_M
                + topk_idx * WEIGHTS_STRIDE_K
            )
            weight = weight / denom
            if APPLY_ROUTED_SCALING_FACTOR_ON_OUTPUT:
                weight *= ROUTED_SCALING_FACTOR
            gl.store(
                topk_weights_ptr
                + token_id * WEIGHTS_STRIDE_M
                + topk_idx * WEIGHTS_STRIDE_K,
                weight,
            )

    if HAS_NUM_TOKEN_NON_PADDED:
        num_token_non_padded = gl.load(num_token_non_padded_ptr)
        if token_id >= num_token_non_padded:
            for topk_idx in range(0, TOPK):
                gl.store(
                    topk_ids_ptr + token_id * IDS_STRIDE_M + topk_idx * IDS_STRIDE_K,
                    -1,
                )


def _biased_grouped_topk_reference(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = 1.0,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    from tokenspeed_kernel.numerics.reference.moe import biased_grouped_topk_gpu

    return biased_grouped_topk_gpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )


@register_kernel(
    "moe",
    "route",
    name="gluon_grouped_biased_topk_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_ROUTE_TOPK_SIGNATURES,
    traits={
        "output_type": frozenset({"topk"}),
        "biased": frozenset({True}),
        "grouped": frozenset({True}),
        "ep": frozenset({True, False}),
        "num_expert_group": _COMMON_GROUPS,
        "topk_group": _COMMON_GROUPS,
        "topk": _COMMON_TOPK,
        "num_fused_shared_experts": frozenset({0}),
    },
    priority=Priority.SPECIALIZED,
    tags={"latency", "determinism"},
)
def gluon_grouped_biased_topk_gfx950(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = 1.0,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    if (
        gating_output.ndim != 2
        or correction_bias.ndim != 1
        or hidden_states.shape[0] != gating_output.shape[0]
        or gating_output.shape[1] != correction_bias.shape[0]
        or num_expert_group is None
        or topk_group is None
        or num_expert_group <= 0
        or topk_group <= 0
        or gating_output.shape[1] % num_expert_group != 0
        or topk_group > num_expert_group
        or topk <= 0
        or topk > gating_output.shape[1]
        or gating_output.shape[1] > 512
        or num_fused_shared_experts != 0
        or routed_scaling_factor is None
    ):
        return _biased_grouped_topk_reference(
            hidden_states,
            gating_output,
            correction_bias,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )

    num_tokens, num_experts = gating_output.shape
    topk_weights = torch.empty(
        (num_tokens, topk),
        device=gating_output.device,
        dtype=torch.float32,
    )
    topk_ids = torch.empty(
        (num_tokens, topk),
        device=gating_output.device,
        dtype=torch.int32,
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    block_e = 1 << (num_experts - 1).bit_length()
    block_e = max(block_e, 64)
    num_token_non_padded_arg = (
        num_token_non_padded if num_token_non_padded is not None else topk_ids
    )
    _grouped_biased_topk_kernel[(num_tokens,)](
        gating_output,
        correction_bias,
        num_token_non_padded_arg,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_experts,
        num_expert_group,
        num_experts // num_expert_group,
        topk_group,
        topk,
        block_e,
        block_e // 64,
        renormalize,
        float(routed_scaling_factor),
        num_token_non_padded is not None,
        bool(apply_routed_scaling_factor_on_output),
        num_warps=1,
    )
    return topk_weights, topk_ids
