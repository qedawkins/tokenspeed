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

import pytest
import tokenspeed_kernel
import torch

from tokenspeed.runtime.layers.moe.backends.ep_self_routing import (
    self_routing_topk,
    self_routing_topk_from_bypassed,
)
from tokenspeed.runtime.layers.moe.topk import (
    TopK,
    TopKConfig,
    TopKOutputFormat,
    select_experts,
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for self-routing fused top-k validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for self-routing fused top-k validation")


def _route_traits(*, n_group: int, topk_group: int, top_k: int) -> dict:
    return {
        "output_type": "topk",
        "biased": True,
        "grouped": True,
        "ep": True,
        "num_expert_group": n_group,
        "topk_group": topk_group,
        "topk": top_k,
        "num_fused_shared_experts": 0,
    }


def _explicit_grouped_biased_route(
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    renormalize: bool = True,
    routed_scaling_factor: float | None = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
):
    hidden_states = _routing_hidden_states(router_logits)
    return tokenspeed_kernel.moe_route(
        hidden_states,
        router_logits,
        routing_bias,
        topk=top_k,
        renormalize=renormalize,
        num_expert_group=n_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        dtype=router_logits.dtype,
        traits=_route_traits(n_group=n_group, topk_group=topk_group, top_k=top_k),
    )


def _routing_hidden_states(router_logits: torch.Tensor) -> torch.Tensor:
    return router_logits.new_empty((router_logits.shape[0], 0))


def _assert_same_topk(actual, expected_weights, expected_ids) -> None:
    assert actual.topk_ids.dtype == torch.int32
    assert actual.topk_weights.dtype == torch.float32
    assert actual.router_logits.shape[0] == actual.topk_ids.shape[0]
    assert torch.equal(actual.topk_ids, expected_ids)
    torch.testing.assert_close(
        actual.topk_weights,
        expected_weights,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 32])
def test_self_routing_topk_matches_explicit_grouped_biased_route(
    num_tokens: int,
) -> None:
    _require_cdna4_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts = 16
    top_k = 4
    n_group = 4
    topk_group = 2

    torch.manual_seed(9100 + num_tokens)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_bias = torch.linspace(-0.10, 0.20, num_experts, device=device)

    actual = self_routing_topk(
        router_logits,
        routing_bias=routing_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )
    expected_weights, expected_ids = _explicit_grouped_biased_route(
        router_logits,
        routing_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )
    torch.cuda.synchronize()

    _assert_same_topk(actual, expected_weights, expected_ids)


def test_self_routing_topk_preserves_ties_and_routed_scaling() -> None:
    _require_cdna4_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    router_logits = torch.zeros(1, 8, device=device, dtype=torch.float32)
    routing_bias = torch.tensor(
        [0.20, 0.20, 0.15, 0.15, 0.90, 0.90, 0.89, 0.89],
        device=device,
    )
    routed_scaling_factor = 2.5

    actual = self_routing_topk(
        router_logits,
        routing_bias=routing_bias,
        top_k=4,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    expected_weights, expected_ids = _explicit_grouped_biased_route(
        router_logits,
        routing_bias,
        top_k=4,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    torch.cuda.synchronize()

    _assert_same_topk(actual, expected_weights, expected_ids)
    assert actual.topk_ids.tolist() == [[4, 5, 6, 7]]
    torch.testing.assert_close(
        actual.topk_weights.sum(dim=1),
        torch.tensor([routed_scaling_factor], device=device),
    )


def test_self_routing_topk_treats_top_k_as_config_value_for_small_decode() -> None:
    _require_cdna4_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    router_logits = torch.randn(1, 16, device=device, dtype=torch.float32)
    routing_bias = torch.zeros(16, device=device)

    actual = self_routing_topk(
        router_logits,
        routing_bias=routing_bias,
        top_k=3,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=1.0,
    )
    expected_weights, expected_ids = _explicit_grouped_biased_route(
        router_logits,
        routing_bias,
        top_k=3,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=1.0,
    )
    torch.cuda.synchronize()

    assert actual.topk_ids.shape == (1, 3)
    assert actual.topk_weights.shape == (1, 3)
    _assert_same_topk(actual, expected_weights, expected_ids)


def test_self_routing_topk_matches_unbiased_explicit_topk() -> None:
    _require_cdna4_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(9211)
    hidden_states = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(4, 16, device=device, dtype=torch.float32)

    actual = self_routing_topk(
        router_logits,
        top_k=4,
        n_group=4,
        topk_group=2,
        renormalize=True,
        routed_scaling_factor=1.0,
    )
    explicit = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=4,
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=1.0,
            topk_indices_dtype=torch.int32,
        ),
    )
    torch.cuda.synchronize()

    _assert_same_topk(actual, explicit.topk_weights, explicit.topk_ids)


def test_self_routing_topk_consumes_existing_fused_bypassed_contract() -> None:
    _require_cdna4_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    hidden_states = torch.randn(2, 32, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(2, 16, device=device, dtype=torch.float32)
    routing_bias = torch.linspace(-0.15, 0.15, 16, device=device)
    topk = TopK(
        4,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        correction_bias=routing_bias,
        routed_scaling_factor=1.0,
        output_format=TopKOutputFormat.BYPASSED,
    )

    bypassed = topk(hidden_states, router_logits)
    assert bypassed.format.is_bypassed()
    actual = self_routing_topk_from_bypassed(bypassed)
    expected_weights, expected_ids = _explicit_grouped_biased_route(
        router_logits,
        routing_bias,
        top_k=4,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=1.0,
    )
    torch.cuda.synchronize()

    _assert_same_topk(actual, expected_weights, expected_ids)


def test_self_routing_topk_rejects_incomplete_group_config() -> None:
    router_logits = torch.empty(0, 16)
    with pytest.raises(ValueError, match="n_group and topk_group"):
        self_routing_topk(
            router_logits,
            top_k=4,
            n_group=4,
        )


def test_self_routing_topk_rejects_hidden_token_mismatch() -> None:
    hidden_states = torch.empty(2, 32)
    router_logits = torch.empty(1, 16)
    with pytest.raises(ValueError, match="same token dimension"):
        self_routing_topk(
            router_logits,
            top_k=4,
            hidden_states=hidden_states,
        )
