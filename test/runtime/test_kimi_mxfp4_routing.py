from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.mxfp4.routing import (
    is_kimi_sigmoid_noaux_topk_config,
    mxfp4_kimi_sigmoid_ragged_route_from_bypassed,
    select_kimi_sigmoid_noaux_topk,
    topk_to_ragged_metadata,
)


def _metadata_factory(col_sum: torch.Tensor, n_total_rows: int) -> SimpleNamespace:
    return SimpleNamespace(col_sum=col_sum.clone(), n_total_rows=n_total_rows)


def _topk_config(**overrides) -> SimpleNamespace:
    config = dict(
        top_k=3,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        correction_bias=torch.zeros(5, dtype=torch.float32),
        renormalize=True,
        routed_scaling_factor=2.827,
        apply_routed_scaling_factor_on_output=True,
        num_fused_shared_experts=0,
        custom_routing_function=None,
        topk_indices_dtype=torch.int32,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_kimi_sigmoid_noaux_topk_uses_bias_for_choice_not_weight() -> None:
    router_logits = torch.tensor(
        [
            [0.0, 0.0, 3.0, -2.0, 1.0],
            [1.5, -0.5, 0.2, -1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    correction_bias = torch.tensor([0.3, 0.9, -0.2, 1.2, 0.0])

    weights, ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=3,
        correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=2.827,
        apply_routed_scaling_factor_on_output=True,
    )

    assert ids.tolist() == [[1, 3, 0], [3, 1, 0]]
    raw_scores = router_logits.sigmoid()
    gathered = raw_scores.gather(1, ids.long())
    expected = gathered / gathered.sum(dim=1, keepdim=True) * 2.827
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(dim=1), torch.full((2,), 2.827))
    assert weights[0, 1] < weights[0, 0]


def test_kimi_sigmoid_noaux_topk_uses_lowest_expert_for_ties() -> None:
    router_logits = torch.zeros(1, 5, dtype=torch.float32)
    correction_bias = torch.tensor([0.5, 0.5, 0.25, 0.25, 0.0])

    weights, ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=4,
        correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )

    assert ids.tolist() == [[0, 1, 2, 3]]
    torch.testing.assert_close(weights, torch.full((1, 4), 0.25))


def test_kimi_sigmoid_noaux_topk_does_not_scale_without_normalization() -> None:
    router_logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    correction_bias = torch.tensor([0.0, 0.0, 0.0])

    weights, ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=2,
        correction_bias=correction_bias,
        renormalize=False,
        routed_scaling_factor=99.0,
        apply_routed_scaling_factor_on_output=True,
    )

    assert ids.tolist() == [[2, 1]]
    torch.testing.assert_close(weights, router_logits.sigmoid().gather(1, ids.long()))


def test_kimi_sigmoid_config_detection_is_narrow() -> None:
    assert is_kimi_sigmoid_noaux_topk_config(_topk_config())
    assert not is_kimi_sigmoid_noaux_topk_config(
        _topk_config(correction_bias=None)
    )
    assert not is_kimi_sigmoid_noaux_topk_config(
        _topk_config(num_expert_group=2)
    )
    assert not is_kimi_sigmoid_noaux_topk_config(
        _topk_config(custom_routing_function=lambda **_: None)
    )


def test_mxfp4_kimi_ragged_route_matches_selected_topk_contract() -> None:
    router_logits = torch.tensor(
        [
            [0.0, 0.0, 3.0, -2.0, 1.0],
            [1.5, -0.5, 0.2, -1.0, 0.1],
            [-0.5, 2.0, 0.3, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    correction_bias = torch.tensor([0.3, 0.9, -0.2, 1.2, 0.0])
    config = _topk_config(correction_bias=correction_bias)
    topk_output = SimpleNamespace(
        hidden_states=torch.empty(router_logits.shape[0], 7),
        router_logits=router_logits,
        topk_config=config,
    )

    metadata, gather_indx, scatter_indx, gate_scal = (
        mxfp4_kimi_sigmoid_ragged_route_from_bypassed(
            topk_output,
            num_experts=router_logits.shape[1],
            metadata_factory=_metadata_factory,
            gate_dtype=torch.float16,
        )
    )

    expected_weights, expected_ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=config.top_k,
        correction_bias=correction_bias,
        renormalize=config.renormalize,
        routed_scaling_factor=config.routed_scaling_factor,
        apply_routed_scaling_factor_on_output=(
            config.apply_routed_scaling_factor_on_output
        ),
    )
    flat_ids = expected_ids.reshape(-1).long()
    sort_order = torch.argsort(flat_ids, stable=True)

    assert metadata.n_total_rows == expected_ids.numel()
    assert torch.equal(
        metadata.col_sum,
        torch.bincount(flat_ids, minlength=router_logits.shape[1]).to(torch.int32),
    )
    assert torch.equal(gather_indx, (sort_order // config.top_k).to(torch.int32))
    assert torch.equal(scatter_indx, sort_order.to(torch.int32))
    torch.testing.assert_close(
        gate_scal,
        expected_weights.reshape(-1)[sort_order].to(torch.float16),
    )


def test_topk_to_ragged_metadata_rejects_bad_ids() -> None:
    with pytest.raises(ValueError, match="topk_ids must be in"):
        topk_to_ragged_metadata(
            torch.tensor([[0, 5]], dtype=torch.int32),
            torch.ones(1, 2),
            num_experts=5,
            metadata_factory=_metadata_factory,
        )
