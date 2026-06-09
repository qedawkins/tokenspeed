from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.moe.backends.mxfp4.routing as routing_module
from tokenspeed.runtime.layers.moe.backends.mxfp4.routing import (
    is_kimi_sigmoid_noaux_topk_config,
    mxfp4_kimi_sigmoid_ragged_route_from_bypassed,
    select_kimi_sigmoid_noaux_topk,
    topk_to_ragged_metadata,
)


def _require_cdna4_gpu() -> None:
    try:
        from tokenspeed_kernel.platform import current_platform
    except (ImportError, RuntimeError):
        pytest.skip("AMD CDNA4 GPU is required for the MXFP4 routing capture test")

    if not torch.cuda.is_available() or not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the MXFP4 routing capture test")


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


def test_kimi_sigmoid_noaux_topk_uses_available_kernel_output(monkeypatch) -> None:
    router_logits = torch.randn(2, 5, dtype=torch.float32)
    correction_bias = torch.zeros(5, dtype=torch.float32)
    hidden_states = torch.randn(2, 7, dtype=torch.float32)
    kernel_weights = torch.tensor(
        [[0.6, 0.4], [0.75, 0.25]],
        dtype=torch.float32,
    )
    kernel_ids = torch.tensor([[2, 1], [4, 3]], dtype=torch.int32)
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return kernel_weights, kernel_ids

    monkeypatch.setattr(
        routing_module,
        "_try_select_kimi_sigmoid_noaux_topk_kernel",
        fake_kernel,
    )

    weights, ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=2,
        correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
        topk_indices_dtype=torch.int64,
        hidden_states=hidden_states,
    )

    torch.testing.assert_close(weights, kernel_weights)
    assert torch.equal(ids, kernel_ids.to(torch.int64))
    assert calls[0][0][0] is router_logits
    assert calls[0][1]["hidden_states"] is hidden_states
    assert calls[0][1]["top_k"] == 2


def test_kimi_sigmoid_noaux_topk_gpu_kernel_captures_kimi_like_shape() -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7000)
    hidden_states = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    router_logits = torch.randn(8, 384, device="cuda", dtype=torch.bfloat16)
    correction_bias = (
        torch.randn(384, device="cuda", dtype=torch.float32) * 0.05
    )

    expected_weights, expected_ids = select_kimi_sigmoid_noaux_topk(
        router_logits.float().cpu(),
        top_k=8,
        correction_bias=correction_bias.cpu(),
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )
    kernel_output = routing_module._try_select_kimi_sigmoid_noaux_topk_kernel(
        router_logits,
        top_k=8,
        correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
        hidden_states=hidden_states,
    )
    assert kernel_output is not None
    kernel_weights, kernel_ids = kernel_output
    assert torch.equal(kernel_ids.cpu(), expected_ids)
    torch.testing.assert_close(
        kernel_weights.cpu(),
        expected_weights,
        atol=5e-4,
        rtol=5e-4,
    )

    weights, ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=8,
        correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
        hidden_states=hidden_states,
    )
    torch.cuda.synchronize()

    assert torch.equal(ids.cpu(), expected_ids)
    torch.testing.assert_close(
        weights.cpu(),
        expected_weights,
        atol=5e-4,
        rtol=5e-4,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_weights, captured_ids = select_kimi_sigmoid_noaux_topk(
            router_logits,
            top_k=8,
            correction_bias=correction_bias,
            renormalize=True,
            routed_scaling_factor=1.0,
            apply_routed_scaling_factor_on_output=False,
            hidden_states=hidden_states,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured_ids.cpu(), expected_ids)
    torch.testing.assert_close(
        captured_weights.cpu(),
        expected_weights,
        atol=5e-4,
        rtol=5e-4,
    )


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


def test_topk_to_ragged_metadata_skips_host_range_check_during_capture(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.execution.cuda_graph_wrapper",
        SimpleNamespace(get_is_capture_mode=lambda: True),
    )

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("range validation must not run during graph capture")

    monkeypatch.setattr(routing_module, "_validate_topk_id_range", fail_validation)

    def fail_bincount(*_args, **_kwargs):
        raise AssertionError("torch.bincount must not run during graph capture")

    monkeypatch.setattr(torch, "bincount", fail_bincount)

    topk_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32)

    metadata, gather_indx, scatter_indx, gate_scal = topk_to_ragged_metadata(
        topk_ids,
        topk_weights,
        num_experts=3,
        metadata_factory=_metadata_factory,
    )

    flat_ids = topk_ids.reshape(-1).long()
    sort_order = torch.argsort(flat_ids, stable=True)
    assert torch.equal(metadata.col_sum, torch.tensor([1, 1, 2], dtype=torch.int32))
    assert torch.equal(gather_indx, (sort_order // topk_ids.shape[1]).to(torch.int32))
    assert torch.equal(scatter_indx, sort_order.to(torch.int32))
    torch.testing.assert_close(gate_scal, topk_weights.reshape(-1)[sort_order])
