from __future__ import annotations

import pytest
import torch

from test.runtime.layers.test_moe_mxfp4_ep_dispatch import (
    _reference_ep_metadata,
    _workspace,
)
from test.runtime.layers.test_moe_mxfp4_experts import (
    _dense_owner_reference,
    _random_e8m0_scales,
    _random_packed_weight,
)
from tokenspeed.runtime.layers.moe.backends.ep_ownership import (
    build_uniform_expert_owner_maps,
)
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    prepare_owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4 import (
    ep_down_combine as ep_down_combine_module,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    dequantize_mxfp4_activation,
    quantize_mxfp4_activation_reference,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_down_combine import (
    mxfp4_ep_down_gemm_combine,
    owner_rank_mxfp4_down_from_gate_up,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_gate_up import (
    dispatch_mxfp4_hidden_states_gate_up,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
    dequantize_mxfp4_expert_weight,
)


def test_owner_rank_mxfp4_ep_down_matches_dequantized_reference() -> None:
    torch.manual_seed(20270)
    intermediate_size = 64
    hidden_size = 24
    local_counts = torch.tensor([4, 0, 3], dtype=torch.int32)
    local_offsets = torch.tensor([0, 4, 4, 7], dtype=torch.int32)
    gate_up = torch.randn(7, intermediate_size, dtype=torch.float32).to(torch.bfloat16)
    gate_up[2].mul_(9.0)
    packed_weight = _random_packed_weight(3, hidden_size, intermediate_size)
    weight_scale = _random_e8m0_scales(3, hidden_size, intermediate_size)
    bias = torch.randn(3, hidden_size, dtype=torch.float32) * 0.02

    actual = owner_rank_mxfp4_down_from_gate_up(
        gate_up,
        packed_weight,
        weight_scale,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
        output_dtype=torch.float32,
    )

    packed_intermediate, intermediate_scale = quantize_mxfp4_activation_reference(
        gate_up
    )
    dequant_intermediate = dequantize_mxfp4_activation(
        packed_intermediate,
        intermediate_scale,
        logical_shape=tuple(gate_up.shape),
    )
    dense_weight = dequantize_mxfp4_expert_weight(packed_weight, weight_scale)
    expected = _dense_owner_reference(
        dequant_intermediate,
        dense_weight,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_mxfp4_ep_down_combine_matches_all_local_distributed_reference() -> None:
    torch.manual_seed(20271)
    rank = 0
    world_size = 4
    num_local_experts = 96
    hidden_size = 64
    intermediate_size = 64
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [
            [0.40, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            [0.80, 0.10, 0.70, 0.20, 0.60, 0.30, 0.50, 0.40],
        ],
        dtype=torch.float32,
    )
    hidden_states = (
        torch.randn(topk_ids.shape[0], hidden_size, dtype=torch.float32) * 0.20
    ).to(torch.bfloat16)
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=world_size,
        device="cpu",
    )
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        dtype=hidden_states.dtype,
    )
    gate_up_weight = _random_packed_weight(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
    )
    gate_up_scale = _random_e8m0_scales(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
    )
    down_weight = _random_packed_weight(
        num_local_experts,
        hidden_size,
        intermediate_size,
    )
    down_scale = _random_e8m0_scales(num_local_experts, hidden_size, intermediate_size)
    down_bias = torch.randn(num_local_experts, hidden_size, dtype=torch.float32) * 0.01
    routed_scaling_factor = 2.827

    gate_up_result = dispatch_mxfp4_hidden_states_gate_up(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        gate_up_weight,
        gate_up_scale,
        output_dtype=torch.bfloat16,
    )
    actual = mxfp4_ep_down_gemm_combine(
        gate_up_result.gate_up,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        gate_up_result.fused_metadata,
        gate_up_result.dispatch_plan,
        down_weight,
        down_scale,
        expert_owner,
        local_expert_id,
        bias=down_bias,
        routed_scaling_factor=routed_scaling_factor,
    )

    packed_intermediate, intermediate_scale = quantize_mxfp4_activation_reference(
        gate_up_result.gate_up
    )
    dequant_intermediate = dequantize_mxfp4_activation(
        packed_intermediate,
        intermediate_scale,
        logical_shape=tuple(gate_up_result.gate_up.shape),
    )
    dense_down = dequantize_mxfp4_expert_weight(down_weight, down_scale)
    owner_outputs = _dense_owner_reference(
        dequant_intermediate,
        dense_down,
        gate_up_result.dispatch_plan.local_expert_counts,
        local_expert_offsets=gate_up_result.dispatch_plan.local_expert_offsets,
        bias=down_bias,
    )
    expected_slots = torch.zeros(
        (*topk_ids.shape, hidden_size),
        dtype=torch.float32,
    )
    for flat_slot, owner_row_tensor in enumerate(ep_metadata.dispatch_offsets.reshape(-1)):
        owner_row = int(owner_row_tensor.item())
        token = flat_slot // topk_ids.shape[1]
        slot = flat_slot - token * topk_ids.shape[1]
        expected_slots[token, slot] = owner_outputs[owner_row]
    expected_slots = expected_slots.to(workspace.dtype)
    expected_weighted_slots = (
        expected_slots.float() * topk_weights.float().unsqueeze(-1)
    ).to(expected_slots.dtype)
    expected = expected_weighted_slots.float().sum(dim=1) * routed_scaling_factor
    expected = expected.to(workspace.dtype)

    torch.testing.assert_close(actual.returned_slots.float(), expected_slots.float())
    torch.testing.assert_close(
        actual.output.float(),
        expected.float(),
        atol=0.15,
        rtol=0.05,
    )


def test_mxfp4_ep_down_combine_reduces_all_remote_returned_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 0
    world_size = 4
    num_local_experts = 96
    hidden_size = 64
    intermediate_size = 64
    topk_ids = torch.tensor(
        [
            [96, 97, 98, 99, 100, 101, 102, 103],
            [192, 193, 194, 195, 288, 289, 290, 291],
        ],
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.1,
        0.8,
        steps=topk_ids.numel(),
        dtype=torch.float32,
    ).reshape(topk_ids.shape)
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=world_size,
        device="cpu",
    )
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        dtype=torch.bfloat16,
    )
    dispatch_plan = prepare_owner_directed_dispatch(ep_metadata, workspace)
    remote_slots = (
        torch.arange(
            topk_ids.numel() * hidden_size,
            dtype=torch.float32,
        )
        .reshape(*topk_ids.shape, hidden_size)
        .remainder(29)
        .sub(14)
        .mul(0.125)
        .to(torch.bfloat16)
    )

    def fake_combine(
        combine_buffer: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> torch.Tensor:
        assert combine_buffer.shape == (0, hidden_size)
        return remote_slots

    monkeypatch.setattr(
        ep_down_combine_module,
        "owner_directed_combine",
        fake_combine,
    )

    actual = mxfp4_ep_down_gemm_combine(
        torch.empty((0, intermediate_size), dtype=torch.bfloat16),
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        object(),
        dispatch_plan,
        _random_packed_weight(num_local_experts, hidden_size, intermediate_size),
        _random_e8m0_scales(num_local_experts, hidden_size, intermediate_size),
        expert_owner,
        local_expert_id,
        routed_scaling_factor=1.75,
    )

    expected_weighted_slots = (
        remote_slots.float() * topk_weights.float().unsqueeze(-1)
    ).to(remote_slots.dtype)
    expected = expected_weighted_slots.float().sum(dim=1) * 1.75
    expected = expected.to(remote_slots.dtype)

    assert actual.owner_outputs.shape == (0, hidden_size)
    torch.testing.assert_close(actual.returned_slots, remote_slots)
    torch.testing.assert_close(actual.output, expected)


def test_mxfp4_ep_down_combine_handles_zero_tokens() -> None:
    topk_ids = torch.empty((0, 8), dtype=torch.int32)
    hidden_states = torch.empty((0, 64), dtype=torch.bfloat16)
    topk_weights = torch.empty((0, 8), dtype=torch.float32)
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=4,
        device="cpu",
    )
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=0,
        world_size=4,
        num_local_experts=96,
    )
    workspace = _workspace(
        world_size=4,
        rank=0,
        hidden_size=64,
        max_tokens_per_rank=0,
        top_k=8,
        dtype=torch.bfloat16,
    )
    gate_up_weight = _random_packed_weight(96, 128, 64)
    gate_up_scale = _random_e8m0_scales(96, 128, 64)
    down_weight = _random_packed_weight(96, 64, 64)
    down_scale = _random_e8m0_scales(96, 64, 64)

    gate_up_result = dispatch_mxfp4_hidden_states_gate_up(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        gate_up_weight,
        gate_up_scale,
        output_dtype=torch.bfloat16,
    )
    actual = mxfp4_ep_down_gemm_combine(
        gate_up_result.gate_up,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        gate_up_result.fused_metadata,
        gate_up_result.dispatch_plan,
        down_weight,
        down_scale,
        expert_owner,
        local_expert_id,
    )

    assert actual.output.shape == (0, 64)
    assert actual.returned_slots.shape == (0, 8, 64)
    assert actual.owner_outputs.shape == (0, 64)
