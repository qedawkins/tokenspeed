from __future__ import annotations

import torch

from test.runtime.layers.test_moe_mxfp4_ep_dispatch import (
    _expected_owner_rows,
    _mixed_kimi_topk_ids,
    _reference_ep_metadata,
    _workspace,
)
from test.runtime.layers.test_moe_mxfp4_experts import (
    _dense_owner_reference,
    _kimi_swiglu_reference,
    _random_e8m0_scales,
    _random_packed_weight,
)
from tokenspeed.runtime.layers.moe.backends.ep_ownership import (
    build_uniform_expert_owner_maps,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    dequantize_mxfp4_activation,
    quantize_mxfp4_activation_reference,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_gate_up import (
    dispatch_mxfp4_hidden_states_gate_up,
    owner_rank_mxfp4_gate_up_from_owner_tokens,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
    dequantize_mxfp4_expert_weight,
)


def test_owner_rank_mxfp4_ep_gate_up_accepts_multi_source_owner_rows() -> None:
    torch.manual_seed(20260)
    hidden_size = 64
    out_features = 16
    local_counts = torch.tensor([4, 0, 3], dtype=torch.int32)
    local_offsets = torch.tensor([0, 4, 4, 7], dtype=torch.int32)
    owner_tokens = torch.randn(7, hidden_size, dtype=torch.float32).to(torch.bfloat16)
    owner_tokens[0].zero_()
    owner_tokens[5].mul_(8.0)
    packed_weight = _random_packed_weight(3, out_features, hidden_size)
    weight_scale = _random_e8m0_scales(3, out_features, hidden_size)
    bias = torch.randn(3, out_features, dtype=torch.float32) * 0.02

    actual = owner_rank_mxfp4_gate_up_from_owner_tokens(
        owner_tokens,
        packed_weight,
        weight_scale,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
        output_dtype=torch.float32,
    )

    packed_tokens, token_scale = quantize_mxfp4_activation_reference(owner_tokens)
    dequant_tokens = dequantize_mxfp4_activation(
        packed_tokens,
        token_scale,
        logical_shape=tuple(owner_tokens.shape),
    )
    dense_weight = dequantize_mxfp4_expert_weight(packed_weight, weight_scale)
    gate_up = _dense_owner_reference(
        dequant_tokens,
        dense_weight,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
    )
    expected = _kimi_swiglu_reference(gate_up, layout="concatenated")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_dispatch_mxfp4_hidden_states_gate_up_matches_kimi_ep_oracle() -> None:
    torch.manual_seed(20261)
    rank = 2
    world_size = 4
    num_local_experts = 96
    hidden_size = 64
    out_features = 16
    topk_ids = _mixed_kimi_topk_ids()
    topk_weights = torch.linspace(
        0.05,
        0.95,
        steps=topk_ids.numel(),
        dtype=torch.float32,
    ).reshape(topk_ids.shape)
    hidden_states = (
        torch.randn(topk_ids.shape[0], hidden_size, dtype=torch.float32) * 0.25
    ).to(torch.bfloat16)
    hidden_states[1].mul_(6.0)
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
    packed_weight = _random_packed_weight(num_local_experts, out_features, hidden_size)
    weight_scale = _random_e8m0_scales(num_local_experts, out_features, hidden_size)
    bias = torch.randn(num_local_experts, out_features, dtype=torch.float32) * 0.01

    result = dispatch_mxfp4_hidden_states_gate_up(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        packed_weight,
        weight_scale,
        bias=bias,
        output_dtype=torch.float32,
    )

    expected_owner_tokens = _expected_owner_rows(
        hidden_states,
        topk_ids,
        ep_metadata,
        rank,
    )
    packed_tokens, token_scale = quantize_mxfp4_activation_reference(
        expected_owner_tokens
    )
    dequant_tokens = dequantize_mxfp4_activation(
        packed_tokens,
        token_scale,
        logical_shape=tuple(expected_owner_tokens.shape),
    )
    dense_weight = dequantize_mxfp4_expert_weight(packed_weight, weight_scale)
    gate_up = _dense_owner_reference(
        dequant_tokens,
        dense_weight,
        result.dispatch_plan.local_expert_counts,
        local_expert_offsets=result.dispatch_plan.local_expert_offsets,
        bias=bias,
    )
    expected_gate_up = _kimi_swiglu_reference(gate_up, layout="concatenated")

    torch.testing.assert_close(result.owner_tokens, expected_owner_tokens)
    torch.testing.assert_close(result.gate_up, expected_gate_up, atol=1e-5, rtol=1e-5)
    assert result.fused_metadata.source_rank == rank
    assert result.dispatch_plan.local_expert_counts.sum().item() == result.gate_up.shape[0]


def test_dispatch_mxfp4_gate_up_handles_empty_owner_rows() -> None:
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    rank = 3
    hidden_states = torch.randn(2, 64, dtype=torch.bfloat16)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=4,
        device="cpu",
    )
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=4,
        num_local_experts=96,
    )
    workspace = _workspace(
        world_size=4,
        rank=rank,
        hidden_size=64,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        dtype=hidden_states.dtype,
    )
    packed_weight = _random_packed_weight(96, 16, 64)
    weight_scale = _random_e8m0_scales(96, 16, 64)

    result = dispatch_mxfp4_hidden_states_gate_up(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        packed_weight,
        weight_scale,
        output_dtype=torch.bfloat16,
    )

    assert result.owner_tokens.shape == (0, 64)
    assert result.gate_up.shape == (0, 8)
    assert result.gate_up.dtype == torch.bfloat16
