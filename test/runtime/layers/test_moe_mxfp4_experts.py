from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    dequantize_mxfp4_activation,
    quantize_mxfp4_activation_reference,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
    dequantize_mxfp4_expert_weight,
    kimi_swiglu_gate_up,
    owner_rank_mxfp4_expert_gemm,
    owner_rank_mxfp4_gate_up_gemm,
)


def test_dequantize_mxfp4_expert_weight_applies_pack_order_and_scales() -> None:
    row = torch.tensor(
        [0x21, 0x43, 0x65, 0x17] + [0] * 12,
        dtype=torch.uint8,
    )
    packed = torch.cat((row, row)).reshape(1, 1, 32)
    scales = torch.tensor([[[127, 128]]], dtype=torch.uint8)

    actual = dequantize_mxfp4_expert_weight(
        packed,
        scales,
        logical_shape=(1, 1, 64),
    )

    first_values = torch.tensor(
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5],
        dtype=torch.float32,
    )
    expected = torch.zeros(1, 1, 64, dtype=torch.float32)
    expected[0, 0, :8] = first_values
    expected[0, 0, 32:40] = first_values * 2.0
    torch.testing.assert_close(actual, expected)


def test_owner_rank_mxfp4_gate_up_matches_dequantized_dense_reference() -> None:
    torch.manual_seed(1729)
    num_experts = 2
    out_features = 12
    in_features = 64
    local_counts = torch.tensor([3, 2], dtype=torch.int32)
    owner_tokens = torch.randn(5, in_features, dtype=torch.float32).to(torch.bfloat16)
    packed_weight = _random_packed_weight(num_experts, out_features, in_features)
    scales = _random_e8m0_scales(num_experts, out_features, in_features)

    actual = owner_rank_mxfp4_expert_gemm(
        owner_tokens,
        packed_weight,
        scales,
        local_counts,
        output_dtype=torch.float32,
    )

    dense_weight = dequantize_mxfp4_expert_weight(packed_weight, scales)
    expected = _dense_owner_reference(owner_tokens, dense_weight, local_counts)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_owner_rank_mxfp4_down_uses_explicit_offsets_bias_and_empty_expert() -> None:
    torch.manual_seed(2718)
    num_experts = 3
    out_features = 8
    in_features = 64
    local_counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    local_offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    owner_tokens = torch.randn(5, in_features, dtype=torch.float32)
    packed_weight = _random_packed_weight(num_experts, out_features, in_features)
    scales = _random_e8m0_scales(num_experts, out_features, in_features)
    bias = torch.randn(num_experts, out_features, dtype=torch.float32) * 0.05
    out = torch.empty(5, out_features, dtype=torch.float32)

    actual = owner_rank_mxfp4_expert_gemm(
        owner_tokens,
        packed_weight,
        scales,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
        out=out,
    )

    dense_weight = dequantize_mxfp4_expert_weight(packed_weight, scales)
    expected = _dense_owner_reference(
        owner_tokens,
        dense_weight,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
    )
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_owner_rank_mxfp4_gate_up_consumes_dynamic_activation_layout() -> None:
    torch.manual_seed(3141)
    num_experts = 3
    out_features = 16
    in_features = 64
    local_counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    local_offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    owner_tokens = torch.randn(5, in_features, dtype=torch.float32)
    owner_tokens[0].zero_()
    owner_tokens[2].mul_(12.0)
    packed_tokens, token_scale = quantize_mxfp4_activation_reference(
        owner_tokens.to(torch.bfloat16)
    )
    packed_weight = _random_packed_weight(num_experts, out_features, in_features)
    weight_scale = _random_e8m0_scales(num_experts, out_features, in_features)
    bias = torch.randn(num_experts, out_features, dtype=torch.float32) * 0.05

    actual = owner_rank_mxfp4_gate_up_gemm(
        packed_tokens,
        token_scale,
        packed_weight,
        weight_scale,
        local_counts,
        local_expert_offsets=local_offsets,
        bias=bias,
    )

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
    expected = _kimi_swiglu_reference(gate_up)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_owner_rank_mxfp4_gate_up_handles_zero_rows_and_preallocated_out() -> None:
    packed_tokens, token_scale = quantize_mxfp4_activation_reference(
        torch.empty(0, 64, dtype=torch.bfloat16)
    )
    packed_weight = _random_packed_weight(2, 8, 64)
    weight_scale = _random_e8m0_scales(2, 8, 64)
    out = torch.empty(0, 4, dtype=torch.bfloat16)

    actual = owner_rank_mxfp4_gate_up_gemm(
        packed_tokens,
        token_scale,
        packed_weight,
        weight_scale,
        torch.tensor([0, 0], dtype=torch.int32),
        output_dtype=torch.bfloat16,
        out=out,
    )

    assert actual is out
    assert actual.shape == (0, 4)


def test_owner_rank_mxfp4_output_dtype_defaults_and_overrides() -> None:
    local_counts = torch.tensor([1], dtype=torch.int32)
    owner_tokens = torch.randn(1, 64, dtype=torch.float32).to(torch.bfloat16)
    packed_weight = _random_packed_weight(1, 4, 64)
    scales = _random_e8m0_scales(1, 4, 64)

    default_out = owner_rank_mxfp4_expert_gemm(
        owner_tokens,
        packed_weight,
        scales,
        local_counts,
    )
    fp32_out = owner_rank_mxfp4_expert_gemm(
        owner_tokens,
        packed_weight,
        scales,
        local_counts,
        output_dtype=torch.float32,
    )
    preallocated = torch.empty(1, 4, dtype=torch.float64)
    returned = owner_rank_mxfp4_expert_gemm(
        owner_tokens,
        packed_weight,
        scales,
        local_counts,
        out=preallocated,
    )

    assert default_out.dtype == torch.bfloat16
    assert fp32_out.dtype == torch.float32
    assert returned is preallocated
    assert returned.dtype == torch.float64


def test_owner_rank_mxfp4_expert_gemm_rejects_bad_owner_layouts() -> None:
    packed_weight = _random_packed_weight(2, 4, 64)
    scales = _random_e8m0_scales(2, 4, 64)
    owner_tokens = torch.randn(3, 64)

    with pytest.raises(ValueError, match="owner token rows"):
        owner_rank_mxfp4_expert_gemm(
            owner_tokens,
            packed_weight,
            scales,
            torch.tensor([2, 2], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="local_expert_offsets must match"):
        owner_rank_mxfp4_expert_gemm(
            owner_tokens,
            packed_weight,
            scales,
            torch.tensor([1, 2], dtype=torch.int32),
            local_expert_offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="scale shape"):
        owner_rank_mxfp4_expert_gemm(
            owner_tokens,
            packed_weight,
            scales[:, :, :1],
            torch.tensor([1, 2], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="local_expert_counts must be torch.int32"):
        owner_rank_mxfp4_expert_gemm(
            owner_tokens,
            packed_weight,
            scales,
            torch.tensor([1, 2], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        owner_rank_mxfp4_expert_gemm(
            owner_tokens,
            packed_weight,
            scales,
            torch.tensor([4, -1], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="local_expert_offsets must be torch.int32"):
        owner_rank_mxfp4_expert_gemm(
            owner_tokens,
            packed_weight,
            scales,
            torch.tensor([1, 2], dtype=torch.int32),
            local_expert_offsets=torch.tensor([0, 1, 3], dtype=torch.int64),
        )


def test_owner_rank_mxfp4_gate_up_rejects_bad_activation_layouts() -> None:
    packed_tokens, token_scale = quantize_mxfp4_activation_reference(
        torch.randn(2, 64, dtype=torch.float32)
    )
    packed_weight = _random_packed_weight(1, 7, 64)
    weight_scale = _random_e8m0_scales(1, 7, 64)

    with pytest.raises(ValueError, match="packed shape"):
        owner_rank_mxfp4_gate_up_gemm(
            packed_tokens[:, :31],
            token_scale,
            _random_packed_weight(1, 8, 64),
            _random_e8m0_scales(1, 8, 64),
            torch.tensor([2], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="gate/up output dim"):
        owner_rank_mxfp4_gate_up_gemm(
            packed_tokens,
            token_scale,
            packed_weight,
            weight_scale,
            torch.tensor([2], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="out shape"):
        kimi_swiglu_gate_up(
            torch.randn(2, 8),
            out=torch.empty(2, 5),
        )


def _random_packed_weight(
    num_experts: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if in_features % 2 != 0:
        raise ValueError("in_features must be divisible by 2")
    return torch.randint(
        0,
        256,
        (num_experts, out_features, in_features // 2),
        dtype=torch.uint8,
    )


def _random_e8m0_scales(
    num_experts: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by 32")
    return torch.randint(
        125,
        130,
        (num_experts, out_features, in_features // 32),
        dtype=torch.uint8,
    )


def _dense_owner_reference(
    owner_tokens: torch.Tensor,
    dense_weight: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if local_expert_offsets is None:
        local_expert_offsets = torch.empty(
            local_expert_counts.numel() + 1,
            dtype=torch.int32,
        )
        local_expert_offsets[0] = 0
        local_expert_offsets[1:] = torch.cumsum(
            local_expert_counts,
            dim=0,
            dtype=torch.int32,
        )

    expected = torch.empty(
        owner_tokens.shape[0],
        dense_weight.shape[1],
        dtype=torch.float32,
        device=owner_tokens.device,
    )
    for expert_idx in range(dense_weight.shape[0]):
        start = int(local_expert_offsets[expert_idx].item())
        end = int(local_expert_offsets[expert_idx + 1].item())
        if start == end:
            continue
        result = owner_tokens[start:end].float() @ dense_weight[expert_idx].T
        if bias is not None:
            result = result + bias[expert_idx].float()
        expected[start:end] = result
    return expected


def _kimi_swiglu_reference(
    gate_up: torch.Tensor,
    *,
    alpha: float = 1.702,
    limit: float | None = 7.0,
) -> torch.Tensor:
    gate = gate_up[..., 0::2].float()
    up = gate_up[..., 1::2].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)
