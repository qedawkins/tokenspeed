"""CPU-only Kimi Quark MXFP4 reorder/layout preparation tests."""

from __future__ import annotations

from math import prod

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
    dequantize_mxfp4_expert_weight,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    prepare_mxfp4_for_layout_conversion,
    restore_mxfp4_from_layout_conversion,
)


def _uint8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    return ((values + offset) % 251).to(torch.uint8)


def _dequantize_linear(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return dequantize_mxfp4_expert_weight(
        packed.unsqueeze(0),
        scales.unsqueeze(0),
        logical_shape=(1, int(packed.shape[0]), int(packed.shape[1]) * 2),
    ).squeeze(0)


@pytest.mark.parametrize(
    ("name", "packed_shape", "scale_shape"),
    [
        ("w13_gate_up", (1, 64, 32), (1, 64, 2)),
        ("w2_down", (1, 64, 16), (1, 64, 1)),
    ],
)
def test_quark_mxfp4_expert_layout_prep_round_trips_dequantized_matrix(
    name: str,
    packed_shape: tuple[int, int, int],
    scale_shape: tuple[int, int, int],
) -> None:
    del name
    packed = _uint8_pattern(packed_shape, offset=17)
    scales = _uint8_pattern(scale_shape, offset=121)

    logical_shape = (
        packed_shape[0],
        packed_shape[1],
        packed_shape[2] * 2,
    )
    expected = dequantize_mxfp4_expert_weight(
        packed,
        scales,
        logical_shape=logical_shape,
    )

    prepared_weight, prepared_scale = prepare_mxfp4_for_layout_conversion(
        packed,
        scales,
    )
    restored_weight, restored_scale = restore_mxfp4_from_layout_conversion(
        prepared_weight,
        prepared_scale,
    )
    actual = dequantize_mxfp4_expert_weight(
        restored_weight,
        restored_scale,
        logical_shape=logical_shape,
    )

    assert prepared_weight.dtype == torch.uint8
    assert prepared_scale.dtype == torch.uint8
    assert tuple(prepared_weight.shape) == (
        packed_shape[0],
        packed_shape[2],
        packed_shape[1],
    )
    assert tuple(prepared_scale.shape) == (
        scale_shape[0],
        scale_shape[2],
        scale_shape[1],
    )
    assert torch.equal(restored_weight, packed)
    assert torch.equal(restored_scale, scales)
    torch.testing.assert_close(actual, expected)


def test_quark_mxfp4_w13_gate_up_halves_keep_order_through_layout_prep() -> None:
    gate = _uint8_pattern((4, 32), offset=3)
    up = _uint8_pattern((4, 32), offset=97)
    packed = torch.cat((gate, up), dim=0).unsqueeze(0)
    scales = torch.cat(
        (
            torch.full((4, 2), 127, dtype=torch.uint8),
            torch.full((4, 2), 128, dtype=torch.uint8),
        ),
        dim=0,
    ).unsqueeze(0)

    prepared_weight, prepared_scale = prepare_mxfp4_for_layout_conversion(
        packed,
        scales,
    )
    restored_weight, restored_scale = restore_mxfp4_from_layout_conversion(
        prepared_weight,
        prepared_scale,
    )
    dense = dequantize_mxfp4_expert_weight(
        restored_weight,
        restored_scale,
        logical_shape=(1, 8, 64),
    )
    expected_gate = dequantize_mxfp4_expert_weight(
        gate.unsqueeze(0),
        scales[:, :4],
        logical_shape=(1, 4, 64),
    )
    expected_up = dequantize_mxfp4_expert_weight(
        up.unsqueeze(0),
        scales[:, 4:],
        logical_shape=(1, 4, 64),
    )

    torch.testing.assert_close(dense[:, :4], expected_gate)
    torch.testing.assert_close(dense[:, 4:], expected_up)


def test_quark_mxfp4_dense_layout_prep_round_trips_dequantized_matrix() -> None:
    packed = _uint8_pattern((8, 32), offset=41)
    scales = _uint8_pattern((8, 2), offset=123)
    expected = _dequantize_linear(packed, scales)

    prepared_weight, prepared_scale = prepare_mxfp4_for_layout_conversion(
        packed,
        scales,
    )
    restored_weight, restored_scale = restore_mxfp4_from_layout_conversion(
        prepared_weight,
        prepared_scale,
    )
    actual = _dequantize_linear(restored_weight, restored_scale)

    assert tuple(prepared_weight.shape) == (32, 8)
    assert tuple(prepared_scale.shape) == (2, 8)
    assert torch.equal(restored_weight, packed)
    assert torch.equal(restored_scale, scales)
    torch.testing.assert_close(actual, expected)
