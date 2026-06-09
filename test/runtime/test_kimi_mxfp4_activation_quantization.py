from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT,
    dequantize_mxfp4_activation,
    quantize_mxfp4_activation,
    quantize_mxfp4_activation_reference,
    validate_mxfp4_activation_format,
)


def test_mxfp4_activation_signature_describes_dynamic_layout() -> None:
    signature = MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT

    assert signature.name == "mxfp4_activation_e2m1_block32"
    assert signature.storage_dtype is torch.uint8
    assert signature.logical_dtype == "float4_e2m1"
    assert signature.pack_factor == 2
    assert signature.scale_dtype is torch.uint8
    assert signature.group_size == 32
    assert signature.quantization_axis == -1
    assert signature.scale_layout == "linear"
    assert signature.expected_packed_shape((5, 64)) == (5, 32)
    assert signature.expected_scale_shape((5, 64)) == (5, 2)


def test_reference_quantizes_pack_order_and_e8m0_scales() -> None:
    row = torch.zeros(32, dtype=torch.float32)
    row[:8] = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    packed, scale = quantize_mxfp4_activation_reference(row.to(torch.bfloat16))

    assert packed.dtype is torch.uint8
    assert scale.dtype is torch.uint8
    assert packed.shape == (16,)
    assert scale.shape == (1,)
    assert torch.equal(
        packed[:4],
        torch.tensor([0x10, 0x32, 0x54, 0x76], dtype=torch.uint8),
    )
    assert scale.tolist() == [127]

    actual = dequantize_mxfp4_activation(
        packed,
        scale,
        logical_shape=(32,),
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(actual[:8], row[:8])
    torch.testing.assert_close(actual[8:], torch.zeros(24))


def test_dequantize_activation_cuda_graph_safe() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for CUDA graph activation dequant smoke test")
    row = torch.zeros(32, dtype=torch.float32)
    row[:8] = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    packed, scale = quantize_mxfp4_activation_reference(row.to(torch.bfloat16))
    packed = packed.to("cuda")
    scale = scale.to("cuda")

    expected = dequantize_mxfp4_activation(
        packed,
        scale,
        logical_shape=(32,),
        output_dtype=torch.float32,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dequantize_mxfp4_activation(
            packed,
            scale,
            logical_shape=(32,),
            output_dtype=torch.float32,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected)


def test_reference_rounds_scale_up_to_power_of_two() -> None:
    activations = torch.full((1, 32), 12.0, dtype=torch.float16)

    packed, scale = quantize_mxfp4_activation_reference(activations)

    assert torch.equal(scale, torch.tensor([[128]], dtype=torch.uint8))
    actual = dequantize_mxfp4_activation(
        packed,
        scale,
        logical_shape=tuple(activations.shape),
    )
    torch.testing.assert_close(actual, activations.to(torch.float32))


def test_reference_handles_empty_rows() -> None:
    activations = torch.empty((0, 64), dtype=torch.bfloat16)

    packed, scale = quantize_mxfp4_activation_reference(activations)
    actual = dequantize_mxfp4_activation(
        packed,
        scale,
        logical_shape=tuple(activations.shape),
    )

    assert packed.shape == (0, 32)
    assert scale.shape == (0, 2)
    assert actual.shape == (0, 64)


def test_reference_handles_non_multiple_token_counts() -> None:
    torch.manual_seed(711)
    activations = torch.randn(5, 64, dtype=torch.float32).to(torch.bfloat16)

    packed, scale = quantize_mxfp4_activation_reference(activations)
    actual = dequantize_mxfp4_activation(
        packed,
        scale,
        logical_shape=tuple(activations.shape),
    )

    assert packed.shape == (5, 32)
    assert scale.shape == (5, 2)
    assert torch.isfinite(actual).all()


def test_cpu_dispatch_uses_reference_contract() -> None:
    torch.manual_seed(1776)
    activations = torch.randn(3, 64, dtype=torch.float32).to(torch.float16)

    packed, scale = quantize_mxfp4_activation(activations)
    expected_packed, expected_scale = quantize_mxfp4_activation_reference(activations)

    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


def test_activation_validation_rejects_bad_layouts() -> None:
    packed = torch.zeros(2, 32, dtype=torch.uint8)
    scale = torch.zeros(2, 2, dtype=torch.uint8)

    assert validate_mxfp4_activation_format(
        packed,
        scale,
        logical_shape=(2, 64),
    ) is MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT

    with pytest.raises(ValueError, match="input must be"):
        quantize_mxfp4_activation_reference(torch.zeros(2, 64, dtype=torch.int32))
    with pytest.raises(ValueError, match="divisible by group_size"):
        quantize_mxfp4_activation_reference(torch.zeros(2, 48, dtype=torch.float16))
    with pytest.raises(ValueError, match="packed shape"):
        validate_mxfp4_activation_format(
            packed[:, :31],
            scale,
            logical_shape=(2, 64),
        )
    with pytest.raises(ValueError, match="scale dtype"):
        validate_mxfp4_activation_format(
            packed,
            scale.to(torch.float32),
            logical_shape=(2, 64),
        )
