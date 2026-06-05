"""Regression tests for shared MXFP4 MoE weight allocation."""

# ruff: noqa: E402

import os
import sys
import unittest
from dataclasses import replace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import tokenspeed_kernel  # noqa: E402, F401
import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    ExpertLocalLayout,
    MXFP4_BLOCK,
    MXFP4_E2M1_BLOCK32_FORMAT,
    PackedScaleGranularity,
    create_mxfp4_weights,
    validate_mxfp4_expert_weight_format,
    validate_packed_expert_weight_format,
)
from tokenspeed.runtime.layers.moe.backends.weight_loaders import load_model_weight


class _Backend:
    def _make_weight_loader(self):
        def _weight_loader(*args, **kwargs):
            del args, kwargs

        return _weight_loader


class TestMxfp4Weights(unittest.TestCase):
    def test_scale_weights_store_checkpoint_bytes(self):
        layer = nn.Module()
        create_mxfp4_weights(
            _Backend(),
            layer,
            num_local_experts=2,
            hidden_size_padded=64,
            ispp_padded=96,
        )

        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.uint8)

    def test_e8m0_scale_load_preserves_checkpoint_bytes(self):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            self.skipTest("torch.float8_e8m0fnu is unavailable")

        layer = nn.Module()
        create_mxfp4_weights(
            _Backend(),
            layer,
            num_local_experts=1,
            hidden_size_padded=64,
            ispp_padded=64,
        )

        raw_w1_scale = (
            torch.tensor([120, 121], dtype=torch.uint8).repeat(64).reshape(64, 2)
        )
        load_model_weight(
            layer.w13_weight_scale,
            raw_w1_scale.view(e8m0_dtype),
            "w1",
            local_expert_id=0,
            tp_rank=0,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
        )
        self.assertTrue(torch.equal(layer.w13_weight_scale.data[0, :64], raw_w1_scale))

        raw_w2_scale = (
            torch.tensor([122, 123], dtype=torch.uint8).repeat(64).reshape(64, 2)
        )
        load_model_weight(
            layer.w2_weight_scale,
            raw_w2_scale.view(e8m0_dtype),
            "w2",
            local_expert_id=0,
            tp_rank=0,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
        )
        self.assertTrue(torch.equal(layer.w2_weight_scale.data[0], raw_w2_scale))

    def test_mxfp4_format_signature_describes_encoding(self):
        signature = MXFP4_E2M1_BLOCK32_FORMAT

        self.assertEqual(signature.name, "mxfp4_e2m1_block32")
        self.assertEqual(signature.storage_dtype, torch.uint8)
        self.assertEqual(signature.logical_dtype, "float4_e2m1")
        self.assertEqual(signature.pack_factor, 2)
        self.assertEqual(signature.scale_dtype, torch.uint8)
        self.assertEqual(signature.scale_granularity, PackedScaleGranularity.BLOCK)
        self.assertEqual(signature.block_shape, (1, MXFP4_BLOCK))
        self.assertEqual(signature.expert_local_layout, ExpertLocalLayout.EXPERT_OUT_IN)
        self.assertFalse(signature.transposed_from_logical)

    def test_mxfp4_created_gate_up_and_down_layouts_validate(self):
        layer = nn.Module()
        create_mxfp4_weights(
            _Backend(),
            layer,
            num_local_experts=2,
            hidden_size_padded=64,
            ispp_padded=96,
        )

        self.assertIs(
            validate_mxfp4_expert_weight_format(
                layer.w13_weight,
                layer.w13_weight_scale,
                logical_shape=(2, 192, 64),
                tensor_name="w13_weight",
            ),
            MXFP4_E2M1_BLOCK32_FORMAT,
        )
        self.assertIs(
            validate_mxfp4_expert_weight_format(
                layer.w2_weight,
                layer.w2_weight_scale,
                logical_shape=(2, 64, 96),
                tensor_name="w2_weight",
            ),
            MXFP4_E2M1_BLOCK32_FORMAT,
        )

        with self.assertRaisesRegex(ValueError, "scale shape"):
            validate_mxfp4_expert_weight_format(
                layer.w13_weight,
                layer.w13_weight_scale[:, :, :1],
                logical_shape=(2, 192, 64),
                tensor_name="w13_weight",
            )

    def test_mxfp4_format_validation_rejects_invalid_combinations(self):
        weight = torch.zeros(1, 2, 16, dtype=torch.uint8)
        scale = torch.zeros(1, 2, 1, dtype=torch.uint8)
        logical_shape = (1, 2, 32)

        with self.assertRaisesRegex(ValueError, "scale tensor is required"):
            validate_mxfp4_expert_weight_format(
                weight,
                None,
                logical_shape=logical_shape,
            )
        with self.assertRaisesRegex(ValueError, "scale dtype"):
            validate_mxfp4_expert_weight_format(
                weight,
                scale.to(torch.float32),
                logical_shape=logical_shape,
            )
        with self.assertRaisesRegex(ValueError, "storage dtype"):
            validate_mxfp4_expert_weight_format(
                weight.to(torch.int16),
                scale,
                logical_shape=logical_shape,
            )
        with self.assertRaisesRegex(ValueError, "packed weight shape"):
            validate_mxfp4_expert_weight_format(
                weight[:, :, :15],
                scale,
                logical_shape=logical_shape,
            )
        with self.assertRaisesRegex(ValueError, "logical input dim"):
            validate_mxfp4_expert_weight_format(
                weight,
                scale,
                logical_shape=(1, 2, 33),
            )
        with self.assertRaisesRegex(ValueError, "unsupported pack_factor"):
            validate_packed_expert_weight_format(
                weight,
                scale,
                logical_shape=logical_shape,
                signature=replace(MXFP4_E2M1_BLOCK32_FORMAT, pack_factor=4),
            )
        with self.assertRaisesRegex(ValueError, "unsupported scale block shape"):
            validate_packed_expert_weight_format(
                weight,
                scale,
                logical_shape=logical_shape,
                signature=replace(MXFP4_E2M1_BLOCK32_FORMAT, block_shape=(2, 32)),
            )
        with self.assertRaisesRegex(ValueError, "unsupported scale granularity"):
            validate_packed_expert_weight_format(
                weight,
                scale,
                logical_shape=logical_shape,
                signature=replace(
                    MXFP4_E2M1_BLOCK32_FORMAT,
                    scale_granularity="tensor",
                ),
            )
        with self.assertRaisesRegex(ValueError, "unsupported expert-local layout"):
            validate_packed_expert_weight_format(
                weight,
                scale,
                logical_shape=logical_shape,
                signature=replace(
                    MXFP4_E2M1_BLOCK32_FORMAT,
                    expert_local_layout=ExpertLocalLayout.EXPERT_IN_OUT,
                ),
            )
        with self.assertRaisesRegex(ValueError, "transposed"):
            validate_packed_expert_weight_format(
                weight,
                scale,
                logical_shape=logical_shape,
                signature=replace(
                    MXFP4_E2M1_BLOCK32_FORMAT,
                    transposed_from_logical=True,
                ),
            )
        with self.assertRaisesRegex(ValueError, "unsupported logical dtype"):
            validate_packed_expert_weight_format(
                weight,
                scale,
                logical_shape=logical_shape,
                signature=replace(
                    MXFP4_E2M1_BLOCK32_FORMAT,
                    logical_dtype="float4_e3m0",
                ),
            )

    def test_mxfp4_reference_dequantizes_pack_order_and_scale_blocks(self):
        row = torch.tensor(
            [0x21, 0x43, 0x65, 0x17] + [0] * 12,
            dtype=torch.uint8,
        )
        weight = torch.cat((row, row)).reshape(1, 1, 32)
        scale = torch.tensor([[[127, 128]]], dtype=torch.uint8)

        actual = _dequantize_mxfp4_expert_weight_reference(
            weight,
            scale,
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


def _dequantize_mxfp4_expert_weight_reference(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    logical_shape: tuple[int, int, int],
) -> torch.Tensor:
    signature = validate_mxfp4_expert_weight_format(
        weight,
        scale,
        logical_shape=logical_shape,
    )
    in_features = logical_shape[2]
    values = weight.new_empty(logical_shape, dtype=torch.float32)
    packed = weight.reshape(*weight.shape[:-1], in_features // signature.pack_factor)
    values[..., 0::2] = _e2m1_values(packed & 0xF)
    values[..., 1::2] = _e2m1_values(packed >> 4)

    _, block_in = signature.block_shape
    scales = torch.pow(2.0, scale.to(torch.int32) - 127).to(torch.float32)
    expanded_scales = scales.repeat_interleave(block_in, dim=2)
    return values * expanded_scales


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


if __name__ == "__main__":
    unittest.main()
