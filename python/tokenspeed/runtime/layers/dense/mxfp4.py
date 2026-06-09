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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MXFP4 dense linear bridge for checkpoint-serialized Kimi MLP weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MXFP4_BLOCK = 32


def dequantize_mxfp4_linear_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return dense [out, in] weights from packed MXFP4 linear storage."""

    dense = _dequantize_mxfp4_block32_weight(packed_weight, weight_scale)
    if output_dtype is not None:
        dense = dense.to(output_dtype)
    return dense


def _dequantize_mxfp4_block32_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if packed_weight.ndim != 2:
        raise ValueError(f"packed_weight must be rank-2, got {packed_weight.shape}")
    if weight_scale.ndim != 2:
        raise ValueError(f"weight_scale must be rank-2, got {weight_scale.shape}")
    if packed_weight.dtype != torch.uint8:
        raise ValueError(f"packed_weight must be uint8, got {packed_weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"weight_scale must be uint8, got {weight_scale.dtype}")

    out_features, packed_in_features = packed_weight.shape
    in_features = packed_in_features * 2
    expected_scale_shape = (out_features, in_features // MXFP4_BLOCK)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale shape must be {expected_scale_shape}, "
            f"got {tuple(weight_scale.shape)}"
        )

    values = packed_weight.new_empty((out_features, in_features), dtype=torch.float32)
    values[..., 0::2] = _e2m1_values(packed_weight & 0xF)
    values[..., 1::2] = _e2m1_values(packed_weight >> 4)
    scales = torch.pow(2.0, weight_scale.to(torch.int32) - 127).to(torch.float32)
    return values * scales.repeat_interleave(MXFP4_BLOCK, dim=-1)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    return magnitude * sign


class Mxfp4LinearMethod:
    """Reference dense linear path for checkpoint-serialized MXFP4 weights.

    Kimi-K2.5 MXFP4 stores dense layer-0 MLP and MoE shared-expert MLP tensors
    in the same packed FP4/e8m0 format as routed experts. This method preserves
    the packed checkpoint loading contract, then dequantizes to a normal dense
    floating-point weight in ``process_weights_after_loading`` so the existing
    linear forward path remains correct until optimized dense MXFP4 kernels are
    wired in.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        _validate_mxfp4_partition(input_size_per_partition)
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        scale_loader = _wrap_e8m0_scale_loader(weight_loader)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype
        layer.orig_dtype = params_dtype

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        weight.output_dim = 0
        weight.input_dim = 1
        if weight_loader is not None:
            weight.weight_loader = weight_loader
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP4_BLOCK,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        weight_scale.output_dim = 0
        weight_scale.input_dim = 1
        if scale_loader is not None:
            weight_scale.weight_loader = scale_loader
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_mxfp4_dense_dequantized", False):
            return
        if layer.weight.dtype != torch.uint8:
            layer._mxfp4_dense_dequantized = True
            return
        dense_weight = dequantize_mxfp4_linear_weight(
            layer.weight.data,
            layer.weight_scale.data,
            output_dtype=layer.params_dtype,
        )
        layer.weight = Parameter(dense_weight.contiguous(), requires_grad=False)
        layer.register_parameter("weight_scale", None)
        layer._mxfp4_dense_dequantized = True

    def apply(self, layer, x, bias=None):
        if not getattr(layer, "_mxfp4_dense_dequantized", False):
            self.process_weights_after_loading(layer)
        return F.linear(x, layer.weight, bias)


def _validate_mxfp4_partition(input_size_per_partition: int) -> None:
    if input_size_per_partition % 2 != 0:
        raise ValueError(
            f"MXFP4 input partition {input_size_per_partition} must be divisible by 2"
        )
    if input_size_per_partition % MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP4 input partition {input_size_per_partition} must be divisible by "
            f"{MXFP4_BLOCK}"
        )


def _wrap_e8m0_scale_loader(weight_loader):
    if weight_loader is None:
        return None

    def scale_loader(param, loaded_weight, *args, **kwargs):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if (
            e8m0_dtype is not None
            and param.dtype == torch.uint8
            and loaded_weight.dtype == e8m0_dtype
        ):
            loaded_weight = loaded_weight.view(torch.uint8)
        return weight_loader(param, loaded_weight, *args, **kwargs)

    return scale_loader


__all__ = [
    "Mxfp4LinearMethod",
    "dequantize_mxfp4_linear_weight",
]
