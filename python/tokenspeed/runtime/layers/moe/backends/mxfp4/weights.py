# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend

MXFP4_BLOCK = 32
MXFP4_LOGICAL_DTYPE = "float4_e2m1"


class PackedScaleGranularity(str, Enum):
    BLOCK = "block"
    CHANNEL = "channel"


class ExpertLocalLayout(str, Enum):
    EXPERT_OUT_IN = "expert_local_out_in"
    EXPERT_IN_OUT = "expert_local_in_out"


@dataclass(frozen=True)
class PackedExpertWeightFormat:
    name: str
    storage_dtype: torch.dtype
    logical_dtype: str
    pack_factor: int
    scale_dtype: torch.dtype
    scale_granularity: PackedScaleGranularity
    block_shape: tuple[int, int] | None
    expert_local_layout: ExpertLocalLayout
    transposed_from_logical: bool

    def expected_weight_shape(
        self,
        logical_shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        self._validate_supported()
        num_experts, out_features, in_features = _normalize_logical_shape(
            logical_shape
        )
        if in_features % self.pack_factor != 0:
            raise ValueError(
                f"logical input dim {in_features} must be divisible by "
                f"pack_factor {self.pack_factor}"
            )
        return (num_experts, out_features, in_features // self.pack_factor)

    def expected_scale_shape(
        self,
        logical_shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        self._validate_supported()
        num_experts, out_features, in_features = _normalize_logical_shape(
            logical_shape
        )
        if self.scale_granularity == PackedScaleGranularity.CHANNEL:
            return (num_experts, out_features, 1)

        assert self.block_shape is not None
        block_out, block_in = self.block_shape
        if self.scale_granularity == PackedScaleGranularity.BLOCK:
            if out_features % block_out != 0:
                raise ValueError(
                    f"logical output dim {out_features} must be divisible by "
                    f"scale block output {block_out}"
                )
            if in_features % block_in != 0:
                raise ValueError(
                    f"logical input dim {in_features} must be divisible by "
                    f"scale block input {block_in}"
                )
            return (num_experts, out_features // block_out, in_features // block_in)

        raise ValueError(f"unsupported scale granularity {self.scale_granularity!r}")

    def _validate_supported(self) -> None:
        if self.storage_dtype != torch.uint8:
            raise ValueError(
                f"unsupported packed weight storage dtype {self.storage_dtype}"
            )
        if self.pack_factor != 2:
            raise ValueError(f"unsupported pack_factor {self.pack_factor}")
        if self.expert_local_layout != ExpertLocalLayout.EXPERT_OUT_IN:
            raise ValueError(
                f"unsupported expert-local layout {self.expert_local_layout!r}"
            )
        if self.transposed_from_logical:
            raise ValueError("transposed packed expert weights are not supported")
        if self.logical_dtype == MXFP4_LOGICAL_DTYPE:
            if self.scale_dtype != torch.uint8:
                raise ValueError(f"unsupported scale dtype {self.scale_dtype}")
            if self.scale_granularity != PackedScaleGranularity.BLOCK:
                raise ValueError(
                    f"unsupported scale granularity {self.scale_granularity!r}"
                )
            if self.block_shape != (1, MXFP4_BLOCK):
                raise ValueError(f"unsupported scale block shape {self.block_shape!r}")
            return
        if self.logical_dtype == "int4":
            if self.scale_dtype != torch.float32:
                raise ValueError(f"unsupported scale dtype {self.scale_dtype}")
            if self.scale_granularity != PackedScaleGranularity.CHANNEL:
                raise ValueError(
                    f"unsupported scale granularity {self.scale_granularity!r}"
                )
            if self.block_shape is not None:
                raise ValueError(f"unsupported scale block shape {self.block_shape!r}")
            return
        raise ValueError(f"unsupported logical dtype {self.logical_dtype!r}")


MXFP4_E2M1_BLOCK32_FORMAT = PackedExpertWeightFormat(
    name="mxfp4_e2m1_block32",
    storage_dtype=torch.uint8,
    logical_dtype=MXFP4_LOGICAL_DTYPE,
    pack_factor=2,
    scale_dtype=torch.uint8,
    scale_granularity=PackedScaleGranularity.BLOCK,
    block_shape=(1, MXFP4_BLOCK),
    expert_local_layout=ExpertLocalLayout.EXPERT_OUT_IN,
    transposed_from_logical=False,
)


def validate_packed_expert_weight_format(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    logical_shape: tuple[int, int, int],
    signature: PackedExpertWeightFormat,
    tensor_name: str = "weight",
) -> PackedExpertWeightFormat:
    if scale is None:
        raise ValueError(f"{tensor_name} scale tensor is required")
    expected_weight_shape = signature.expected_weight_shape(logical_shape)
    expected_scale_shape = signature.expected_scale_shape(logical_shape)
    if weight.dtype != signature.storage_dtype:
        raise ValueError(
            f"{tensor_name} storage dtype must be {signature.storage_dtype}, "
            f"got {weight.dtype}"
        )
    if scale.dtype != signature.scale_dtype:
        raise ValueError(
            f"{tensor_name} scale dtype must be {signature.scale_dtype}, "
            f"got {scale.dtype}"
        )
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"{tensor_name} packed weight shape must be {expected_weight_shape}, "
            f"got {tuple(weight.shape)}"
        )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"{tensor_name} scale shape must be {expected_scale_shape}, "
            f"got {tuple(scale.shape)}"
        )
    return signature


def validate_mxfp4_expert_weight_format(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    logical_shape: tuple[int, int, int],
    tensor_name: str = "weight",
) -> PackedExpertWeightFormat:
    return validate_packed_expert_weight_format(
        weight,
        scale,
        logical_shape=logical_shape,
        signature=MXFP4_E2M1_BLOCK32_FORMAT,
        tensor_name=tensor_name,
    )


def prepare_mxfp4_for_layout_conversion(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose packed MXFP4 tensors for TokenSpeed layout conversion."""

    return packed_weight.transpose(-2, -1), weight_scale.transpose(-2, -1)


def restore_mxfp4_from_layout_conversion(
    prepared_weight: torch.Tensor,
    prepared_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert ``prepare_mxfp4_for_layout_conversion`` for CPU validation."""

    return prepared_weight.transpose(-2, -1), prepared_scale.transpose(-2, -1)


def _normalize_logical_shape(
    logical_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(logical_shape) != 3:
        raise ValueError(f"logical_shape must be 3D [E,out,in], got {logical_shape!r}")
    normalized = tuple(int(dim) for dim in logical_shape)
    if any(dim <= 0 for dim in normalized):
        raise ValueError(f"logical_shape dimensions must be positive, got {normalized}")
    return normalized


def create_mxfp4_weights(
    backend: MoEBackend,
    layer: nn.Module,
    num_local_experts: int,
    hidden_size_padded: int,
    ispp_padded: int,
    with_bias: bool = False,
) -> None:
    # Fused gate_up_proj (column parallel)
    w13_weight = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            2 * ispp_padded,
            hidden_size_padded // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)

    w13_weight_scale = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            2 * ispp_padded,
            hidden_size_padded // MXFP4_BLOCK,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)

    # down_proj (row parallel)
    w2_weight = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            hidden_size_padded,
            ispp_padded // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight", w2_weight)

    w2_weight_scale = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            hidden_size_padded,
            ispp_padded // MXFP4_BLOCK,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    if with_bias:
        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(num_local_experts, 2 * ispp_padded, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(num_local_experts, hidden_size_padded, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)

    # Set up weight loader (no transpose for packed uint8 mxfp4)
    weight_loader = backend._make_weight_loader()
    _set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    _set_weight_attrs(w2_weight, {"weight_loader": weight_loader})
    _set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
    _set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})
    if with_bias:
        _set_weight_attrs(w13_weight_bias, {"weight_loader": weight_loader})
        _set_weight_attrs(w2_weight_bias, {"weight_loader": weight_loader})


def _per_tensor_input_scale_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    value = loaded_weight.detach().to(torch.float32).reshape(())
    if shard_id in ("w1", "w3"):
        prev = param.data[local_expert_id]
        param.data[local_expert_id] = torch.maximum(prev, value)
    elif shard_id == "w2":
        param.data[local_expert_id] = value
    else:
        raise ValueError(f"Unknown shard_id for input_scale: {shard_id!r}")


def create_mxfp4_fp8_input_scales(layer: nn.Module, num_local_experts: int) -> None:
    w13 = nn.Parameter(
        torch.zeros(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    w2 = nn.Parameter(
        torch.zeros(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_input_scale", w13)
    layer.register_parameter("w2_input_scale", w2)
    _set_weight_attrs(w13, {"weight_loader": _per_tensor_input_scale_loader})
    _set_weight_attrs(w2, {"weight_loader": _per_tensor_input_scale_loader})


def _set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict | None,
) -> None:
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


__all__ = [
    "ExpertLocalLayout",
    "MXFP4_BLOCK",
    "MXFP4_E2M1_BLOCK32_FORMAT",
    "PackedExpertWeightFormat",
    "PackedScaleGranularity",
    "create_mxfp4_weights",
    "create_mxfp4_fp8_input_scales",
    "prepare_mxfp4_for_layout_conversion",
    "restore_mxfp4_from_layout_conversion",
    "validate_mxfp4_expert_weight_format",
    "validate_packed_expert_weight_format",
]
