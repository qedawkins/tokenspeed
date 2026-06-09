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
from typing import Literal

import torch

from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_BLOCK,
    MXFP4_LOGICAL_DTYPE,
)

MXFP4_ACTIVATION_SCALE_LAYOUT = "linear"
MXFP4_E2M1_MAX = 6.0


@dataclass(frozen=True)
class Mxfp4ActivationFormat:
    name: str
    storage_dtype: torch.dtype
    logical_dtype: str
    pack_factor: int
    scale_dtype: torch.dtype
    group_size: int
    quantization_axis: int
    scale_layout: str

    def expected_packed_shape(self, logical_shape: tuple[int, ...]) -> tuple[int, ...]:
        logical_shape = _normalize_logical_shape(logical_shape, self.group_size)
        return (*logical_shape[:-1], logical_shape[-1] // self.pack_factor)

    def expected_scale_shape(self, logical_shape: tuple[int, ...]) -> tuple[int, ...]:
        logical_shape = _normalize_logical_shape(logical_shape, self.group_size)
        return (*logical_shape[:-1], logical_shape[-1] // self.group_size)


MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT = Mxfp4ActivationFormat(
    name="mxfp4_activation_e2m1_block32",
    storage_dtype=torch.uint8,
    logical_dtype=MXFP4_LOGICAL_DTYPE,
    pack_factor=2,
    scale_dtype=torch.uint8,
    group_size=MXFP4_BLOCK,
    quantization_axis=-1,
    scale_layout=MXFP4_ACTIVATION_SCALE_LAYOUT,
)


def quantize_mxfp4_activation(
    activations: torch.Tensor,
    *,
    group_size: int = MXFP4_BLOCK,
    scale_layout: Literal["linear"] = MXFP4_ACTIVATION_SCALE_LAYOUT,
    solution: str | None = "triton",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize BF16/FP16 activations to packed MXFP4.

    CPU tensors use the torch reference implementation so shape/format tests do
    not depend on GPU visibility. Device tensors dispatch through
    ``tokenspeed_kernel.quantize_mxfp4`` using the same contract.
    """

    _validate_activation_input(activations, group_size)
    if scale_layout != MXFP4_ACTIVATION_SCALE_LAYOUT:
        raise ValueError(
            f"unsupported MXFP4 activation scale_layout {scale_layout!r}; "
            f"expected {MXFP4_ACTIVATION_SCALE_LAYOUT!r}"
        )
    if not activations.is_cuda:
        return quantize_mxfp4_activation_reference(
            activations,
            group_size=group_size,
        )

    import tokenspeed_kernel

    packed, scale = tokenspeed_kernel.quantize_mxfp4(
        activations.contiguous(),
        scale_size=group_size,
        scale_layout=scale_layout,
        solution=solution,
        enable_pdl=enable_pdl,
    )
    validate_mxfp4_activation_format(
        packed,
        scale,
        logical_shape=tuple(activations.shape),
        group_size=group_size,
    )
    return packed, scale


def quantize_mxfp4_activation_reference(
    activations: torch.Tensor,
    *,
    group_size: int = MXFP4_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for dynamic activation MXFP4 quantization."""

    _validate_activation_input(activations, group_size)
    logical_shape = tuple(activations.shape)
    last_dim = logical_shape[-1]
    num_groups = last_dim // group_size
    src = activations.to(torch.float32).contiguous()

    grouped = src.reshape(-1, num_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale_bits, dequant_scale = _round_up_e8m0_scale(max_abs / MXFP4_E2M1_MAX)
    quant_scale = torch.where(
        dequant_scale == 0,
        torch.zeros((), dtype=torch.float32, device=src.device),
        1.0 / dequant_scale,
    )
    quantized = (grouped * quant_scale).reshape(logical_shape)
    nibbles = _float32_to_e2m1_nibbles(quantized)

    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    scale = (scale_bits.squeeze(-1) >> 23).to(torch.uint8)
    scale = scale.reshape(*logical_shape[:-1], num_groups)
    return packed, scale


def dequantize_mxfp4_activation(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    logical_shape: tuple[int, ...],
    output_dtype: torch.dtype = torch.float32,
    group_size: int = MXFP4_BLOCK,
) -> torch.Tensor:
    validate_mxfp4_activation_format(
        packed,
        scale,
        logical_shape=logical_shape,
        group_size=group_size,
    )
    values = packed.new_empty(logical_shape, dtype=torch.float32)
    values[..., 0::2] = _e2m1_values(packed & 0xF)
    values[..., 1::2] = _e2m1_values(packed >> 4)
    scales = _e8m0_to_float32(scale).repeat_interleave(group_size, dim=-1)
    return (values * scales).to(output_dtype)


def validate_mxfp4_activation_format(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    logical_shape: tuple[int, ...],
    group_size: int = MXFP4_BLOCK,
) -> Mxfp4ActivationFormat:
    signature = _activation_signature(group_size)
    expected_packed_shape = signature.expected_packed_shape(logical_shape)
    expected_scale_shape = signature.expected_scale_shape(logical_shape)
    if packed.dtype != signature.storage_dtype:
        raise ValueError(
            f"MXFP4 activation packed dtype must be {signature.storage_dtype}, "
            f"got {packed.dtype}"
        )
    if scale.dtype != signature.scale_dtype:
        raise ValueError(
            f"MXFP4 activation scale dtype must be {signature.scale_dtype}, "
            f"got {scale.dtype}"
        )
    if tuple(packed.shape) != expected_packed_shape:
        raise ValueError(
            "MXFP4 activation packed shape must be "
            f"{expected_packed_shape}, got {tuple(packed.shape)}"
        )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "MXFP4 activation scale shape must be "
            f"{expected_scale_shape}, got {tuple(scale.shape)}"
        )
    if packed.device != scale.device:
        raise ValueError(
            f"MXFP4 activation packed device {packed.device} != scale device "
            f"{scale.device}"
        )
    return signature


def _activation_signature(group_size: int) -> Mxfp4ActivationFormat:
    if group_size != MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT.group_size:
        raise ValueError(
            f"unsupported MXFP4 activation group_size {group_size}; expected "
            f"{MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT.group_size}"
        )
    return MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT


def _validate_activation_input(activations: torch.Tensor, group_size: int) -> None:
    _activation_signature(group_size)
    if activations.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "MXFP4 activation input must be torch.bfloat16, torch.float16, "
            f"or torch.float32, got {activations.dtype}"
        )
    _normalize_logical_shape(tuple(activations.shape), group_size)


def _normalize_logical_shape(
    logical_shape: tuple[int, ...],
    group_size: int,
) -> tuple[int, ...]:
    if len(logical_shape) < 1:
        raise ValueError("MXFP4 activation logical_shape must have at least one dim")
    normalized = tuple(int(dim) for dim in logical_shape)
    if any(dim < 0 for dim in normalized):
        raise ValueError(
            f"MXFP4 activation logical_shape dimensions must be non-negative, "
            f"got {normalized}"
        )
    last_dim = normalized[-1]
    if last_dim <= 0:
        raise ValueError(
            f"MXFP4 activation quantization dim must be positive, got {last_dim}"
        )
    if last_dim % 2 != 0:
        raise ValueError(
            f"MXFP4 activation quantization dim {last_dim} must be even"
        )
    if last_dim % group_size != 0:
        raise ValueError(
            f"MXFP4 activation quantization dim {last_dim} must be divisible by "
            f"group_size {group_size}"
        )
    return normalized


def _round_up_e8m0_scale(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale_bits = scale.contiguous().view(torch.int32)
    scale_bits = (scale_bits + 0x007FFFFF) & 0x7F800000
    dequant_scale = scale_bits.view(torch.float32)
    return scale_bits, dequant_scale


def _float32_to_e2m1_nibbles(values: torch.Tensor) -> torch.Tensor:
    values = values.contiguous().to(torch.float32)
    q_int = values.view(torch.int32)
    signs = q_int & 0x80000000
    exponents = _right_shift_unsigned(q_int, 23) & 0xFF
    mantissas_orig = q_int & 0x7FFFFF

    e8_bias = 127
    e2_bias = 1
    is_subnormal = exponents < e8_bias
    shift = e8_bias - exponents - 1
    mantissas_pre = 0x400000 | _right_shift_unsigned(mantissas_orig, 1)
    bit0_dropped = (mantissas_orig & 0x1) != 0
    mask = (1 << shift.clamp(max=31)) - 1
    dropped_post = (mantissas_pre & mask) != 0
    sticky = is_subnormal & (bit0_dropped | dropped_post)
    mantissas = torch.where(is_subnormal, mantissas_pre >> shift, mantissas_orig)
    exponents = (
        torch.maximum(
            exponents,
            torch.tensor(e8_bias - e2_bias, device=values.device),
        )
        - (e8_bias - e2_bias)
    )

    m2bits = _right_shift_unsigned(mantissas, 21) & 0x3
    lsb_keep = _right_shift_unsigned(m2bits, 1) & 0x1
    guard = m2bits & 0x1
    sticky |= (mantissas & ((1 << 21) - 1)) != 0
    round_inc = guard & (sticky.to(torch.int32) | lsb_keep)
    e2m1_tmp = _right_shift_unsigned(((exponents << 2) | m2bits) + round_inc, 1)
    e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=values.device))
    return (_right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)


def _right_shift_unsigned(x: torch.Tensor, shift: int | torch.Tensor) -> torch.Tensor:
    if isinstance(shift, int):
        return (x >> shift) & ((1 << (32 - shift)) - 1)
    return (x >> shift) & ((1 << (32 - shift.clamp(max=31))) - 1)


def _e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    return (scale.to(torch.int32) << 23).contiguous().view(torch.float32)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    return magnitude * sign


__all__ = [
    "MXFP4_ACTIVATION_SCALE_LAYOUT",
    "MXFP4_E2M1_BLOCK32_ACTIVATION_FORMAT",
    "Mxfp4ActivationFormat",
    "dequantize_mxfp4_activation",
    "quantize_mxfp4_activation",
    "quantize_mxfp4_activation_reference",
    "validate_mxfp4_activation_format",
]
