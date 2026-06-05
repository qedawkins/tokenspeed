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

"""Packed MXFP4 expert math components.

These helpers implement the tensor-encoding math for the existing packed
``moe.experts``/EP owner-buffer contracts. They intentionally do not register a
backend, choose kernels, or expose vendor-specific packed variants.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    dequantize_mxfp4_activation,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_E2M1_BLOCK32_FORMAT,
    validate_mxfp4_expert_weight_format,
)


def dequantize_mxfp4_expert_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    logical_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return dense FP32 expert weights from packed MXFP4 storage."""

    if logical_shape is None:
        logical_shape = (
            int(packed_weight.shape[0]),
            int(packed_weight.shape[1]),
            int(packed_weight.shape[2]) * MXFP4_E2M1_BLOCK32_FORMAT.pack_factor,
        )
    signature = validate_mxfp4_expert_weight_format(
        packed_weight,
        weight_scale,
        logical_shape=logical_shape,
    )
    _, _, in_features = logical_shape
    values = packed_weight.new_empty(logical_shape, dtype=torch.float32)
    packed = packed_weight.reshape(
        *packed_weight.shape[:-1],
        in_features // signature.pack_factor,
    )
    values[..., 0::2] = _e2m1_values(packed & 0xF)
    values[..., 1::2] = _e2m1_values(packed >> 4)

    _, block_in = signature.block_shape
    scales = torch.pow(2.0, weight_scale.to(torch.int32) - 127).to(torch.float32)
    return values * scales.repeat_interleave(block_in, dim=2)


def owner_rank_mxfp4_expert_gemm(
    owner_tokens: torch.Tensor,
    local_packed_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run packed MXFP4 expert GEMM for owner-local ragged expert slices.

    ``owner_tokens`` is arranged as contiguous expert slices described by
    ``local_expert_counts`` and optional offsets. The packed weight tensors use
    the S7 MXFP4 signature: logical ``[E_local, out, in]`` with two E2M1 values
    per uint8 along the innermost dimension and uint8 E8M0 block scales.
    """

    _validate_owner_tokens(owner_tokens)
    offsets = _validate_counts_and_offsets(
        local_expert_counts,
        owner_tokens.shape[0],
        local_expert_offsets=local_expert_offsets,
    )
    output_dtype = output_dtype or owner_tokens.dtype
    logical_shape = (
        int(local_packed_weight.shape[0]),
        int(local_packed_weight.shape[1]),
        int(owner_tokens.shape[1]),
    )
    dense_weight = dequantize_mxfp4_expert_weight(
        local_packed_weight,
        local_weight_scale,
        logical_shape=logical_shape,
    )
    _validate_component_shapes(owner_tokens, dense_weight, offsets, bias, out)

    if out is None:
        out = torch.empty(
            (owner_tokens.shape[0], dense_weight.shape[1]),
            dtype=output_dtype,
            device=owner_tokens.device,
        )
    if owner_tokens.shape[0] == 0:
        return out

    for expert_idx in range(dense_weight.shape[0]):
        start = int(offsets[expert_idx].item())
        end = int(offsets[expert_idx + 1].item())
        if start == end:
            continue
        result = owner_tokens[start:end].float() @ dense_weight[expert_idx].T
        if bias is not None:
            result = result + bias[expert_idx].float()
        out[start:end].copy_(result.to(out.dtype))
    return out


def owner_rank_mxfp4_gate_up_gemm(
    packed_owner_tokens: torch.Tensor,
    owner_token_scale: torch.Tensor,
    local_packed_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float | None = 7.0,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run local MXFP4 gate/up expert GEMM and Kimi SwiGLU activation.

    ``packed_owner_tokens`` is the dynamic MXFP4 activation format produced by
    the local activation quantizer: packed E2M1 values with one uint8 E8M0 scale
    per 32 input elements. Weight tensors use the fused ``w13`` local expert
    layout ``[E_local, 2 * intermediate, hidden / 2]``.
    """

    hidden_size = _validate_gate_up_weight_shape(local_packed_weight)
    owner_tokens = dequantize_mxfp4_activation(
        packed_owner_tokens,
        owner_token_scale,
        logical_shape=(*packed_owner_tokens.shape[:-1], hidden_size),
        output_dtype=torch.float32,
    )
    gate_up = owner_rank_mxfp4_expert_gemm(
        owner_tokens,
        local_packed_weight,
        local_weight_scale,
        local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        bias=bias,
        output_dtype=torch.float32,
    )
    return kimi_swiglu_gate_up(
        gate_up,
        alpha=swiglu_alpha,
        limit=swiglu_limit,
        output_dtype=output_dtype,
        out=out,
    )


def kimi_swiglu_gate_up(
    gate_up: torch.Tensor,
    *,
    alpha: float = 1.702,
    limit: float | None = 7.0,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the interleaved gate/up SwiGLU used by Kimi packed kernels."""

    if gate_up.ndim < 1:
        raise ValueError("gate_up must have at least one dimension")
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"gate/up output dim must be even, got {gate_up.shape[-1]}"
        )
    output_dtype = output_dtype or gate_up.dtype
    expected_out_shape = (*gate_up.shape[:-1], gate_up.shape[-1] // 2)
    if out is not None:
        if out.shape != expected_out_shape:
            raise ValueError(f"out shape {tuple(out.shape)} != {expected_out_shape}")
        if out.device != gate_up.device:
            raise ValueError(
                f"out device {out.device} != gate_up device {gate_up.device}"
            )
        if out.dtype != output_dtype:
            raise ValueError(f"out dtype {out.dtype} != {output_dtype}")

    gate = gate_up[..., 0::2].float()
    up = gate_up[..., 1::2].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    activated = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
    activated = activated.to(output_dtype)
    if out is None:
        return activated
    out.copy_(activated)
    return out


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


def _validate_owner_tokens(owner_tokens: torch.Tensor) -> None:
    if owner_tokens.ndim != 2:
        raise ValueError(
            f"owner_tokens must be rank-2, got {tuple(owner_tokens.shape)}"
        )


def _validate_counts_and_offsets(
    local_expert_counts: torch.Tensor,
    num_rows: int,
    *,
    local_expert_offsets: torch.Tensor | None,
) -> torch.Tensor:
    if local_expert_counts.ndim != 1:
        raise ValueError(
            "local_expert_counts must be rank-1, got "
            f"{tuple(local_expert_counts.shape)}"
        )
    if local_expert_counts.dtype != torch.int32:
        raise ValueError(
            f"local_expert_counts must be torch.int32, got {local_expert_counts.dtype}"
        )
    if bool(local_expert_counts.lt(0).any().item()):
        raise ValueError("local_expert_counts must be non-negative")
    if local_expert_offsets is None:
        offsets = torch.empty(
            (local_expert_counts.numel() + 1,),
            dtype=torch.int32,
            device=local_expert_counts.device,
        )
        offsets[0] = 0
        offsets[1:] = torch.cumsum(local_expert_counts, dim=0, dtype=torch.int32)
    else:
        offsets = local_expert_offsets
        if offsets.shape != (local_expert_counts.numel() + 1,):
            raise ValueError(
                f"local_expert_offsets shape {tuple(offsets.shape)} != "
                f"{(local_expert_counts.numel() + 1,)}"
            )
        if offsets.dtype != torch.int32:
            raise ValueError(
                f"local_expert_offsets must be torch.int32, got {offsets.dtype}"
            )
        if offsets.device != local_expert_counts.device:
            raise ValueError(
                "local_expert_offsets must be on the same device as "
                "local_expert_counts"
            )
        if int(offsets[0].item()) != 0:
            raise ValueError("local_expert_offsets must start at zero")
        if bool((offsets[1:] - offsets[:-1]).ne(local_expert_counts).any().item()):
            raise ValueError("local_expert_offsets must match local_expert_counts")

    if int(offsets[-1].item()) != num_rows:
        raise ValueError(
            f"local expert rows {int(offsets[-1].item())} != owner token rows "
            f"{num_rows}"
        )
    return offsets


def _validate_component_shapes(
    owner_tokens: torch.Tensor,
    dense_weight: torch.Tensor,
    offsets: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
) -> None:
    if dense_weight.device != owner_tokens.device:
        raise ValueError(
            f"dense weight device {dense_weight.device} != owner token device "
            f"{owner_tokens.device}"
        )
    if offsets.device != owner_tokens.device:
        raise ValueError(
            f"local_expert_offsets device {offsets.device} != owner token device "
            f"{owner_tokens.device}"
        )
    if dense_weight.shape[0] != offsets.numel() - 1:
        raise ValueError(
            f"local weight experts {dense_weight.shape[0]} != "
            f"{offsets.numel() - 1}"
        )
    if dense_weight.shape[2] != owner_tokens.shape[1]:
        raise ValueError(
            f"local weight input {dense_weight.shape[2]} != owner token hidden "
            f"{owner_tokens.shape[1]}"
        )
    if bias is not None:
        expected_bias_shape = (dense_weight.shape[0], dense_weight.shape[1])
        if bias.shape != expected_bias_shape:
            raise ValueError(f"bias shape {tuple(bias.shape)} != {expected_bias_shape}")
        if bias.device != owner_tokens.device:
            raise ValueError(
                f"bias device {bias.device} != owner token device {owner_tokens.device}"
            )
    if out is not None:
        expected_out_shape = (owner_tokens.shape[0], dense_weight.shape[1])
        if out.shape != expected_out_shape:
            raise ValueError(f"out shape {tuple(out.shape)} != {expected_out_shape}")
        if out.device != owner_tokens.device:
            raise ValueError(
                f"out device {out.device} != owner token device {owner_tokens.device}"
            )


def _validate_gate_up_weight_shape(local_packed_weight: torch.Tensor) -> int:
    if local_packed_weight.ndim != 3:
        raise ValueError(
            "local_packed_weight must be rank-3, got "
            f"{tuple(local_packed_weight.shape)}"
        )
    if local_packed_weight.shape[1] % 2 != 0:
        raise ValueError(
            f"gate/up output dim must be even, got {local_packed_weight.shape[1]}"
        )
    return int(local_packed_weight.shape[2]) * MXFP4_E2M1_BLOCK32_FORMAT.pack_factor


__all__ = [
    "dequantize_mxfp4_expert_weight",
    "kimi_swiglu_gate_up",
    "owner_rank_mxfp4_expert_gemm",
    "owner_rank_mxfp4_gate_up_gemm",
]
