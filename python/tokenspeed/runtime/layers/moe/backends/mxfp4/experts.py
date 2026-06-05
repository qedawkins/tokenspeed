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

from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
    dequantize_mxfp4_activation,
    quantize_mxfp4_activation,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_E2M1_BLOCK32_FORMAT,
    PackedExpertWeightFormat,
    validate_mxfp4_expert_weight_format,
)
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton


try:
    with redirect_triton_to_tokenspeed_triton():
        import triton
        import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is None:
    _owner_rank_mxfp4_expert_gemm_kernel = None


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
    return _dequantize_mxfp4_expert_weight_unchecked(
        packed_weight,
        weight_scale,
        logical_shape=logical_shape,
        signature=signature,
    )


def _dequantize_mxfp4_expert_weight_unchecked(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    logical_shape: tuple[int, int, int],
    signature: PackedExpertWeightFormat,
) -> torch.Tensor:
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


def _dequantize_single_mxfp4_expert_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_idx: int,
    *,
    in_features: int,
    signature: PackedExpertWeightFormat,
) -> torch.Tensor:
    logical_shape = (1, int(packed_weight.shape[1]), in_features)
    dense = _dequantize_mxfp4_expert_weight_unchecked(
        packed_weight[expert_idx : expert_idx + 1],
        weight_scale[expert_idx : expert_idx + 1],
        logical_shape=logical_shape,
        signature=signature,
    )
    return dense.squeeze(0)


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
    signature = validate_mxfp4_expert_weight_format(
        local_packed_weight,
        local_weight_scale,
        logical_shape=logical_shape,
    )
    _validate_owner_gemm_component_shapes(
        owner_tokens,
        local_packed_weight,
        local_weight_scale,
        offsets,
        bias,
        out,
    )

    if out is None:
        out = torch.empty(
            (owner_tokens.shape[0], local_packed_weight.shape[1]),
            dtype=output_dtype,
            device=owner_tokens.device,
        )
    if owner_tokens.shape[0] == 0:
        return out

    if _should_use_triton_owner_gemm(owner_tokens, out):
        return _owner_rank_mxfp4_expert_gemm_triton(
            owner_tokens,
            local_packed_weight,
            local_weight_scale,
            offsets,
            bias,
            out,
        )

    for expert_idx in range(local_packed_weight.shape[0]):
        start = int(offsets[expert_idx].item())
        end = int(offsets[expert_idx + 1].item())
        if start == end:
            continue
        expert_weight = _dequantize_single_mxfp4_expert_weight(
            local_packed_weight,
            local_weight_scale,
            expert_idx,
            in_features=owner_tokens.shape[1],
            signature=signature,
        )
        result = owner_tokens[start:end].float() @ expert_weight.T
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


def owner_rank_mxfp4_down_gemm(
    intermediate: torch.Tensor,
    local_packed_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run local MXFP4 down-projection expert GEMM over sorted expert rows."""

    _validate_down_weight_shape(local_packed_weight, intermediate)
    packed_intermediate, intermediate_scale = quantize_mxfp4_activation(intermediate)
    dequant_intermediate = dequantize_mxfp4_activation(
        packed_intermediate,
        intermediate_scale,
        logical_shape=tuple(intermediate.shape),
        output_dtype=torch.float32,
    )
    return owner_rank_mxfp4_expert_gemm(
        dequant_intermediate,
        local_packed_weight,
        local_weight_scale,
        local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        bias=bias,
        output_dtype=output_dtype or intermediate.dtype,
        out=out,
    )


def local_mxfp4_down_gemm_combine(
    intermediate: torch.Tensor,
    local_packed_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    scatter_indices: torch.Tensor,
    routed_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
    local_expert_offsets: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run local MXFP4 down GEMM and reduce sorted top-k slots to tokens."""

    owner_outputs = owner_rank_mxfp4_down_gemm(
        intermediate,
        local_packed_weight,
        local_weight_scale,
        local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        bias=bias,
        output_dtype=torch.float32,
    )
    return combine_mxfp4_routed_down_outputs(
        owner_outputs,
        scatter_indices,
        routed_weights,
        num_tokens=num_tokens,
        top_k=top_k,
        output_dtype=output_dtype or intermediate.dtype,
        out=out,
    )


def combine_mxfp4_routed_down_outputs(
    sorted_outputs: torch.Tensor,
    scatter_indices: torch.Tensor,
    routed_weights: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map sorted expert rows back to top-k slots and apply routed weights."""

    _validate_combine_inputs(
        sorted_outputs,
        scatter_indices,
        routed_weights,
        num_tokens=num_tokens,
        top_k=top_k,
        out=out,
    )
    output_dtype = output_dtype or sorted_outputs.dtype
    hidden_size = sorted_outputs.shape[1]
    slot_outputs = torch.zeros(
        (num_tokens * top_k, hidden_size),
        dtype=torch.float32,
        device=sorted_outputs.device,
    )
    weighted = sorted_outputs.float() * routed_weights.reshape(-1, 1).float()
    slot_outputs.index_add_(0, scatter_indices.long(), weighted)
    reduced = slot_outputs.view(num_tokens, top_k, hidden_size).sum(dim=1)
    reduced = reduced.to(output_dtype)
    if out is None:
        return reduced
    out.copy_(reduced)
    return out


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
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
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
    capture_mode = _is_capture_mode()
    if local_expert_counts.ndim != 1:
        raise ValueError(
            "local_expert_counts must be rank-1, got "
            f"{tuple(local_expert_counts.shape)}"
        )
    if local_expert_counts.dtype != torch.int32:
        raise ValueError(
            f"local_expert_counts must be torch.int32, got {local_expert_counts.dtype}"
        )
    if not capture_mode and bool(local_expert_counts.lt(0).any().item()):
        raise ValueError("local_expert_counts must be non-negative")
    if local_expert_offsets is None:
        offsets = _offsets_from_counts(local_expert_counts)
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
        if capture_mode:
            return offsets
        if int(offsets[0].item()) != 0:
            raise ValueError("local_expert_offsets must start at zero")
        if bool((offsets[1:] - offsets[:-1]).ne(local_expert_counts).any().item()):
            raise ValueError("local_expert_offsets must match local_expert_counts")

    if not capture_mode and int(offsets[-1].item()) != num_rows:
        raise ValueError(
            f"local expert rows {int(offsets[-1].item())} != owner token rows "
            f"{num_rows}"
        )
    return offsets


def _validate_owner_gemm_component_shapes(
    owner_tokens: torch.Tensor,
    local_packed_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    offsets: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
) -> None:
    num_local_experts = int(local_packed_weight.shape[0])
    out_features = int(local_packed_weight.shape[1])
    if local_packed_weight.device != owner_tokens.device:
        raise ValueError(
            f"local packed weight device {local_packed_weight.device} != "
            f"owner token device {owner_tokens.device}"
        )
    if local_weight_scale.device != local_packed_weight.device:
        raise ValueError(
            f"local weight scale device {local_weight_scale.device} != "
            f"local packed weight device {local_packed_weight.device}"
        )
    if offsets.device != owner_tokens.device:
        raise ValueError(
            f"local_expert_offsets device {offsets.device} != owner token device "
            f"{owner_tokens.device}"
        )
    if num_local_experts != offsets.numel() - 1:
        raise ValueError(
            f"local weight experts {num_local_experts} != {offsets.numel() - 1}"
        )
    if bias is not None:
        expected_bias_shape = (num_local_experts, out_features)
        if bias.shape != expected_bias_shape:
            raise ValueError(f"bias shape {tuple(bias.shape)} != {expected_bias_shape}")
        if bias.device != owner_tokens.device:
            raise ValueError(
                f"bias device {bias.device} != owner token device {owner_tokens.device}"
            )
    if out is not None:
        expected_out_shape = (owner_tokens.shape[0], out_features)
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


def _validate_down_weight_shape(
    local_packed_weight: torch.Tensor,
    intermediate: torch.Tensor,
) -> None:
    if intermediate.ndim != 2:
        raise ValueError(
            f"intermediate must be rank-2, got {tuple(intermediate.shape)}"
        )
    if local_packed_weight.ndim != 3:
        raise ValueError(
            "local_packed_weight must be rank-3, got "
            f"{tuple(local_packed_weight.shape)}"
        )
    logical_input = (
        int(local_packed_weight.shape[2]) * MXFP4_E2M1_BLOCK32_FORMAT.pack_factor
    )
    if logical_input != intermediate.shape[1]:
        raise ValueError(
            f"local down weight input {logical_input} != intermediate hidden "
            f"{intermediate.shape[1]}"
        )


def _validate_combine_inputs(
    sorted_outputs: torch.Tensor,
    scatter_indices: torch.Tensor,
    routed_weights: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    out: torch.Tensor | None,
) -> None:
    if sorted_outputs.ndim != 2:
        raise ValueError(
            f"sorted_outputs must be rank-2, got {tuple(sorted_outputs.shape)}"
        )
    if scatter_indices.shape != (sorted_outputs.shape[0],):
        raise ValueError(
            f"scatter_indices shape {tuple(scatter_indices.shape)} != "
            f"{(sorted_outputs.shape[0],)}"
        )
    if scatter_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"scatter_indices must be int32/int64, got {scatter_indices.dtype}"
        )
    if scatter_indices.device != sorted_outputs.device:
        raise ValueError(
            "scatter_indices must be on the same device as sorted_outputs"
        )
    if routed_weights.shape != (sorted_outputs.shape[0],):
        raise ValueError(
            f"routed_weights shape {tuple(routed_weights.shape)} != "
            f"{(sorted_outputs.shape[0],)}"
        )
    if routed_weights.device != sorted_outputs.device:
        raise ValueError("routed_weights must be on the same device as sorted_outputs")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    expected_slots = num_tokens * top_k
    if sorted_outputs.shape[0] > 0:
        min_scatter = int(scatter_indices.min().item())
        max_scatter = int(scatter_indices.max().item())
        if min_scatter < 0 or max_scatter >= expected_slots:
            raise ValueError(
                "scatter_indices must be in [0, num_tokens * top_k), got "
                f"min={min_scatter} max={max_scatter} size={expected_slots}"
            )
    if out is not None:
        expected_out_shape = (num_tokens, sorted_outputs.shape[1])
        if out.shape != expected_out_shape:
            raise ValueError(f"out shape {tuple(out.shape)} != {expected_out_shape}")
        if out.device != sorted_outputs.device:
            raise ValueError(
                f"out device {out.device} != sorted_outputs device "
                f"{sorted_outputs.device}"
            )


def _offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        (counts.numel() + 1,),
        dtype=torch.int32,
        device=counts.device,
    )
    offsets[:1].zero_()
    offsets[1:].copy_(torch.cumsum(counts, dim=0, dtype=torch.int32))
    return offsets


def _is_capture_mode() -> bool:
    if get_is_capture_mode():
        return True
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _should_use_triton_owner_gemm(
    owner_tokens: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    return (
        triton is not None
        and owner_tokens.device.type == "cuda"
        and out.dtype in {torch.float16, torch.bfloat16, torch.float32}
    )


def _owner_rank_mxfp4_expert_gemm_triton(
    owner_tokens: torch.Tensor,
    local_packed_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    local_expert_offsets: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    if _owner_rank_mxfp4_expert_gemm_kernel is None:
        if _is_capture_mode():
            raise RuntimeError("MXFP4 CUDA graph expert GEMM requires Triton support")
        return out

    num_rows = owner_tokens.shape[0]
    num_local_experts = local_packed_weight.shape[0]
    out_features = local_packed_weight.shape[1]
    in_features = owner_tokens.shape[1]
    block_m = 8
    block_n = 16
    block_k = 64
    has_bias = bias is not None
    if bias is None:
        bias = out

    out.zero_()
    grid = (
        num_local_experts,
        triton.cdiv(num_rows, block_m),
        triton.cdiv(out_features, block_n),
    )
    _owner_rank_mxfp4_expert_gemm_kernel[grid](
        owner_tokens,
        local_packed_weight,
        local_weight_scale,
        bias,
        local_expert_offsets,
        out,
        owner_tokens.stride(0),
        owner_tokens.stride(1),
        local_packed_weight.stride(0),
        local_packed_weight.stride(1),
        local_packed_weight.stride(2),
        local_weight_scale.stride(0),
        local_weight_scale.stride(1),
        local_weight_scale.stride(2),
        bias.stride(0),
        bias.stride(1) if bias.ndim > 1 else 0,
        out.stride(0),
        out.stride(1),
        num_rows,
        in_features,
        out_features,
        has_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


if triton is not None:

    @triton.jit
    def _mxfp4_e2m1_values(nibbles):
        magnitude_bits = nibbles & 0x7
        exponent = (magnitude_bits >> 1).to(tl.float32)
        mantissa = (magnitude_bits & 0x1).to(tl.float32)
        normal = (1.0 + 0.5 * mantissa) * tl.exp2(exponent - 1.0)
        subnormal = 0.5 * mantissa
        magnitude = tl.where(exponent == 0.0, subnormal, normal)
        sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(tl.float32)
        return magnitude * sign

    @triton.jit
    def _owner_rank_mxfp4_expert_gemm_kernel(
        tokens_ptr,
        weight_ptr,
        scale_ptr,
        bias_ptr,
        offsets_ptr,
        out_ptr,
        TOKENS_STRIDE_M: tl.constexpr,
        TOKENS_STRIDE_K: tl.constexpr,
        WEIGHT_STRIDE_E: tl.constexpr,
        WEIGHT_STRIDE_N: tl.constexpr,
        WEIGHT_STRIDE_K: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_K: tl.constexpr,
        BIAS_STRIDE_E: tl.constexpr,
        BIAS_STRIDE_N: tl.constexpr,
        OUT_STRIDE_M: tl.constexpr,
        OUT_STRIDE_N: tl.constexpr,
        NUM_ROWS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        expert = tl.program_id(0)
        row_block = tl.program_id(1)
        col_block = tl.program_id(2)

        expert_start = tl.load(offsets_ptr + expert).to(tl.int32)
        expert_end = tl.load(offsets_ptr + expert + 1).to(tl.int32)
        rows = expert_start + row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        row_mask = (rows < expert_end) & (rows < NUM_ROWS)
        col_mask = cols < OUT_FEATURES
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, IN_FEATURES, BLOCK_K):
            ks = k_start + k_offsets
            k_mask = ks < IN_FEATURES
            tokens = tl.load(
                tokens_ptr
                + rows[:, None] * TOKENS_STRIDE_M
                + ks[None, :] * TOKENS_STRIDE_K,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            packed = tl.load(
                weight_ptr
                + expert * WEIGHT_STRIDE_E
                + cols[:, None] * WEIGHT_STRIDE_N
                + (ks[None, :] // 2) * WEIGHT_STRIDE_K,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0,
            )
            low_nibble = packed & 0xF
            high_nibble = packed >> 4
            nibbles = tl.where((ks[None, :] & 1) == 0, low_nibble, high_nibble)
            weight_values = _mxfp4_e2m1_values(nibbles)
            scale = tl.load(
                scale_ptr
                + expert * SCALE_STRIDE_E
                + cols[:, None] * SCALE_STRIDE_N
                + (ks[None, :] // 32) * SCALE_STRIDE_K,
                mask=col_mask[:, None] & k_mask[None, :],
                other=127,
            ).to(tl.float32)
            weights = weight_values * tl.exp2(scale - 127.0)
            acc += tl.dot(tokens, tl.trans(weights))

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr
                + expert * BIAS_STRIDE_E
                + cols * BIAS_STRIDE_N,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            acc += bias[None, :]

        tl.store(
            out_ptr + rows[:, None] * OUT_STRIDE_M + cols[None, :] * OUT_STRIDE_N,
            acc,
            mask=row_mask[:, None] & col_mask[None, :],
        )


__all__ = [
    "combine_mxfp4_routed_down_outputs",
    "dequantize_mxfp4_expert_weight",
    "kimi_swiglu_gate_up",
    "local_mxfp4_down_gemm_combine",
    "owner_rank_mxfp4_down_gemm",
    "owner_rank_mxfp4_expert_gemm",
    "owner_rank_mxfp4_gate_up_gemm",
]
