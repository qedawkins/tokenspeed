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

"""Local sorted-dispatch MoE expert GEMM Gluon kernels for AMD GFX950."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from tokenspeed_kernel._triton import gl, gluon, tl
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)


_FP8_E4M3_DTYPES = tuple(
    dtype
    for name in ("float8_e4m3fn", "float8_e4m3fnuz")
    if (dtype := getattr(torch, name, None)) is not None
)
_FP8_CHANNEL_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="channel",
)
_DENSE_EXPERT_SIGNATURES = format_signatures(
    "x", "dense", {torch.float16, torch.bfloat16}
)
_W8A8_EXPERT_SIGNATURES = frozenset(
    format_signature(
        x=dense_tensor_format(activation_dtype),
        weight=tensor_format(
            "scaled-fp8",
            weight_dtype,
            scale=_FP8_CHANNEL_SCALE,
        ),
    )
    for activation_dtype in (torch.float16, torch.bfloat16)
    for weight_dtype in _FP8_E4M3_DTYPES
)
_EXPERT_SIGNATURES = _DENSE_EXPERT_SIGNATURES | _W8A8_EXPERT_SIGNATURES


@gluon.jit
def _dense_sorted_expert_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    A_STRIDE_M: gl.constexpr,
    A_STRIDE_K: gl.constexpr,
    B_STRIDE_E: gl.constexpr,
    B_STRIDE_N: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    C_STRIDE_M: gl.constexpr,
    C_STRIDE_N: gl.constexpr,
    NUM_VALID_SLOTS: gl.constexpr,
    INPUT_K: gl.constexpr,
    OUTPUT_N: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_ELEMS_PER_THREAD: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    C_SORTED: gl.constexpr,
):
    sorted_row = gl.program_id(0)
    out_n = gl.program_id(1)
    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr).to(gl.int32)
    row_active = sorted_row < num_tokens_post_padded

    sorted_slot = gl.load(
        sorted_token_ids_ptr + sorted_row,
        mask=row_active,
        other=NUM_VALID_SLOTS,
    ).to(gl.int32)
    slot_valid = sorted_slot < NUM_VALID_SLOTS
    expert_id = gl.load(
        expert_ids_ptr + sorted_row // BLOCK_SIZE_M,
        mask=row_active,
        other=-1,
    ).to(gl.int32)
    expert_valid = expert_id >= 0
    source_row = sorted_slot // TOP_K

    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_K_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offs_k = gl.arange(0, BLOCK_K, layout=layout)
    k_valid = offs_k < INPUT_K
    load_valid = row_active & slot_valid & expert_valid & k_valid
    a = gl.load(
        A_ptr + source_row * A_STRIDE_M + offs_k * A_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    weight = gl.load(
        B_ptr + expert_id * B_STRIDE_E + out_n * B_STRIDE_N + offs_k * B_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    acc = gl.sum(a * weight, axis=0)

    if MUL_ROUTED_WEIGHT:
        routed_weight = gl.load(
            topk_weights_ptr + sorted_slot,
            mask=row_active & slot_valid,
            other=0.0,
        ).to(gl.float32)
        acc *= routed_weight

    if C_SORTED:
        c_row = sorted_row
        store_valid = row_active
    else:
        c_row = sorted_slot
        store_valid = row_active & slot_valid
    gl.store(
        C_ptr + c_row * C_STRIDE_M + out_n * C_STRIDE_N,
        acc.to(C_ptr.dtype.element_ty),
        mask=store_valid,
    )


@gluon.jit
def _fp8_sorted_expert_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    A_STRIDE_M: gl.constexpr,
    A_STRIDE_K: gl.constexpr,
    B_STRIDE_E: gl.constexpr,
    B_STRIDE_N: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    C_STRIDE_M: gl.constexpr,
    C_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_E: gl.constexpr,
    B_SCALE_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_K: gl.constexpr,
    NUM_VALID_SLOTS: gl.constexpr,
    INPUT_K: gl.constexpr,
    OUTPUT_N: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    WEIGHT_BLOCK_N: gl.constexpr,
    WEIGHT_BLOCK_K: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_ELEMS_PER_THREAD: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    C_SORTED: gl.constexpr,
):
    sorted_row = gl.program_id(0)
    out_n = gl.program_id(1)
    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr).to(gl.int32)
    row_active = sorted_row < num_tokens_post_padded

    sorted_slot = gl.load(
        sorted_token_ids_ptr + sorted_row,
        mask=row_active,
        other=NUM_VALID_SLOTS,
    ).to(gl.int32)
    slot_valid = sorted_slot < NUM_VALID_SLOTS
    expert_id = gl.load(
        expert_ids_ptr + sorted_row // BLOCK_SIZE_M,
        mask=row_active,
        other=-1,
    ).to(gl.int32)
    expert_valid = expert_id >= 0
    source_row = sorted_slot // TOP_K

    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_K_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offs_k = gl.arange(0, BLOCK_K, layout=layout)
    k_valid = offs_k < INPUT_K
    load_valid = row_active & slot_valid & expert_valid & k_valid
    a = gl.load(
        A_ptr + source_row * A_STRIDE_M + offs_k * A_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    weight = gl.load(
        B_ptr + expert_id * B_STRIDE_E + out_n * B_STRIDE_N + offs_k * B_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    scale = gl.load(
        B_scale_ptr
        + expert_id * B_SCALE_STRIDE_E
        + (out_n // WEIGHT_BLOCK_N) * B_SCALE_STRIDE_N
        + (offs_k // WEIGHT_BLOCK_K) * B_SCALE_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    acc = gl.sum(a * weight * scale, axis=0)

    if MUL_ROUTED_WEIGHT:
        routed_weight = gl.load(
            topk_weights_ptr + sorted_slot,
            mask=row_active & slot_valid,
            other=0.0,
        ).to(gl.float32)
        acc *= routed_weight

    if C_SORTED:
        c_row = sorted_row
        store_valid = row_active
    else:
        c_row = sorted_slot
        store_valid = row_active & slot_valid
    gl.store(
        C_ptr + c_row * C_STRIDE_M + out_n * C_STRIDE_N,
        acc.to(C_ptr.dtype.element_ty),
        mask=store_valid,
    )


@gluon.jit
def _w8a8_per_channel_sorted_expert_gemm_kernel(
    A_ptr,
    A_scale_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    A_STRIDE_M: gl.constexpr,
    A_STRIDE_K: gl.constexpr,
    A_SCALE_STRIDE_M: gl.constexpr,
    B_STRIDE_E: gl.constexpr,
    B_STRIDE_N: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    C_STRIDE_M: gl.constexpr,
    C_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_E: gl.constexpr,
    B_SCALE_STRIDE_N: gl.constexpr,
    NUM_VALID_SLOTS: gl.constexpr,
    INPUT_K: gl.constexpr,
    OUTPUT_N: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_ELEMS_PER_THREAD: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    C_SORTED: gl.constexpr,
):
    sorted_row = gl.program_id(0)
    out_n = gl.program_id(1)
    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr).to(gl.int32)
    row_active = sorted_row < num_tokens_post_padded

    sorted_slot = gl.load(
        sorted_token_ids_ptr + sorted_row,
        mask=row_active,
        other=NUM_VALID_SLOTS,
    ).to(gl.int32)
    slot_valid = sorted_slot < NUM_VALID_SLOTS
    expert_id = gl.load(
        expert_ids_ptr + sorted_row // BLOCK_SIZE_M,
        mask=row_active,
        other=-1,
    ).to(gl.int32)
    expert_valid = expert_id >= 0
    source_row = sorted_slot // TOP_K

    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_K_ELEMS_PER_THREAD], [64], [1], [0]
    )
    offs_k = gl.arange(0, BLOCK_K, layout=layout)
    k_valid = offs_k < INPUT_K
    load_valid = row_active & slot_valid & expert_valid & k_valid
    a = gl.load(
        A_ptr + source_row * A_STRIDE_M + offs_k * A_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    weight = gl.load(
        B_ptr + expert_id * B_STRIDE_E + out_n * B_STRIDE_N + offs_k * B_STRIDE_K,
        mask=load_valid,
        other=0.0,
    ).to(gl.float32)
    a_scale = gl.load(
        A_scale_ptr + source_row * A_SCALE_STRIDE_M,
        mask=row_active & slot_valid,
        other=0.0,
    ).to(gl.float32)
    b_scale = gl.load(
        B_scale_ptr + expert_id * B_SCALE_STRIDE_E + out_n * B_SCALE_STRIDE_N,
        mask=row_active & expert_valid,
        other=0.0,
    ).to(gl.float32)
    acc = gl.sum(a * weight, axis=0) * a_scale * b_scale

    if MUL_ROUTED_WEIGHT:
        routed_weight = gl.load(
            topk_weights_ptr + sorted_slot,
            mask=row_active & slot_valid,
            other=0.0,
        ).to(gl.float32)
        acc *= routed_weight

    if C_SORTED:
        c_row = sorted_row
        store_valid = row_active
    else:
        c_row = sorted_slot
        store_valid = row_active & slot_valid
    gl.store(
        C_ptr + c_row * C_STRIDE_M + out_n * C_STRIDE_N,
        acc.to(C_ptr.dtype.element_ty),
        mask=store_valid,
    )


def _triton_experts_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
) -> None:
    from tokenspeed_kernel.ops.moe.triton import invoke_fused_moe_kernel

    return invoke_fused_moe_kernel(
        A,
        B,
        bias,
        C,
        A_scale,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=block_shape,
        a_use_tma=a_use_tma,
        b_use_tma=b_use_tma,
        c_sorted=c_sorted,
        filter_expert=filter_expert,
    )


@register_kernel(
    "moe",
    "experts",
    name="gluon_fp8_local_experts_gfx950",
    features={"dispatch_sorted"},
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_EXPERT_SIGNATURES,
    priority=Priority.SPECIALIZED,
    tags={"latency"},
)
def gluon_fp8_local_experts_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
) -> None:
    if (
        not use_fp8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and not per_channel_quant
        and A_scale is None
        and B_scale is None
        and block_shape is None
        and bias is None
        and not a_use_tma
        and not b_use_tma
        and filter_expert
        and B.ndim == 3
    ):
        if A.ndim != 2:
            raise ValueError(f"A must be rank-2, got shape {tuple(A.shape)}")
        if C.shape[-1] != B.shape[1]:
            raise ValueError(
                f"C output dim {C.shape[-1]} does not match B N {B.shape[1]}"
            )
        if B.shape[2] != A.shape[1]:
            raise ValueError(f"B K {B.shape[2]} does not match A K {A.shape[1]}")
        if topk_weights.numel() != topk_ids.numel():
            raise ValueError("topk_weights and topk_ids must have matching element counts")

        output_n = B.shape[1]
        input_k = B.shape[2]
        if sorted_token_ids.numel() == 0 or output_n == 0:
            return
        block_k = max(64, 1 << (input_k - 1).bit_length())
        grid = (sorted_token_ids.shape[0], output_n)
        _dense_sorted_expert_gemm_kernel[grid](
            A,
            B,
            C,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(-2),
            C.stride(-1),
            topk_ids.numel(),
            input_k,
            output_n,
            top_k,
            config["BLOCK_SIZE_M"],
            block_k,
            block_k // 64,
            mul_routed_weight,
            c_sorted,
            num_warps=1,
        )
        return

    if (
        use_fp8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and per_channel_quant
        and A_scale is None
        and B_scale is not None
        and block_shape is None
        and bias is None
        and not a_use_tma
        and not b_use_tma
        and filter_expert
        and B.ndim == 3
        and B_scale.ndim in (2, 3)
    ):
        if A.ndim != 2:
            raise ValueError(f"A must be rank-2, got shape {tuple(A.shape)}")
        if C.shape[-1] != B.shape[1]:
            raise ValueError(
                f"C output dim {C.shape[-1]} does not match B N {B.shape[1]}"
            )
        if B.shape[2] != A.shape[1]:
            raise ValueError(f"B K {B.shape[2]} does not match A K {A.shape[1]}")
        if topk_weights.numel() != topk_ids.numel():
            raise ValueError("topk_weights and topk_ids must have matching element counts")
        if B_scale.shape[0] != B.shape[0] or B_scale.shape[1] != B.shape[1]:
            raise ValueError(
                f"B_scale shape {tuple(B_scale.shape)} is incompatible with "
                f"B shape {tuple(B.shape)} for per-channel quantization"
            )
        if B_scale.ndim == 3 and B_scale.shape[2] != 1:
            raise ValueError(
                f"B_scale per-channel rank-3 shape must end in 1, got "
                f"{tuple(B_scale.shape)}"
            )

        output_n = B.shape[1]
        input_k = B.shape[2]
        if sorted_token_ids.numel() == 0 or output_n == 0:
            return

        from tokenspeed_kernel.ops.gemm.fp8_utils import scaled_fp8_quant

        A_fp8, A_scale = scaled_fp8_quant(
            A,
            None,
            use_per_token_if_dynamic=True,
        )
        if A_scale.ndim != 2 or A_scale.shape[0] < A.shape[0]:
            raise ValueError(
                "per-token activation scale must have shape [M, 1], got "
                f"{tuple(A_scale.shape)}"
            )
        block_k = max(64, 1 << (input_k - 1).bit_length())
        grid = (sorted_token_ids.shape[0], output_n)
        _w8a8_per_channel_sorted_expert_gemm_kernel[grid](
            A_fp8,
            A_scale,
            B,
            C,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            A_fp8.stride(0),
            A_fp8.stride(1),
            A_scale.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(1),
            topk_ids.numel(),
            input_k,
            output_n,
            top_k,
            config["BLOCK_SIZE_M"],
            block_k,
            block_k // 64,
            mul_routed_weight,
            c_sorted,
            num_warps=1,
        )
        return

    if (
        not use_fp8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or per_channel_quant
        or A_scale is not None
        or B_scale is None
        or block_shape is None
        or bias is not None
        or a_use_tma
        or b_use_tma
        or not filter_expert
        or B.ndim != 3
        or B_scale.ndim != 3
    ):
        return _triton_experts_fallback(
            A,
            B,
            bias,
            C,
            A_scale,
            B_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape=block_shape,
            a_use_tma=a_use_tma,
            b_use_tma=b_use_tma,
            c_sorted=c_sorted,
            filter_expert=filter_expert,
        )

    if A.ndim != 2:
        raise ValueError(f"A must be rank-2, got shape {tuple(A.shape)}")
    if C.shape[-1] != B.shape[1]:
        raise ValueError(f"C output dim {C.shape[-1]} does not match B N {B.shape[1]}")
    if B.shape[2] != A.shape[1]:
        raise ValueError(f"B K {B.shape[2]} does not match A K {A.shape[1]}")
    if topk_weights.numel() != topk_ids.numel():
        raise ValueError("topk_weights and topk_ids must have matching element counts")

    output_n = B.shape[1]
    input_k = B.shape[2]
    if sorted_token_ids.numel() == 0 or output_n == 0:
        return
    block_k = max(64, 1 << (input_k - 1).bit_length())
    weight_block_n, weight_block_k = int(block_shape[0]), int(block_shape[1])
    grid = (sorted_token_ids.shape[0], output_n)
    _fp8_sorted_expert_gemm_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(-2),
        C.stride(-1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        topk_ids.numel(),
        input_k,
        output_n,
        top_k,
        config["BLOCK_SIZE_M"],
        weight_block_n,
        weight_block_k,
        block_k,
        block_k // 64,
        mul_routed_weight,
        c_sorted,
        num_warps=1,
    )
