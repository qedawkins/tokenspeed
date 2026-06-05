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

import math

import pytest
import tokenspeed_kernel
import torch
from s1_component_utils import (
    assert_finite_close,
    assert_selected_amd_kernel_not_reference,
)
from tokenspeed_kernel._triton import tl
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon MoE expert GEMM test")


def _expert_config(block_size: int, block_shape: tuple[int, int]) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": block_shape[0],
        "BLOCK_SIZE_K": block_shape[1],
        "GROUP_SIZE_M": 1,
        "num_warps": 1,
        "num_stages": 1,
    }


def _dispatch_metadata(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tokenspeed_kernel.moe_dispatch(
        topk_ids,
        block_size,
        num_experts,
        dtype=torch.int32,
        traits={"comm_strategy": "local"},
    )


def _make_fp8_weight(
    dense: torch.Tensor,
    block_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = current_platform().fp8e4m3fn
    block_n, block_k = block_shape
    num_experts, n_size, k_size = dense.shape
    scale = torch.empty(
        (
            num_experts,
            math.ceil(n_size / block_n),
            math.ceil(k_size / block_k),
        ),
        device=dense.device,
        dtype=torch.float32,
    )
    quantized = torch.empty(dense.shape, device=dense.device, dtype=fp8.dtype)

    for expert in range(num_experts):
        for n_block, n_start in enumerate(range(0, n_size, block_n)):
            n_end = min(n_start + block_n, n_size)
            for k_block, k_start in enumerate(range(0, k_size, block_k)):
                k_end = min(k_start + block_k, k_size)
                block = dense[expert, n_start:n_end, k_start:k_end].float()
                block_scale = torch.clamp(block.abs().max() / fp8.max, min=1e-6)
                scale[expert, n_block, k_block] = block_scale
                quantized[expert, n_start:n_end, k_start:k_end] = torch.clamp(
                    block / block_scale,
                    min=fp8.min,
                    max=fp8.max,
                ).to(fp8.dtype)

    return quantized, scale


def _dequantize_fp8_weight(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_shape
    num_experts, n_size, k_size = quantized.shape
    dense = torch.empty(quantized.shape, device=quantized.device, dtype=torch.float32)

    for expert in range(num_experts):
        for n_block, n_start in enumerate(range(0, n_size, block_n)):
            n_end = min(n_start + block_n, n_size)
            for k_block, k_start in enumerate(range(0, k_size, block_k)):
                k_end = min(k_start + block_k, k_size)
                dense[expert, n_start:n_end, k_start:k_end] = (
                    quantized[expert, n_start:n_end, k_start:k_end].float()
                    * scale[expert, n_block, k_block]
                )

    return dense


def _make_per_channel_fp8_weight(
    dense: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = current_platform().fp8e4m3fn
    scale = torch.clamp(
        dense.float().abs().amax(dim=2, keepdim=True) / fp8.max,
        min=1e-6,
    )
    quantized = torch.clamp(
        dense.float() / scale,
        min=fp8.min,
        max=fp8.max,
    ).to(fp8.dtype)
    return quantized, scale


def _dequantize_per_channel_fp8_weight(
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return quantized.float() * scale.float()


def _reference_slots(
    A: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    top_k: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    flat_ids = topk_ids.flatten()
    flat_weights = topk_weights.flatten()
    out = torch.zeros(
        (flat_ids.numel(), weight.shape[1]),
        device=A.device,
        dtype=torch.float32,
    )

    for slot, expert_tensor in enumerate(flat_ids):
        expert = int(expert_tensor.item())
        if expert < 0 or expert >= weight.shape[0]:
            continue
        source_row = slot // top_k
        out[slot] = A[source_row].float() @ weight[expert].float().T
        if mul_routed_weight:
            out[slot] *= flat_weights[slot].float()

    return out


def _reference_w8a8_slots(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    top_k: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    from tokenspeed_kernel.ops.gemm.fp8_utils import scaled_fp8_quant

    A_fp8, A_scale = scaled_fp8_quant(
        A.contiguous(),
        None,
        use_per_token_if_dynamic=True,
    )
    A_dequantized = A_fp8.float() * A_scale.float()
    return _reference_slots(
        A_dequantized,
        _dequantize_per_channel_fp8_weight(B, B_scale),
        topk_ids,
        topk_weights,
        top_k=top_k,
        mul_routed_weight=mul_routed_weight,
    )


def _call_experts(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    block_size: int,
    block_shape: tuple[int, int],
    top_k: int,
    mul_routed_weight: bool,
) -> None:
    tokenspeed_kernel.moe_experts(
        A,
        B,
        None,
        C,
        None,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        _expert_config(block_size, block_shape),
        tl.bfloat16,
        True,
        False,
        False,
        False,
        block_shape=list(block_shape),
        dtype=A.dtype,
        features={"dispatch_sorted"},
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )


def _call_w8a8_per_channel_experts(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    block_size: int,
    top_k: int,
    mul_routed_weight: bool,
) -> None:
    tokenspeed_kernel.moe_experts(
        A,
        B,
        None,
        C,
        None,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        _expert_config(block_size, (16, 16)),
        tl.bfloat16,
        True,
        False,
        False,
        True,
        block_shape=None,
        dtype=A.dtype,
        features={"dispatch_sorted"},
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )


def _call_bf16_experts(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    block_size: int,
    top_k: int,
    mul_routed_weight: bool,
) -> None:
    tokenspeed_kernel.moe_experts(
        A,
        B,
        None,
        C,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        _expert_config(block_size, (16, 16)),
        tl.bfloat16,
        False,
        False,
        False,
        False,
        block_shape=None,
        dtype=A.dtype,
        features={"dispatch_sorted"},
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )


def _forbid_triton_experts_fallback(monkeypatch) -> None:
    import tokenspeed_kernel.ops.moe.gluon.experts_fp8_gfx950 as experts_mod

    def _fail_fallback(*args, **kwargs):
        pytest.fail("W8A8 per-channel expert GEMM used Triton fallback")

    monkeypatch.setattr(experts_mod, "_triton_experts_fallback", _fail_fallback)


def test_gluon_bf16_experts_gate_up_matches_dense_reference(device: str) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7201)
    num_tokens = 8
    top_k = 2
    num_experts = 4
    hidden = 32
    intermediate_tp = 24
    block_size = 16
    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [3, 0],
            [0, 3],
            [1, 0],
            [3, 1],
            [0, 1],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.25,
        1.0,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    A = (torch.randn(num_tokens, hidden, device=device) * 0.25).bfloat16()
    B = (
        torch.randn(num_experts, 2 * intermediate_tp, hidden, device=device) * 0.20
    ).bfloat16()
    C = torch.zeros((num_tokens * top_k, 2 * intermediate_tp), device=device).bfloat16()

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_bf16_experts(
        A,
        B,
        C,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        top_k=top_k,
        mul_routed_weight=False,
    )
    torch.cuda.synchronize()

    expected = _reference_slots(
        A,
        B,
        topk_ids,
        topk_weights,
        top_k=top_k,
        mul_routed_weight=False,
    )
    assert not topk_ids.eq(2).any()
    assert_finite_close(C, expected, atol=4e-2, rtol=4e-2)


def test_gluon_bf16_experts_down_matches_dense_reference_with_routed_weights(
    device: str,
) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7202)
    num_tokens = 6
    top_k = 2
    num_experts = 4
    intermediate_tp = 24
    hidden_tp = 32
    block_size = 16
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, -1],
            [3, 0],
            [4, 1],
            [1, 0],
            [3, 2],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.15,
        0.90,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    A = (
        torch.randn(num_tokens * top_k, intermediate_tp, device=device) * 0.20
    ).bfloat16()
    B = (
        torch.randn(num_experts, hidden_tp, intermediate_tp, device=device) * 0.18
    ).bfloat16()
    C = torch.full(
        (num_tokens, top_k, hidden_tp),
        9.0,
        device=device,
        dtype=torch.bfloat16,
    )

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_bf16_experts(
        A,
        B,
        C,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        top_k=1,
        mul_routed_weight=True,
    )
    torch.cuda.synchronize()

    expected = _reference_slots(
        A,
        B,
        topk_ids,
        topk_weights,
        top_k=1,
        mul_routed_weight=True,
    )
    assert bool(topk_ids.eq(-1).any().item())
    assert bool(topk_ids.ge(num_experts).any().item())
    assert_finite_close(C.view(-1, hidden_tp), expected, atol=3e-2, rtol=4e-2)


def test_gluon_fp8_experts_gate_up_matches_reference_with_empty_expert(
    device: str,
) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7101)
    num_tokens = 8
    top_k = 2
    num_experts = 4
    hidden = 32
    intermediate_tp = 24
    block_size = 16
    block_shape = (16, 16)
    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [3, 0],
            [0, 3],
            [1, 0],
            [3, 1],
            [0, 1],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.25,
        1.0,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    A = (torch.randn(num_tokens, hidden, device=device) * 0.25).to(torch.bfloat16)
    dense_weight = (
        torch.randn(num_experts, 2 * intermediate_tp, hidden, device=device) * 0.20
    )
    B, B_scale = _make_fp8_weight(dense_weight, block_shape)
    C = torch.zeros((num_tokens * top_k, 2 * intermediate_tp), device=device).bfloat16()

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_experts(
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        block_shape=block_shape,
        top_k=top_k,
        mul_routed_weight=False,
    )
    torch.cuda.synchronize()

    expected = _reference_slots(
        A,
        _dequantize_fp8_weight(B, B_scale, block_shape),
        topk_ids,
        topk_weights,
        top_k=top_k,
        mul_routed_weight=False,
    )
    assert not topk_ids.eq(2).any()
    assert_finite_close(C, expected, atol=4e-2, rtol=4e-2)


def test_gluon_fp8_experts_down_matches_reference_with_routed_weights(
    device: str,
) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7102)
    num_tokens = 8
    top_k = 2
    num_experts = 4
    intermediate_tp = 24
    hidden_tp = 32
    block_size = 16
    block_shape = (16, 16)
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [3, 0],
            [2, 1],
            [1, 0],
            [3, 2],
            [0, 2],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.15,
        0.90,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    A = (
        torch.randn(num_tokens * top_k, intermediate_tp, device=device) * 0.20
    ).bfloat16()
    dense_weight = (
        torch.randn(num_experts, hidden_tp, intermediate_tp, device=device) * 0.18
    )
    B, B_scale = _make_fp8_weight(dense_weight, block_shape)
    C = torch.zeros((num_tokens, top_k, hidden_tp), device=device).bfloat16()

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_experts(
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        block_shape=block_shape,
        top_k=1,
        mul_routed_weight=True,
    )
    torch.cuda.synchronize()

    expected = _reference_slots(
        A,
        _dequantize_fp8_weight(B, B_scale, block_shape),
        topk_ids,
        topk_weights,
        top_k=1,
        mul_routed_weight=True,
    )
    assert_finite_close(C.view(-1, hidden_tp), expected, atol=3e-2, rtol=4e-2)


def test_gluon_fp8_experts_handles_one_expert_and_invalid_slots(device: str) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7103)
    num_tokens = 4
    top_k = 2
    num_experts = 4
    hidden = 16
    output = 32
    block_size = 16
    block_shape = (16, 16)
    topk_ids = torch.tensor(
        [
            [3, 3],
            [3, -1],
            [4, 3],
            [3, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.ones((num_tokens, top_k), device=device, dtype=torch.float32)
    A = (torch.randn(num_tokens, hidden, device=device) * 0.15).bfloat16()
    dense_weight = (torch.randn(num_experts, output, hidden, device=device) * 0.12)
    B, B_scale = _make_fp8_weight(dense_weight, block_shape)
    C = torch.full(
        (num_tokens * top_k, output),
        9.0,
        device=device,
        dtype=torch.bfloat16,
    )

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_experts(
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        block_shape=block_shape,
        top_k=top_k,
        mul_routed_weight=False,
    )
    torch.cuda.synchronize()

    expected = _reference_slots(
        A,
        _dequantize_fp8_weight(B, B_scale, block_shape),
        topk_ids,
        topk_weights,
        top_k=top_k,
        mul_routed_weight=False,
    )
    assert expert_ids[: num_post.item() // block_size].tolist().count(3) == 1
    assert -1 in expert_ids[: num_post.item() // block_size].tolist()
    assert_finite_close(C, expected, atol=2e-2, rtol=4e-2)


def test_gluon_w8a8_per_channel_experts_gate_up_matches_dequantized_reference(
    device: str,
    monkeypatch,
) -> None:
    _require_cdna4_gpu()
    _forbid_triton_experts_fallback(monkeypatch)
    torch.manual_seed(7301)
    num_tokens = 8
    top_k = 2
    num_experts = 4
    hidden = 32
    output = 48
    block_size = 16
    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [3, 0],
            [0, 3],
            [1, 0],
            [3, 1],
            [0, 1],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.2,
        0.95,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    row_scales = torch.tensor(
        [0.015, 0.04, 0.09, 0.16, 0.25, 0.40, 0.70, 1.20],
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, 1)
    A = (torch.randn(num_tokens, hidden, device=device) * row_scales).bfloat16()
    channel_scales = torch.linspace(
        0.03,
        1.80,
        steps=output,
        device=device,
        dtype=torch.float32,
    ).view(1, output, 1)
    dense_weight = (
        torch.randn(num_experts, output, hidden, device=device)
        * channel_scales
        * 0.18
    )
    B, B_scale = _make_per_channel_fp8_weight(dense_weight)
    C = torch.zeros((num_tokens * top_k, output), device=device).bfloat16()

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_w8a8_per_channel_experts(
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        top_k=top_k,
        mul_routed_weight=False,
    )
    torch.cuda.synchronize()

    expected = _reference_w8a8_slots(
        A,
        B,
        B_scale,
        topk_ids,
        topk_weights,
        top_k=top_k,
        mul_routed_weight=False,
    )
    assert not topk_ids.eq(2).any()
    assert B_scale.shape == (num_experts, output, 1)
    assert_finite_close(C, expected, atol=6e-2, rtol=6e-2)


def test_gluon_w8a8_per_channel_experts_down_matches_dequantized_reference(
    device: str,
    monkeypatch,
) -> None:
    _require_cdna4_gpu()
    _forbid_triton_experts_fallback(monkeypatch)
    torch.manual_seed(7302)
    num_tokens = 6
    top_k = 2
    num_experts = 4
    intermediate_tp = 24
    hidden_tp = 32
    block_size = 16
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [3, 0],
            [2, 1],
            [1, 0],
            [3, 2],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.15,
        0.90,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)
    slot_scales = torch.linspace(
        0.04,
        0.85,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens * top_k, 1)
    A = (
        torch.randn(num_tokens * top_k, intermediate_tp, device=device)
        * slot_scales
    ).bfloat16()
    channel_scales = torch.linspace(
        0.05,
        1.50,
        steps=hidden_tp,
        device=device,
        dtype=torch.float32,
    ).view(1, hidden_tp, 1)
    dense_weight = (
        torch.randn(num_experts, hidden_tp, intermediate_tp, device=device)
        * channel_scales
        * 0.15
    )
    B, B_scale = _make_per_channel_fp8_weight(dense_weight)
    C = torch.zeros((num_tokens, top_k, hidden_tp), device=device).bfloat16()

    sorted_ids, expert_ids, num_post = _dispatch_metadata(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )
    _call_w8a8_per_channel_experts(
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size=block_size,
        top_k=1,
        mul_routed_weight=True,
    )
    torch.cuda.synchronize()

    expected = _reference_w8a8_slots(
        A,
        B,
        B_scale,
        topk_ids,
        topk_weights,
        top_k=1,
        mul_routed_weight=True,
    )
    assert bool(torch.all(B_scale[..., 0] > 0).item())
    assert_finite_close(C.view(-1, hidden_tp), expected, atol=6e-2, rtol=6e-2)


def test_amd_fp8_local_experts_selection_uses_gluon(mi350_platform) -> None:
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "experts",
        platform=mi350_platform,
        format_signature=format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=frozenset({"dispatch_sorted"}),
        expected_solution="gluon",
    )
    assert selected.name == "gluon_fp8_local_experts_gfx950"


def test_amd_w8a8_local_experts_signature_uses_gluon(mi350_platform) -> None:
    fp8_scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None) or getattr(
        torch,
        "float8_e4m3fnuz",
    )
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "experts",
        platform=mi350_platform,
        format_signature=format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=tensor_format(
                "scaled-fp8",
                fp8_dtype,
                scale=fp8_scale,
            ),
        ),
        features=frozenset({"dispatch_sorted"}),
        expected_solution="gluon",
    )
    assert selected.name == "gluon_fp8_local_experts_gfx950"
