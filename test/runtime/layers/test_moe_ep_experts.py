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
import torch

from tokenspeed.runtime.layers.moe.backends.ep_experts import (
    build_owner_expert_metadata,
    owner_rank_fp8_expert_gemm,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import EPWorkspaceError


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for owner-rank EP expert GEMM")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for owner-rank EP expert GEMM")


def _make_fp8_weight(
    dense: torch.Tensor,
    block_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.platform import current_platform

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


def _reference_owner_experts(
    owner_tokens: torch.Tensor,
    dense_weight: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    routed_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    output = torch.empty(
        (owner_tokens.shape[0], dense_weight.shape[1]),
        dtype=torch.float32,
        device=owner_tokens.device,
    )
    for expert in range(counts.numel()):
        start = int(offsets[expert].item())
        end = int(offsets[expert + 1].item())
        if start == end:
            continue
        output[start:end] = owner_tokens[start:end].float() @ dense_weight[expert].T
        if routed_weights is not None:
            output[start:end] *= routed_weights[start:end, None].float()
    return output


def test_build_owner_expert_metadata_preserves_ragged_owner_offsets() -> None:
    counts = torch.tensor([3, 0, 17], dtype=torch.int32)
    offsets = torch.tensor([0, 3, 3, 20], dtype=torch.int32)

    metadata = build_owner_expert_metadata(
        counts,
        block_size=16,
        local_expert_offsets=offsets,
    )

    assert metadata.num_rows == 20
    assert metadata.num_tokens_post_padded.tolist() == [48]
    assert metadata.expert_ids.tolist() == [0, 2, 2]
    assert metadata.sorted_token_ids[:3].tolist() == [0, 1, 2]
    assert metadata.sorted_token_ids[3:16].tolist() == [20] * 13
    assert metadata.sorted_token_ids[16:33].tolist() == list(range(3, 20))
    assert metadata.sorted_token_ids[33:].tolist() == [20] * 15


def test_build_owner_expert_metadata_rejects_offset_count_mismatch() -> None:
    counts = torch.tensor([2, 3], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)

    with pytest.raises(EPWorkspaceError, match="prefix sums"):
        build_owner_expert_metadata(
            counts,
            block_size=16,
            local_expert_offsets=offsets,
        )


def test_owner_rank_fp8_expert_gemm_rejects_bad_scale_shape_without_kernel_import() -> None:
    counts = torch.tensor([2], dtype=torch.int32)
    owner_tokens = torch.empty((2, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 16, 16), dtype=torch.float8_e4m3fn)
    bad_scale = torch.empty((1, 2, 1), dtype=torch.float32)

    with pytest.raises(EPWorkspaceError, match="local_weight_scale shape"):
        owner_rank_fp8_expert_gemm(
            owner_tokens,
            weight,
            bad_scale,
            counts,
            block_shape=(16, 16),
            block_size=16,
        )


def test_owner_rank_fp8_expert_gemm_rejects_non_fp8_weight() -> None:
    counts = torch.tensor([1], dtype=torch.int32)
    owner_tokens = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 16, 16), dtype=torch.bfloat16)
    scale = torch.empty((1, 1, 1), dtype=torch.float32)

    with pytest.raises(EPWorkspaceError, match="FP8 E4M3"):
        owner_rank_fp8_expert_gemm(
            owner_tokens,
            weight,
            scale,
            counts,
            block_shape=(16, 16),
            block_size=16,
        )


def test_owner_rank_fp8_expert_gemm_matches_empty_and_hot_experts() -> None:
    _require_cdna4_gpu()
    torch.manual_seed(8404)
    device = torch.device("cuda")
    counts = torch.tensor([2, 0, 33, 3], dtype=torch.int32, device=device)
    offsets = torch.zeros((counts.numel() + 1,), dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    hidden = 32
    output = 48
    block_shape = (16, 16)
    block_size = 16

    owner_tokens = (torch.randn(int(offsets[-1].item()), hidden, device=device) * 0.18).to(
        torch.bfloat16
    )
    dense_weight = (
        torch.randn(counts.numel(), output, hidden, device=device) * 0.16
    )
    weight, weight_scale = _make_fp8_weight(dense_weight, block_shape)

    result = owner_rank_fp8_expert_gemm(
        owner_tokens,
        weight,
        weight_scale,
        counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=offsets,
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()

    expected = _reference_owner_experts(
        owner_tokens,
        _dequantize_fp8_weight(weight, weight_scale, block_shape),
        counts,
        offsets,
    )
    assert counts[1].item() == 0
    torch.testing.assert_close(result.float(), expected, atol=4e-2, rtol=5e-2)


def test_owner_rank_fp8_expert_gemm_handles_multiple_sources_per_expert() -> None:
    _require_cdna4_gpu()
    torch.manual_seed(8405)
    device = torch.device("cuda")
    source_counts_by_expert = [
        [1, 2, 0],
        [2, 1, 2],
        [0, 3, 1],
    ]
    counts = torch.tensor(
        [sum(per_source) for per_source in source_counts_by_expert],
        dtype=torch.int32,
        device=device,
    )
    offsets = torch.zeros((counts.numel() + 1,), dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    hidden = 32
    output = 32
    block_shape = (16, 16)
    block_size = 16

    rows = []
    for expert, per_source in enumerate(source_counts_by_expert):
        for source_rank, count in enumerate(per_source):
            for row in range(count):
                values = torch.arange(hidden, dtype=torch.float32)
                rows.append(values * 0.01 + expert + source_rank * 0.1 + row * 0.001)
    owner_tokens = torch.stack(rows).to(device=device, dtype=torch.bfloat16)
    routed_weights = torch.linspace(
        0.25,
        1.0,
        steps=owner_tokens.shape[0],
        device=device,
        dtype=torch.float32,
    )
    dense_weight = (
        torch.randn(counts.numel(), output, hidden, device=device) * 0.12
    )
    weight, weight_scale = _make_fp8_weight(dense_weight, block_shape)

    result = owner_rank_fp8_expert_gemm(
        owner_tokens,
        weight,
        weight_scale,
        counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=offsets,
        routed_weights=routed_weights,
        mul_routed_weight=True,
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()

    expected = _reference_owner_experts(
        owner_tokens,
        _dequantize_fp8_weight(weight, weight_scale, block_shape),
        counts,
        offsets,
        routed_weights=routed_weights,
    )
    assert all(len([c for c in per_source if c > 0]) >= 2 for per_source in source_counts_by_expert)
    torch.testing.assert_close(result.float(), expected, atol=3e-2, rtol=5e-2)
