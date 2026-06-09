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

import pytest
import tokenspeed_kernel
import torch
from s1_component_utils import assert_selected_amd_kernel_not_reference
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon MoE dispatch test")


def _reference_local_dispatch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_ids_cpu = topk_ids.detach().cpu()
    total_tokens, topk = topk_ids_cpu.shape
    pad_id = total_tokens * topk
    max_num_tokens_padded = pad_id + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_ids = torch.full((max_num_tokens_padded,), pad_id, dtype=torch.int32)
    expert_ids = torch.zeros((max_num_m_blocks,), dtype=torch.int32)

    flat = topk_ids_cpu.flatten()
    write_pos = 0
    block_idx = 0
    flat_slots = torch.arange(pad_id, dtype=torch.int32)
    for expert_id in range(num_experts + 1):
        if expert_id == num_experts:
            mask = (flat < 0) | (flat >= num_experts)
            stored_expert = -1
        else:
            mask = flat == expert_id
            stored_expert = expert_id
        slots = flat_slots[mask]
        n_blocks = (slots.numel() + block_size - 1) // block_size
        for block in range(n_blocks):
            start = write_pos + block * block_size
            block_slots = slots[block * block_size : (block + 1) * block_size]
            sorted_ids[start : start + block_slots.numel()] = block_slots
            expert_ids[block_idx + block] = stored_expert
        write_pos += n_blocks * block_size
        block_idx += n_blocks

    num_tokens_post_pad = torch.tensor([write_pos], dtype=torch.int32)
    return (
        sorted_ids.to(topk_ids.device),
        expert_ids.to(topk_ids.device),
        num_tokens_post_pad.to(topk_ids.device),
    )


def _canonical_dispatch_metadata(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    *,
    block_size: int,
    pad_id: int,
) -> torch.Tensor:
    block_aligned = expert_ids.numel() * block_size
    padded_sorted = torch.full(
        (block_aligned,),
        pad_id,
        dtype=torch.int32,
        device=sorted_ids.device,
    )
    padded_sorted[: sorted_ids.numel()] = sorted_ids
    sorted_blocks = padded_sorted.view(expert_ids.numel(), block_size).clone()
    start = 0
    while start < expert_ids.numel():
        end = start + 1
        while end < expert_ids.numel() and expert_ids[end] == expert_ids[start]:
            end += 1
        run = sorted_blocks[start:end].flatten().sort().values
        sorted_blocks[start:end] = run.view(end - start, block_size)
        start = end
    return torch.cat(
        [
            num_tokens_post_pad.reshape(-1).to(torch.int32),
            expert_ids.to(torch.int32),
            sorted_blocks.flatten().to(torch.int32),
        ]
    )


def _assert_dispatch_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_size: int,
    pad_id: int,
) -> None:
    actual_sorted, actual_experts, actual_num = actual
    expected_sorted, expected_experts, expected_num = expected
    assert actual_sorted.dtype == torch.int32
    assert actual_experts.dtype == torch.int32
    assert actual_num.dtype == torch.int32
    assert actual_sorted.shape == expected_sorted.shape
    assert actual_experts.shape == expected_experts.shape
    assert actual_num.shape == expected_num.shape
    assert torch.equal(actual_num, expected_num)
    actual_canon = _canonical_dispatch_metadata(
        actual_sorted,
        actual_experts,
        actual_num,
        block_size=block_size,
        pad_id=pad_id,
    )
    expected_canon = _canonical_dispatch_metadata(
        expected_sorted,
        expected_experts,
        expected_num,
        block_size=block_size,
        pad_id=pad_id,
    )
    assert torch.equal(actual_canon, expected_canon)


def _call_dispatch(
    topk_ids: torch.Tensor,
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


@pytest.mark.parametrize(
    ("num_tokens", "topk", "block_size"),
    [
        (0, 1, 16),
        (1, 2, 64),
        (8, 8, 64),
        (32, 1, 128),
        (257, 8, 64),
    ],
)
def test_gluon_local_dispatch_matches_reference_shape_matrix(
    device: str,
    num_tokens: int,
    topk: int,
    block_size: int,
) -> None:
    _require_cdna4_gpu()
    num_experts = 16
    torch.manual_seed(7000 + num_tokens + topk)
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
    )

    actual = _call_dispatch(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()
    expected = _reference_local_dispatch(topk_ids, block_size, num_experts)
    _assert_dispatch_matches_reference(
        actual,
        expected,
        block_size=block_size,
        pad_id=num_tokens * topk,
    )


@pytest.mark.parametrize("block_size", [16, 64, 128])
def test_gluon_local_dispatch_handles_all_tokens_on_one_expert(
    device: str, block_size: int
) -> None:
    _require_cdna4_gpu()
    num_tokens = 33
    topk = 2
    num_experts = 8
    topk_ids = torch.full(
        (num_tokens, topk),
        3,
        device=device,
        dtype=torch.int32,
    )

    actual = _call_dispatch(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()
    expected = _reference_local_dispatch(topk_ids, block_size, num_experts)
    _assert_dispatch_matches_reference(
        actual,
        expected,
        block_size=block_size,
        pad_id=num_tokens * topk,
    )


def test_gluon_local_dispatch_handles_empty_and_invalid_experts(device: str) -> None:
    _require_cdna4_gpu()
    topk_ids = torch.tensor(
        [
            [0, 1, -1, 4],
            [1, 9, 4, -1],
            [0, 0, 4, 4],
            [9, -1, 1, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    num_experts = 5
    block_size = 16

    actual = _call_dispatch(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()
    expected = _reference_local_dispatch(topk_ids, block_size, num_experts)
    _, actual_experts, actual_num = actual
    assert -1 in actual_experts[: actual_num.item() // block_size].tolist()
    _assert_dispatch_matches_reference(
        actual,
        expected,
        block_size=block_size,
        pad_id=topk_ids.numel(),
    )


def test_amd_local_dispatch_selection_uses_gluon(mi350_platform) -> None:
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "dispatch",
        platform=mi350_platform,
        format_signature=format_signature(indices=dense_tensor_format(torch.int32)),
        traits={"comm_strategy": "local"},
        expected_solution="gluon",
    )
    assert selected.name == "gluon_local_dispatch_gfx950"
