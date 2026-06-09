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
from s1_component_utils import (
    assert_exact_metadata_equal,
    assert_selected_amd_kernel_not_reference,
)
from tokenspeed_kernel.ops.moe.gluon.ep_metadata_gfx950 import EPDispatchMetadata
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon EP metadata test")


def _uniform_owner_maps(
    world_size: int,
    num_local_experts: int,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = world_size * num_local_experts
    experts = torch.arange(num_experts, device=device, dtype=torch.int32)
    return experts // num_local_experts, experts % num_local_experts


def _reference_ep_metadata(
    topk_ids: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_local_experts: int,
) -> EPDispatchMetadata:
    topk_cpu = topk_ids.detach().cpu()
    owner_cpu = expert_owner.detach().cpu()
    local_cpu = local_expert_id.detach().cpu()
    flat = topk_cpu.flatten()
    owner_expert_counts = torch.zeros(
        (world_size, num_local_experts),
        dtype=torch.int32,
    )
    for expert_tensor in flat:
        expert = int(expert_tensor.item())
        if expert < 0 or expert >= owner_cpu.numel():
            continue
        owner = int(owner_cpu[expert].item())
        local_id = int(local_cpu[expert].item())
        if not (0 <= owner < world_size and 0 <= local_id < num_local_experts):
            continue
        owner_expert_counts[owner, local_id] += 1

    owner_counts = owner_expert_counts.sum(dim=1).to(torch.int32)
    owner_offsets = torch.zeros((world_size + 1,), dtype=torch.int32)
    owner_offsets[1:] = owner_counts.cumsum(dim=0)
    owner_expert_offsets = torch.zeros(
        (world_size, num_local_experts + 1),
        dtype=torch.int32,
    )
    owner_expert_offsets[:, 1:] = owner_expert_counts.cumsum(dim=1)

    dispatch_offsets = torch.full(topk_cpu.shape, -1, dtype=torch.int32)
    combine_offsets = torch.full(
        (world_size, flat.numel()),
        -1,
        dtype=torch.int32,
    )
    write_offsets = owner_expert_offsets[:, :-1].clone()
    for flat_slot, expert_tensor in enumerate(flat):
        expert = int(expert_tensor.item())
        if expert < 0 or expert >= owner_cpu.numel():
            continue
        owner = int(owner_cpu[expert].item())
        local_id = int(local_cpu[expert].item())
        if not (0 <= owner < world_size and 0 <= local_id < num_local_experts):
            continue
        row = int(write_offsets[owner, local_id].item())
        write_offsets[owner, local_id] += 1
        token = flat_slot // topk_cpu.shape[1]
        slot = flat_slot - token * topk_cpu.shape[1]
        dispatch_offsets[token, slot] = row
        combine_offsets[owner, row] = flat_slot

    return EPDispatchMetadata(
        owner_counts=owner_counts.to(topk_ids.device),
        owner_offsets=owner_offsets.to(topk_ids.device),
        owner_expert_counts=owner_expert_counts.to(topk_ids.device),
        owner_expert_offsets=owner_expert_offsets.to(topk_ids.device),
        local_expert_counts=owner_expert_counts[rank].to(topk_ids.device),
        local_expert_offsets=owner_expert_offsets[rank].to(topk_ids.device),
        dispatch_offsets=dispatch_offsets.to(topk_ids.device),
        combine_offsets=combine_offsets.to(topk_ids.device),
    )


def _call_ep_metadata(
    topk_ids: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_local_experts: int,
) -> EPDispatchMetadata:
    return tokenspeed_kernel.moe_dispatch(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank,
        world_size,
        num_local_experts,
        dtype=torch.int32,
        traits={"comm_strategy": "ep_metadata"},
    )


def _assert_ep_metadata_matches(actual: EPDispatchMetadata, expected: EPDispatchMetadata):
    assert_exact_metadata_equal(actual.owner_counts, expected.owner_counts)
    assert_exact_metadata_equal(actual.owner_offsets, expected.owner_offsets)
    assert_exact_metadata_equal(actual.owner_expert_counts, expected.owner_expert_counts)
    assert_exact_metadata_equal(
        actual.owner_expert_offsets,
        expected.owner_expert_offsets,
    )
    assert_exact_metadata_equal(actual.local_expert_counts, expected.local_expert_counts)
    assert_exact_metadata_equal(
        actual.local_expert_offsets,
        expected.local_expert_offsets,
    )
    assert_exact_metadata_equal(actual.dispatch_offsets, expected.dispatch_offsets)
    assert_exact_metadata_equal(actual.combine_offsets, expected.combine_offsets)


@pytest.mark.parametrize("world_size", [4, 8])
def test_gluon_ep_metadata_matches_cpu_oracle_mixed_layout(
    device: str,
    world_size: int,
) -> None:
    _require_cdna4_gpu()
    num_local_experts = 2
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
    )
    topk_ids = torch.tensor(
        [
            [0, 1, 2],
            [0, 3, 5],
            [6, 0, -1],
            [2, 3, 99],
            [0, 1, 6],
        ],
        device=device,
        dtype=torch.int32,
    )
    rank = min(2, world_size - 1)

    actual = _call_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    torch.cuda.synchronize()
    expected = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    assert actual.owner_counts[0].item() >= 5
    assert actual.owner_expert_counts.flatten().eq(0).any()
    _assert_ep_metadata_matches(actual, expected)


@pytest.mark.parametrize(
    ("rank", "topk_ids"),
    [
        (1, [[2, 3], [2, 3], [3, 2]]),
        (2, [[0, 1], [2, 3], [6, 7]]),
    ],
)
def test_gluon_ep_metadata_handles_all_local_and_all_remote(
    device: str,
    rank: int,
    topk_ids: list[list[int]],
) -> None:
    _require_cdna4_gpu()
    world_size = 4
    num_local_experts = 2
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
    )
    topk_tensor = torch.tensor(topk_ids, device=device, dtype=torch.int32)

    actual = _call_ep_metadata(
        topk_tensor,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    torch.cuda.synchronize()
    expected = _reference_ep_metadata(
        topk_tensor,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    _assert_ep_metadata_matches(actual, expected)
    if rank == 1:
        assert actual.local_expert_counts.sum().item() == topk_tensor.numel()
    else:
        assert actual.local_expert_counts.sum().item() == 0


def test_gluon_ep_metadata_handles_zero_token_rank(device: str) -> None:
    _require_cdna4_gpu()
    world_size = 8
    num_local_experts = 2
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
    )
    topk_ids = torch.empty((0, 2), device=device, dtype=torch.int32)

    actual = _call_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=3,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    torch.cuda.synchronize()
    expected = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=3,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    _assert_ep_metadata_matches(actual, expected)


def test_amd_ep_metadata_selection_uses_gluon(mi350_platform) -> None:
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "dispatch",
        platform=mi350_platform,
        format_signature=format_signature(indices=dense_tensor_format(torch.int32)),
        traits={"comm_strategy": "ep_metadata"},
        expected_solution="gluon",
    )
    assert selected.name == "gluon_ep_metadata_gfx950"
