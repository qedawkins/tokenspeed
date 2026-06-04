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

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import EPOwnerDispatchPlan
from tokenspeed.runtime.layers.moe.backends import ep_fused_metadata
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    build_pre_routed_fused_ep_metadata,
    prepare_pre_routed_fused_ep_metadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for S4 EP metadata kernel comparison")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for S4 EP metadata kernel comparison")


def _uniform_owner_maps(
    world_size: int,
    num_local_experts: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = world_size * num_local_experts
    experts = torch.arange(num_experts, dtype=torch.int32, device=device)
    return experts // num_local_experts, experts % num_local_experts


def _workspace(
    *,
    world_size: int,
    rank: int,
    hidden_size: int,
    max_tokens_per_rank: int = 8,
    top_k: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EPCommunicationWorkspace:
    return EPCommunicationWorkspace.allocate(
        max_tokens_per_rank=max_tokens_per_rank,
        hidden_size=hidden_size,
        top_k=top_k,
        world_size=world_size,
        rank=rank,
        dtype=dtype,
        device=device,
        iris_mode="disabled",
    )


def _reference_ep_metadata(
    topk_ids: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_local_experts: int,
) -> SimpleNamespace:
    topk_cpu = topk_ids.detach().cpu()
    owner_cpu = expert_owner.detach().cpu()
    local_cpu = local_expert_id.detach().cpu()
    flat = topk_cpu.reshape(-1)
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
        if 0 <= owner < world_size and 0 <= local_id < num_local_experts:
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

    return SimpleNamespace(
        owner_counts=owner_counts.to(topk_ids.device),
        owner_offsets=owner_offsets.to(topk_ids.device),
        owner_expert_counts=owner_expert_counts.to(topk_ids.device),
        owner_expert_offsets=owner_expert_offsets.to(topk_ids.device),
        local_expert_counts=owner_expert_counts[rank].to(topk_ids.device),
        local_expert_offsets=owner_expert_offsets[rank].to(topk_ids.device),
        dispatch_offsets=dispatch_offsets.to(topk_ids.device),
        combine_offsets=combine_offsets.to(topk_ids.device),
    )


def _build(
    topk_ids: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_local_experts: int,
    hidden_size: int = 4,
    dispatch_plan: EPOwnerDispatchPlan | None = None,
):
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=topk_ids.device,
    )
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=max(topk_ids.shape[0], 1),
        top_k=topk_ids.shape[1],
        device=topk_ids.device,
    )
    hidden_states = torch.arange(
        topk_ids.shape[0] * hidden_size,
        dtype=torch.float32,
        device=topk_ids.device,
    ).reshape(topk_ids.shape[0], hidden_size)
    topk_weights = torch.linspace(
        0.1,
        0.9,
        steps=max(topk_ids.numel(), 1),
        dtype=torch.float32,
        device=topk_ids.device,
    )[: topk_ids.numel()].reshape(topk_ids.shape)
    fused = build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        dispatch_plan=dispatch_plan,
    )
    return fused, ep_metadata, topk_weights


def test_pre_routed_fused_metadata_matches_s4_unfused_metadata() -> None:
    topk_ids = torch.tensor(
        [
            [0, 1, 2],
            [0, 3, 5],
            [6, 0, -1],
            [2, 3, 99],
            [0, 1, 6],
        ],
        dtype=torch.int32,
    )
    fused, ep_metadata, topk_weights = _build(
        topk_ids,
        rank=2,
        world_size=4,
        num_local_experts=2,
    )

    torch.testing.assert_close(fused.owner_counts, ep_metadata.owner_counts)
    torch.testing.assert_close(fused.owner_offsets, ep_metadata.owner_offsets)
    torch.testing.assert_close(
        fused.owner_expert_counts,
        ep_metadata.owner_expert_counts,
    )
    torch.testing.assert_close(
        fused.owner_expert_offsets,
        ep_metadata.owner_expert_offsets,
    )
    torch.testing.assert_close(
        fused.aggregate_owner_expert_counts,
        ep_metadata.owner_expert_counts,
    )
    torch.testing.assert_close(
        fused.aggregate_owner_expert_offsets,
        ep_metadata.owner_expert_offsets,
    )
    torch.testing.assert_close(fused.owner_rows, ep_metadata.dispatch_offsets.reshape(-1))
    torch.testing.assert_close(fused.topk_weights, topk_weights.reshape(-1))
    assert fused.num_valid_slots == 13
    invalid_flat_slots = torch.tensor([8, 11], dtype=torch.long)
    assert fused.owner_ranks[invalid_flat_slots].eq(-1).all()
    assert fused.local_expert_ids[invalid_flat_slots].eq(-1).all()
    assert fused.owner_rows[invalid_flat_slots].eq(-1).all()


def test_pre_routed_fused_metadata_matches_actual_s4_metadata_kernel() -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel

    device = "cuda"
    rank = 2
    world_size = 4
    num_local_experts = 2
    topk_ids = torch.tensor(
        [
            [0, 1, 2],
            [0, 3, 5],
            [6, 0, -1],
            [2, 3, 99],
            [0, 1, 6],
        ],
        dtype=torch.int32,
        device=device,
    )
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
    )
    ep_metadata = tokenspeed_kernel.moe_dispatch(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank,
        world_size,
        num_local_experts,
        dtype=torch.int32,
        traits={"comm_strategy": "ep_metadata"},
        expected_kernel_name="gluon_ep_metadata_gfx950",
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
    workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=4,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        device=device,
    )
    hidden_states = torch.arange(
        topk_ids.shape[0] * 4,
        dtype=torch.float32,
        device=device,
    ).reshape(topk_ids.shape[0], 4)
    topk_weights = torch.linspace(
        0.1,
        0.9,
        steps=topk_ids.numel(),
        dtype=torch.float32,
        device=device,
    ).reshape(topk_ids.shape)

    fused = build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )

    torch.testing.assert_close(ep_metadata.owner_counts, expected.owner_counts)
    torch.testing.assert_close(
        ep_metadata.owner_expert_offsets,
        expected.owner_expert_offsets,
    )
    torch.testing.assert_close(fused.owner_counts, ep_metadata.owner_counts)
    torch.testing.assert_close(fused.owner_rows, ep_metadata.dispatch_offsets.reshape(-1))
    torch.testing.assert_close(fused.topk_weights, topk_weights.reshape(-1))


@pytest.mark.parametrize(
    ("rank", "topk_ids", "expected_valid_slots"),
    [
        (1, [[2, 3], [2, 3], [3, 2]], 6),
        (2, [[0, 1], [2, 3], [6, 7]], 6),
    ],
)
def test_pre_routed_fused_metadata_handles_all_local_and_all_remote(
    rank: int,
    topk_ids: list[list[int]],
    expected_valid_slots: int,
) -> None:
    fused, ep_metadata, _ = _build(
        torch.tensor(topk_ids, dtype=torch.int32),
        rank=rank,
        world_size=4,
        num_local_experts=2,
    )

    assert fused.num_valid_slots == expected_valid_slots
    torch.testing.assert_close(fused.owner_rows, ep_metadata.dispatch_offsets.reshape(-1))
    if rank == 1:
        assert fused.local_expert_counts.sum().item() == expected_valid_slots
    else:
        assert fused.local_expert_counts.sum().item() == 0


def test_pre_routed_fused_metadata_handles_zero_token_rank() -> None:
    topk_ids = torch.empty((0, 2), dtype=torch.int32)
    fused, ep_metadata, _ = _build(
        topk_ids,
        rank=3,
        world_size=8,
        num_local_experts=2,
    )

    assert fused.num_slots == 0
    assert fused.num_valid_slots == 0
    assert fused.owner_row_to_source.shape == (8, 0, 3)
    torch.testing.assert_close(fused.owner_counts, ep_metadata.owner_counts)
    torch.testing.assert_close(
        fused.owner_expert_counts,
        torch.zeros((8, 2), dtype=torch.int32),
    )


def test_prepare_pre_routed_fused_metadata_handles_all_remote_without_dist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ep_fused_metadata,
        "_has_distributed_process_group",
        lambda: False,
    )
    topk_ids = torch.tensor([[0, 1], [2, 3], [6, 7]], dtype=torch.int32)
    world_size = 4
    rank = 2
    num_local_experts = 2
    expert_owner, local_expert_id = _uniform_owner_maps(world_size, num_local_experts)
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
    )
    workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=4,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
    )
    hidden_states = torch.zeros((topk_ids.shape[0], 4), dtype=torch.float32)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

    fused = prepare_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )

    assert fused.num_valid_slots == topk_ids.numel()
    assert fused.local_expert_counts.sum().item() == 0
    torch.testing.assert_close(fused.owner_rows, ep_metadata.dispatch_offsets.reshape(-1))


def test_pre_routed_fused_metadata_handles_hot_expert_and_empty_expert() -> None:
    topk_ids = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.int32)
    fused, ep_metadata, _ = _build(
        topk_ids,
        rank=0,
        world_size=4,
        num_local_experts=2,
    )

    assert fused.num_valid_slots == topk_ids.numel()
    assert fused.owner_counts.tolist() == [6, 0, 0, 0]
    assert fused.owner_expert_counts[0].tolist() == [6, 0]
    assert fused.owner_expert_counts.flatten().eq(0).any()
    torch.testing.assert_close(fused.owner_rows, ep_metadata.dispatch_offsets.reshape(-1))


def test_pre_routed_fused_metadata_round_trips_owner_rows_to_source_identity() -> None:
    topk_ids = torch.tensor(
        [
            [0, 4],
            [7, 1],
            [2, 5],
        ],
        dtype=torch.int32,
    )
    rank = 2
    fused, _, _ = _build(
        topk_ids,
        rank=rank,
        world_size=4,
        num_local_experts=2,
    )

    for flat_slot in range(topk_ids.numel()):
        owner = int(fused.owner_ranks[flat_slot].item())
        owner_row = int(fused.owner_rows[flat_slot].item())
        token = flat_slot // topk_ids.shape[1]
        slot = flat_slot - token * topk_ids.shape[1]
        assert owner >= 0
        assert owner_row >= 0
        assert fused.owner_row_to_source_flat_slot[owner, owner_row].item() == flat_slot
        assert fused.owner_row_to_source[owner, owner_row].tolist() == [
            rank,
            token,
            slot,
        ]


def test_pre_routed_fused_metadata_applies_aggregate_dispatch_plan_offsets() -> None:
    topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    world_size = 2
    num_local_experts = 2
    aggregate_counts = torch.tensor([[5, 4], [0, 0]], dtype=torch.int32)
    aggregate_offsets = torch.tensor([[0, 5, 9], [0, 0, 0]], dtype=torch.int32)
    base_offsets = torch.tensor([[3, 1], [0, 0]], dtype=torch.int32)
    dispatch_plan = EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros((world_size,), dtype=torch.int32),
        owner_expert_base_offsets=base_offsets,
        aggregate_owner_expert_counts=aggregate_counts,
        aggregate_owner_expert_offsets=aggregate_offsets,
        local_expert_counts=aggregate_counts[0],
        local_expert_offsets=aggregate_offsets[0],
    )
    fused, _, _ = _build(
        topk_ids,
        rank=0,
        world_size=world_size,
        num_local_experts=num_local_experts,
        dispatch_plan=dispatch_plan,
    )

    expected_rows = torch.tensor([3, 6, 4, 7], dtype=torch.int32)
    torch.testing.assert_close(fused.owner_rows, expected_rows)
    assert fused.owner_row_to_source_flat_slot[0, 3].item() == 0
    assert fused.owner_row_to_source_flat_slot[0, 6].item() == 1
    assert fused.owner_row_to_source[0, 7].tolist() == [0, 1, 1]


def test_pre_routed_fused_metadata_rejects_shape_mismatch() -> None:
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    expert_owner, local_expert_id = _uniform_owner_maps(1, 2)
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=0,
        world_size=1,
        num_local_experts=2,
    )
    workspace = _workspace(world_size=1, rank=0, hidden_size=4)
    hidden_states = torch.zeros((1, 4), dtype=torch.float32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(EPWorkspaceError, match="topk_weights shape"):
        build_pre_routed_fused_ep_metadata(
            hidden_states,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
        )


def test_pre_routed_fused_metadata_rejects_workspace_topk_mismatch() -> None:
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    expert_owner, local_expert_id = _uniform_owner_maps(1, 2)
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=0,
        world_size=1,
        num_local_experts=2,
    )
    workspace = _workspace(world_size=1, rank=0, hidden_size=4, top_k=1)
    hidden_states = torch.zeros((1, 4), dtype=torch.float32)
    topk_weights = torch.ones((1, 2), dtype=torch.float32)

    with pytest.raises(EPWorkspaceError, match="top_k mismatch"):
        build_pre_routed_fused_ep_metadata(
            hidden_states,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
        )
