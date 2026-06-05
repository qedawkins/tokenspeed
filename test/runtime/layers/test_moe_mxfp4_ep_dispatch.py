from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.ep_ownership import (
    build_uniform_expert_owner_maps,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import EPCommunicationWorkspace
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_dispatch import (
    dispatch_mxfp4_hidden_states_to_owner_buffers,
)


def _workspace(
    *,
    world_size: int,
    rank: int,
    hidden_size: int,
    max_tokens_per_rank: int,
    top_k: int,
    dtype: torch.dtype,
) -> EPCommunicationWorkspace:
    return EPCommunicationWorkspace.allocate(
        max_tokens_per_rank=max_tokens_per_rank,
        hidden_size=hidden_size,
        top_k=top_k,
        world_size=world_size,
        rank=rank,
        dtype=dtype,
        device="cpu",
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
    flat = topk_ids.reshape(-1).cpu()
    owner_cpu = expert_owner.cpu()
    local_cpu = local_expert_id.cpu()
    owner_expert_counts = torch.zeros(
        (world_size, num_local_experts),
        dtype=torch.int32,
    )
    for expert_tensor in flat:
        expert = int(expert_tensor.item())
        owner = int(owner_cpu[expert].item())
        local_id = int(local_cpu[expert].item())
        owner_expert_counts[owner, local_id] += 1

    owner_counts = owner_expert_counts.sum(dim=1).to(torch.int32)
    owner_offsets = torch.zeros((world_size + 1,), dtype=torch.int32)
    owner_offsets[1:] = owner_counts.cumsum(dim=0)
    owner_expert_offsets = torch.zeros(
        (world_size, num_local_experts + 1),
        dtype=torch.int32,
    )
    owner_expert_offsets[:, 1:] = owner_expert_counts.cumsum(dim=1)
    dispatch_offsets = torch.full(topk_ids.shape, -1, dtype=torch.int32)
    combine_offsets = torch.full((world_size, flat.numel()), -1, dtype=torch.int32)
    write_offsets = owner_expert_offsets[:, :-1].clone()
    top_k = topk_ids.shape[1]
    for flat_slot, expert_tensor in enumerate(flat):
        expert = int(expert_tensor.item())
        owner = int(owner_cpu[expert].item())
        local_id = int(local_cpu[expert].item())
        owner_row = int(write_offsets[owner, local_id].item())
        write_offsets[owner, local_id] += 1
        token = flat_slot // top_k
        slot = flat_slot - token * top_k
        dispatch_offsets[token, slot] = owner_row
        combine_offsets[owner, owner_row] = flat_slot

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


def _mixed_kimi_topk_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [95, 96, 191, 192, 287, 288, 383, 1],
            [10, 106, 202, 298, 20, 116, 212, 308],
            [300, 301, 302, 303, 304, 305, 306, 307],
            [94, 95, 96, 97, 190, 191, 192, 193],
            [286, 287, 288, 289, 382, 383, 2, 3],
        ],
        dtype=torch.int32,
    )


def _expected_owner_rows(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_metadata: SimpleNamespace,
    owner: int,
) -> torch.Tensor:
    token_ids: list[int] = []
    for local_id in range(ep_metadata.owner_expert_counts.shape[1]):
        start = int(ep_metadata.owner_expert_offsets[owner, local_id].item())
        end = int(ep_metadata.owner_expert_offsets[owner, local_id + 1].item())
        for owner_row in range(start, end):
            flat_slot = int(ep_metadata.combine_offsets[owner, owner_row].item())
            token_ids.append(flat_slot // topk_ids.shape[1])
    if not token_ids:
        return hidden_states.new_empty((0, hidden_states.shape[1]))
    return hidden_states[torch.tensor(token_ids, dtype=torch.long)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_mxfp4_owner_dispatch_matches_kimi_384_expert_cpu_oracle(
    rank: int,
    dtype: torch.dtype,
) -> None:
    world_size = 4
    num_local_experts = 96
    hidden_size = 16
    topk_ids = _mixed_kimi_topk_ids()
    topk_weights = torch.linspace(
        0.1,
        0.9,
        steps=topk_ids.numel(),
        dtype=torch.float32,
    ).reshape(topk_ids.shape)
    hidden_states = (
        torch.arange(topk_ids.shape[0] * hidden_size, dtype=torch.float32)
        .reshape(topk_ids.shape[0], hidden_size)
        .to(dtype)
    )
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=world_size,
        device="cpu",
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
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        dtype=dtype,
    )

    result = dispatch_mxfp4_hidden_states_to_owner_buffers(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )
    expected = _expected_owner_rows(hidden_states, topk_ids, ep_metadata, rank)

    assert result.owner_tokens.dtype == dtype
    assert result.dispatch_step.num_rows == expected.shape[0]
    torch.testing.assert_close(result.owner_tokens, expected)
    assert result.fused_metadata.source_rank == rank
    assert result.fused_metadata.world_size == world_size
    assert result.fused_metadata.top_k == 8

    local_flat_slots = ep_metadata.combine_offsets[rank, : expected.shape[0]]
    assert torch.unique(local_flat_slots).numel() == local_flat_slots.numel()
    for owner_row, flat_slot_tensor in enumerate(local_flat_slots.tolist()):
        token = flat_slot_tensor // topk_ids.shape[1]
        slot = flat_slot_tensor - token * topk_ids.shape[1]
        assert result.fused_metadata.owner_row_to_source[rank, owner_row].tolist() == [
            rank,
            token,
            slot,
        ]


def test_mxfp4_owner_dispatch_handles_empty_owner_rank() -> None:
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    hidden_states = torch.randn(topk_ids.shape[0], 12, dtype=torch.bfloat16)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=4,
        device="cpu",
    )
    rank = 3
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=4,
        num_local_experts=96,
    )
    workspace = _workspace(
        world_size=4,
        rank=rank,
        hidden_size=hidden_states.shape[1],
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        dtype=hidden_states.dtype,
    )

    result = dispatch_mxfp4_hidden_states_to_owner_buffers(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )

    assert result.owner_tokens.shape == (0, hidden_states.shape[1])
    assert result.dispatch_step.num_rows == 0
    assert result.fused_metadata.owner_counts.tolist() == [topk_ids.numel(), 0, 0, 0]


def test_mxfp4_owner_dispatch_reuses_prebuilt_fused_metadata() -> None:
    topk_ids = _mixed_kimi_topk_ids()
    hidden_states = torch.randn(topk_ids.shape[0], 8, dtype=torch.bfloat16)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    expert_owner, local_expert_id = build_uniform_expert_owner_maps(
        num_experts=384,
        world_size=4,
        device="cpu",
    )
    rank = 1
    ep_metadata = _reference_ep_metadata(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank=rank,
        world_size=4,
        num_local_experts=96,
    )
    workspace = _workspace(
        world_size=4,
        rank=rank,
        hidden_size=hidden_states.shape[1],
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        dtype=hidden_states.dtype,
    )
    first = dispatch_mxfp4_hidden_states_to_owner_buffers(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )
    second = dispatch_mxfp4_hidden_states_to_owner_buffers(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        fused_metadata=first.fused_metadata,
    )

    torch.testing.assert_close(second.owner_tokens, first.owner_tokens)
    torch.testing.assert_close(
        second.dispatch_plan.local_expert_offsets,
        first.dispatch_plan.local_expert_offsets,
    )
