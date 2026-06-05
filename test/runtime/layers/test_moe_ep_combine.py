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

import gc
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import tokenspeed.runtime.layers.moe.backends.ep_combine as ep_combine_module
from tokenspeed.runtime.layers.moe.backends.ep_combine import owner_directed_combine
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import EPOwnerDispatchPlan
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
    EPWorkspaceUnavailable,
)


def _workspace(
    *,
    world_size: int,
    rank: int,
    hidden_size: int,
    max_tokens_per_rank: int = 4,
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


def _metadata(
    dispatch_offsets: list[list[int]],
    owner_expert_counts: list[list[int]],
    *,
    top_k: int | None = None,
    device: torch.device | str = "cpu",
) -> SimpleNamespace:
    if dispatch_offsets:
        dispatch = torch.tensor(dispatch_offsets, dtype=torch.int32, device=device)
    else:
        dispatch = torch.empty((0, top_k or 0), dtype=torch.int32, device=device)
    counts = torch.tensor(owner_expert_counts, dtype=torch.int32, device=device)
    offsets = torch.zeros(
        (counts.shape[0], counts.shape[1] + 1),
        dtype=torch.int32,
        device=device,
    )
    offsets[:, 1:] = torch.cumsum(counts, dim=1, dtype=torch.int32)
    return SimpleNamespace(
        dispatch_offsets=dispatch,
        owner_expert_counts=counts,
        owner_expert_offsets=offsets,
    )


def _plan(
    owner_expert_counts: list[list[int]],
    *,
    rank: int = 0,
    owner_expert_base_offsets: list[list[int]] | None = None,
    device: torch.device | str = "cpu",
) -> EPOwnerDispatchPlan:
    aggregate_counts = torch.tensor(owner_expert_counts, dtype=torch.int32, device=device)
    aggregate_offsets = torch.zeros(
        (aggregate_counts.shape[0], aggregate_counts.shape[1] + 1),
        dtype=torch.int32,
        device=device,
    )
    aggregate_offsets[:, 1:] = torch.cumsum(
        aggregate_counts,
        dim=1,
        dtype=torch.int32,
    )
    if owner_expert_base_offsets is None:
        owner_expert_base_offsets = [
            [0 for _ in range(aggregate_counts.shape[1])]
            for _ in range(aggregate_counts.shape[0])
        ]
    base_offsets = torch.tensor(
        owner_expert_base_offsets,
        dtype=torch.int32,
        device=device,
    )
    local_counts = aggregate_counts[rank].contiguous()
    local_offsets = aggregate_offsets[rank].contiguous()
    return EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros(
            (aggregate_counts.shape[0],),
            dtype=torch.int32,
            device=device,
        ),
        owner_expert_base_offsets=base_offsets,
        aggregate_owner_expert_counts=aggregate_counts,
        aggregate_owner_expert_offsets=aggregate_offsets,
        local_expert_counts=local_counts,
        local_expert_offsets=local_offsets,
    )


def _row_values(num_rows: int, hidden_size: int, *, device: torch.device | str = "cpu"):
    return torch.arange(
        num_rows * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(num_rows, hidden_size)


def test_combine_workspace_sync_disabled_during_capture(monkeypatch) -> None:
    assert ep_combine_module._should_synchronize_workspace(True)
    assert not ep_combine_module._should_synchronize_workspace(False)

    monkeypatch.setattr(ep_combine_module, "get_is_capture_mode", lambda: True)

    assert not ep_combine_module._should_synchronize_workspace(True)


def test_owner_directed_combine_preserves_topk_slot_order() -> None:
    workspace = _workspace(world_size=1, rank=0, hidden_size=3)
    topk_ids = torch.tensor([[0, 1], [1, 0], [2, -1]], dtype=torch.int32)
    metadata = _metadata([[0, 2], [3, 1], [4, -1]], [[2, 2, 1]])
    dispatch_plan = _plan([[2, 2, 1]])
    owner_outputs = _row_values(5, 3)
    expert_owner = torch.tensor([0, 0, 0], dtype=torch.int32)
    local_expert_id = torch.tensor([0, 1, 2], dtype=torch.int32)

    returned = owner_directed_combine(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )

    expected = torch.stack(
        [
            torch.stack([owner_outputs[0], owner_outputs[2]]),
            torch.stack([owner_outputs[3], owner_outputs[1]]),
            torch.stack([owner_outputs[4], torch.zeros(3)]),
        ]
    )
    torch.testing.assert_close(returned, expected)


def test_owner_directed_combine_reads_aggregate_owner_expert_rows() -> None:
    workspace = _workspace(world_size=2, rank=0, hidden_size=2)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = _metadata([[0, 1]], [[1, 1], [0, 0]])
    dispatch_plan = _plan(
        [[3, 2], [0, 0]],
        owner_expert_base_offsets=[[2, 1], [0, 0]],
    )
    owner_outputs = _row_values(5, 2)
    expert_owner = torch.tensor([0, 0], dtype=torch.int32)
    local_expert_id = torch.tensor([0, 1], dtype=torch.int32)

    returned = owner_directed_combine(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )

    expected = torch.stack([torch.stack([owner_outputs[2], owner_outputs[4]])])
    torch.testing.assert_close(returned, expected)


def test_owner_directed_combine_handles_hot_expert_and_empty_owner() -> None:
    workspace = _workspace(world_size=2, rank=0, hidden_size=2)
    topk_ids = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.int32)
    metadata = _metadata([[0, 1], [2, 3], [4, 5]], [[6], [0]])
    dispatch_plan = _plan([[6], [0]])
    owner_outputs = _row_values(6, 2)
    expert_owner = torch.tensor([0], dtype=torch.int32)
    local_expert_id = torch.tensor([0], dtype=torch.int32)

    returned = owner_directed_combine(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )

    torch.testing.assert_close(returned.reshape(-1, 2), owner_outputs)


def test_owner_directed_combine_handles_empty_source_rank() -> None:
    workspace = _workspace(world_size=4, rank=2, hidden_size=4, max_tokens_per_rank=0)
    topk_ids = torch.empty((0, 2), dtype=torch.int32)
    metadata = _metadata([], [[0], [0], [0], [0]], top_k=2)
    dispatch_plan = _plan([[0], [0], [0], [0]], rank=2)
    owner_outputs = torch.empty((0, 4), dtype=torch.float32)
    expert_owner = torch.tensor([2], dtype=torch.int32)
    local_expert_id = torch.tensor([0], dtype=torch.int32)

    returned = owner_directed_combine(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )

    assert returned.shape == (0, 2, 4)


def test_owner_directed_combine_empty_source_rank_keeps_iris_barriers() -> None:
    class FakeIrisContext:
        def __init__(self) -> None:
            self.barrier_count = 0

        def barrier(self) -> None:
            self.barrier_count += 1

    workspace = _workspace(world_size=2, rank=1, hidden_size=4, max_tokens_per_rank=1)
    fake_context = FakeIrisContext()
    workspace.backend = "iris"
    workspace._iris_context = fake_context
    topk_ids = torch.empty((0, 2), dtype=torch.int32)
    metadata = _metadata([], [[0], [1]], top_k=2)
    dispatch_plan = _plan([[0], [1]], rank=1)
    owner_outputs = workspace.combine_buffer[:1]
    expert_owner = torch.tensor([1], dtype=torch.int32)
    local_expert_id = torch.tensor([0], dtype=torch.int32)

    returned = owner_directed_combine(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )

    assert returned.shape == (0, 2, 4)
    assert fake_context.barrier_count == 2


def test_owner_directed_combine_rejects_remote_owner_without_iris() -> None:
    workspace = _workspace(world_size=2, rank=0, hidden_size=2)
    topk_ids = torch.tensor([[1]], dtype=torch.int32)
    metadata = _metadata([[0]], [[0], [1]])
    dispatch_plan = _plan([[0], [1]])
    owner_outputs = _row_values(1, 2)
    expert_owner = torch.tensor([0, 1], dtype=torch.int32)
    local_expert_id = torch.tensor([0, 0], dtype=torch.int32)

    with pytest.raises(EPWorkspaceUnavailable, match="remote owner"):
        owner_directed_combine(
            owner_outputs,
            topk_ids,
            metadata,
            dispatch_plan,
            workspace,
            expert_owner,
            local_expert_id,
        )


def test_owner_directed_combine_rejects_shape_mismatch() -> None:
    workspace = _workspace(world_size=1, rank=0, hidden_size=2)
    topk_ids = torch.tensor([[0, 0]], dtype=torch.int32)
    metadata = _metadata([[0]], [[1]])
    dispatch_plan = _plan([[1]])
    owner_outputs = _row_values(1, 2)
    expert_owner = torch.tensor([0], dtype=torch.int32)
    local_expert_id = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(EPWorkspaceError, match="dispatch_offsets shape"):
        owner_directed_combine(
            owner_outputs,
            topk_ids,
            metadata,
            dispatch_plan,
            workspace,
            expert_owner,
            local_expert_id,
        )


@pytest.mark.parametrize("required_world_size", [2, 4, 8])
def test_iris_owner_directed_combine_rank_filled_tensor(
    required_world_size: int,
) -> None:
    iris = pytest.importorskip("iris")
    if not dist.is_available() or not dist.is_initialized():
        pytest.skip("Iris combine test requires torch.distributed")
    if dist.get_world_size() != required_world_size:
        pytest.skip(f"requires world_size={required_world_size}")

    ctx = iris.iris(heap_size=1 << 27)
    try:
        rank = ctx.get_rank()
        world_size = ctx.get_num_ranks()
        device = ctx.get_device()
        hidden_size = 4
        topk_ids = torch.arange(world_size, dtype=torch.int32, device=device).reshape(
            world_size,
            1,
        )
        metadata = _metadata(
            [[owner] for owner in range(world_size)],
            [[1] for _ in range(world_size)],
            device=device,
        )
        dispatch_plan = _plan(
            [[world_size] for _ in range(world_size)],
            rank=rank,
            owner_expert_base_offsets=[[rank] for _ in range(world_size)],
            device=device,
        )
        workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=1,
            hidden_size=hidden_size,
            top_k=1,
            dtype=torch.float32,
            iris_mode="required",
            iris_context=ctx,
        )
        workspace.combine_buffer[:world_size].fill_(rank * 100)
        workspace.combine_buffer[:world_size] += torch.arange(
            world_size,
            dtype=torch.float32,
            device=device,
        ).reshape(world_size, 1)
        expert_owner = torch.arange(world_size, dtype=torch.int32, device=device)
        local_expert_id = torch.zeros((world_size,), dtype=torch.int32, device=device)

        returned = owner_directed_combine(
            workspace.combine_buffer,
            topk_ids,
            metadata,
            dispatch_plan,
            workspace,
            expert_owner,
            local_expert_id,
        )

        expected = torch.empty_like(returned)
        for token in range(world_size):
            expected[token, 0].fill_(token * 100 + rank)
        torch.testing.assert_close(returned, expected)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()


def test_iris_owner_directed_combine_one_rank_compile_smoke() -> None:
    iris = pytest.importorskip("iris")
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for Iris combine compile smoke")
    initialized_here = _ensure_one_rank_dist_initialized()
    if dist.get_world_size() != 1:
        pytest.skip("one-rank compile smoke requires world_size=1")

    ctx = iris.iris(heap_size=1 << 26)
    try:
        device = ctx.get_device()
        workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=3,
            hidden_size=3,
            top_k=2,
            dtype=torch.float32,
            iris_mode="required",
            iris_context=ctx,
        )
        topk_ids = torch.tensor(
            [[0, 1], [1, 0], [2, -1]],
            dtype=torch.int32,
            device=device,
        )
        metadata = _metadata(
            [[0, 2], [3, 1], [4, -1]],
            [[2, 2, 1]],
            device=device,
        )
        dispatch_plan = _plan([[2, 2, 1]], device=device)
        owner_outputs = _row_values(5, 3, device=device)
        workspace.combine_buffer[:5].copy_(owner_outputs)
        expert_owner = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        local_expert_id = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        returned = owner_directed_combine(
            workspace.combine_buffer,
            topk_ids,
            metadata,
            dispatch_plan,
            workspace,
            expert_owner,
            local_expert_id,
        )

        expected = torch.stack(
            [
                torch.stack([owner_outputs[0], owner_outputs[2]]),
                torch.stack([owner_outputs[3], owner_outputs[1]]),
                torch.stack([owner_outputs[4], torch.zeros(3, device=device)]),
            ]
        )
        torch.testing.assert_close(returned, expected)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()
            if initialized_here:
                dist.destroy_process_group()


def _ensure_one_rank_dist_initialized() -> bool:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if dist.is_initialized():
        return False
    torch.cuda.set_device(0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=1,
        rank=0,
    )
    return True
