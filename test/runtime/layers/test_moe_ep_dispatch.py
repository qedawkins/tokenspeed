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

import tokenspeed.runtime.layers.moe.backends.ep_dispatch as ep_dispatch_module
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    EPOwnerDispatchPlan,
    owner_directed_dispatch,
    prepare_owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)


def _metadata(
    owner_counts: list[int],
    combine_rows: list[list[int]],
    *,
    owner_expert_counts: list[list[int]] | None = None,
    device: torch.device | str = "cpu",
) -> SimpleNamespace:
    world_size = len(owner_counts)
    max_rows = max((len(rows) for rows in combine_rows), default=0)
    combine_offsets = torch.full(
        (world_size, max_rows),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for owner, rows in enumerate(combine_rows):
        if rows:
            combine_offsets[owner, : len(rows)] = torch.tensor(
                rows,
                dtype=torch.int32,
                device=device,
            )
    counts = torch.tensor(owner_counts, dtype=torch.int32, device=device)
    if owner_expert_counts is None:
        owner_expert_counts = [[count] for count in owner_counts]
    expert_counts = torch.tensor(owner_expert_counts, dtype=torch.int32, device=device)
    expert_offsets = torch.zeros(
        (world_size, expert_counts.shape[1] + 1),
        dtype=torch.int32,
        device=device,
    )
    expert_offsets[:, 1:] = torch.cumsum(expert_counts, dim=1, dtype=torch.int32)
    owner_offsets = torch.zeros((world_size + 1,), dtype=torch.int32, device=device)
    owner_offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return SimpleNamespace(
        owner_counts=counts,
        owner_offsets=owner_offsets,
        owner_expert_counts=expert_counts,
        owner_expert_offsets=expert_offsets,
        combine_offsets=combine_offsets,
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


def test_owner_directed_dispatch_one_rank_duplicates_rows() -> None:
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor([[0, 0], [0, -1], [0, 0]], dtype=torch.int32)
    metadata = _metadata([5], [[0, 1, 2, 4, 5]])
    workspace = _workspace(world_size=1, rank=0, hidden_size=4)

    step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

    expected_tokens = torch.tensor([0, 0, 1, 2, 2])
    torch.testing.assert_close(step.dispatch_buffer, hidden_states[expected_tokens])
    assert step.num_rows == 5
    assert torch.equal(step.rank_counts, torch.tensor([5], dtype=torch.int32))
    assert torch.equal(step.rank_offsets, torch.tensor([0, 5], dtype=torch.int32))


def test_owner_expert_metadata_value_checks_skip_during_capture(monkeypatch) -> None:
    workspace = _workspace(world_size=2, rank=0, hidden_size=4)
    metadata = _metadata(
        [2, 2],
        [[0, 1], [2, 3]],
        owner_expert_counts=[[1, 1], [1, 1]],
    )
    metadata.owner_expert_offsets[0, 0] = 1

    with pytest.raises(EPWorkspaceError, match="must start at zero"):
        ep_dispatch_module._validate_owner_expert_metadata(metadata, workspace)

    monkeypatch.setattr(ep_dispatch_module, "get_is_capture_mode", lambda: True)
    ep_dispatch_module._validate_owner_expert_metadata(metadata, workspace)


def test_dispatch_plan_value_checks_skip_during_capture(monkeypatch) -> None:
    workspace = _workspace(world_size=2, rank=0, hidden_size=4)
    metadata = _metadata(
        [2, 2],
        [[0, 1], [2, 3]],
        owner_expert_counts=[[1, 1], [1, 1]],
    )
    dispatch_plan = EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros(2, dtype=torch.int32),
        owner_expert_base_offsets=torch.zeros((2, 2), dtype=torch.int32),
        aggregate_owner_expert_counts=torch.ones((2, 2), dtype=torch.int32),
        aggregate_owner_expert_offsets=torch.tensor(
            [[0, 1, 2], [0, 1, 2]],
            dtype=torch.int32,
        ),
        local_expert_counts=torch.tensor([1, 1], dtype=torch.int32),
        local_expert_offsets=torch.tensor([0, 2, 2], dtype=torch.int32),
    )

    with pytest.raises(EPWorkspaceError, match="local_expert_offsets"):
        ep_dispatch_module._validate_dispatch_plan(dispatch_plan, metadata, workspace)

    monkeypatch.setattr(ep_dispatch_module, "get_is_capture_mode", lambda: True)
    ep_dispatch_module._validate_dispatch_plan(dispatch_plan, metadata, workspace)


def test_dispatch_offsets_from_counts() -> None:
    counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    offsets = ep_dispatch_module._offsets_from_counts(counts)
    owner_counts = torch.tensor([[1, 0], [2, 3]], dtype=torch.int32)
    owner_offsets = ep_dispatch_module._owner_offsets_from_counts(owner_counts)

    assert offsets.tolist() == [0, 2, 2, 5]
    assert owner_offsets.tolist() == [[0, 1, 1], [0, 2, 5]]


def test_dispatch_offsets_from_counts_cuda_graph_safe() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for CUDA graph offset smoke test")
    device = torch.device("cuda")
    counts = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
    owner_counts = torch.tensor([[1, 0], [2, 3]], dtype=torch.int32, device=device)

    ep_dispatch_module._offsets_from_counts(counts)
    ep_dispatch_module._owner_offsets_from_counts(owner_counts)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        offsets = ep_dispatch_module._offsets_from_counts(counts)
        owner_offsets = ep_dispatch_module._owner_offsets_from_counts(owner_counts)
    graph.replay()
    torch.cuda.synchronize()

    assert offsets.cpu().tolist() == [0, 2, 2, 5]
    assert owner_offsets.cpu().tolist() == [[0, 1, 1], [0, 2, 5]]


def test_prepare_dispatch_uses_static_rows_during_capture(monkeypatch) -> None:
    metadata = _metadata([5], [[0, 1, 2, 4, 5]])
    workspace = _workspace(world_size=1, rank=0, hidden_size=4)
    observed: dict[str, int | None] = {}
    prepare_step = workspace.prepare_step

    def spy_prepare_step(
        rank_counts: torch.Tensor,
        rank_offsets: torch.Tensor,
        *,
        num_rows: int | None = None,
    ):
        observed["num_rows"] = num_rows
        return prepare_step(rank_counts, rank_offsets, num_rows=num_rows)

    monkeypatch.setattr(ep_dispatch_module, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(workspace, "prepare_step", spy_prepare_step)

    prepare_owner_directed_dispatch(metadata, workspace)

    assert observed["num_rows"] == workspace.max_dispatch_rows


def test_owner_directed_dispatch_returns_static_view_during_capture(monkeypatch) -> None:
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor([[0, 0], [0, -1], [0, 0]], dtype=torch.int32)
    metadata = _metadata([5], [[0, 1, 2, 4, 5]])
    workspace = _workspace(world_size=1, rank=0, hidden_size=4)

    monkeypatch.setattr(ep_dispatch_module, "get_is_capture_mode", lambda: True)

    step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

    assert step.num_rows == workspace.max_dispatch_rows
    assert step.dispatch_buffer.shape == (workspace.max_dispatch_rows, 4)


def test_dispatch_grid_owner_rows_uses_static_capacity_during_capture(
    monkeypatch,
) -> None:
    metadata = _metadata([5], [[0, 1, 2, 4, 5]])
    workspace = _workspace(world_size=1, rank=0, hidden_size=4)

    assert ep_dispatch_module._dispatch_grid_owner_rows(metadata, workspace) == 5

    monkeypatch.setattr(ep_dispatch_module, "get_is_capture_mode", lambda: True)

    assert (
        ep_dispatch_module._dispatch_grid_owner_rows(metadata, workspace)
        == workspace.max_dispatch_rows
    )


def test_dispatch_workspace_sync_disabled_during_capture(monkeypatch) -> None:
    assert ep_dispatch_module._should_synchronize_workspace(True)
    assert not ep_dispatch_module._should_synchronize_workspace(False)

    monkeypatch.setattr(ep_dispatch_module, "get_is_capture_mode", lambda: True)

    assert not ep_dispatch_module._should_synchronize_workspace(True)


def test_owner_directed_dispatch_one_rank_gpu_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for owner-directed dispatch smoke test")
    device = torch.device("cuda")
    hidden_states = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    topk_ids = torch.tensor(
        [[0, 0], [0, -1], [0, 0]],
        dtype=torch.int32,
        device=device,
    )
    metadata = _metadata([5], [[0, 1, 2, 4, 5]], device=device)
    workspace = _workspace(world_size=1, rank=0, hidden_size=4, device=device)

    step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

    expected_tokens = torch.tensor([0, 0, 1, 2, 2], device=device)
    torch.testing.assert_close(step.dispatch_buffer, hidden_states[expected_tokens])


def test_owner_directed_dispatch_zero_source_rows() -> None:
    hidden_states = torch.empty((0, 4), dtype=torch.float32)
    topk_ids = torch.empty((0, 2), dtype=torch.int32)
    metadata = _metadata([0, 0, 0, 0], [[], [], [], []])
    workspace = _workspace(world_size=4, rank=2, hidden_size=4, max_tokens_per_rank=0)

    step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

    assert step.dispatch_buffer.shape == (0, 4)
    assert step.num_rows == 0
    assert torch.equal(step.rank_counts, torch.zeros((4,), dtype=torch.int32))
    assert torch.equal(step.rank_offsets, torch.zeros((5,), dtype=torch.int32))


@pytest.mark.parametrize(
    ("rank", "owner_counts", "combine_rows", "expected_tokens"),
    [
        (1, [0, 3, 0, 0], [[], [0, 2, 3], [], []], [0, 1, 1]),
        (2, [2, 1, 0, 3], [[0, 1], [2], [], [3, 4, 5]], []),
    ],
)
def test_owner_directed_dispatch_all_local_and_all_remote_single_source(
    rank: int,
    owner_counts: list[int],
    combine_rows: list[list[int]],
    expected_tokens: list[int],
) -> None:
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    metadata = _metadata(owner_counts, combine_rows)
    workspace = _workspace(world_size=4, rank=rank, hidden_size=4)

    step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

    assert step.num_rows == len(expected_tokens)
    if expected_tokens:
        torch.testing.assert_close(
            step.dispatch_buffer,
            hidden_states[torch.tensor(expected_tokens)],
        )
    else:
        assert step.dispatch_buffer.shape == (0, 4)


def test_prepare_owner_directed_dispatch_aggregates_owner_expert_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(
        [3, 1],
        [[0, 1, 2], [3]],
        owner_expert_counts=[[2, 1], [0, 1]],
    )
    workspace = _workspace(world_size=2, rank=1, hidden_size=4)
    source0_owner_counts = torch.tensor([2, 1], dtype=torch.int32)
    source1_owner_counts = metadata.owner_counts
    source0_owner_expert_counts = torch.tensor(
        [[1, 1], [1, 0]],
        dtype=torch.int32,
    )
    source1_owner_expert_counts = metadata.owner_expert_counts

    def fake_all_gather(outputs: list[torch.Tensor], tensor: torch.Tensor) -> None:
        sources = (
            [source0_owner_counts, source1_owner_counts]
            if tensor.ndim == 1
            else [source0_owner_expert_counts, source1_owner_expert_counts]
        )
        for output, source in zip(outputs, sources):
            output.copy_(source)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    plan = prepare_owner_directed_dispatch(metadata, workspace)

    assert plan.owner_expert_base_offsets.tolist() == [[1, 1], [1, 0]]
    assert plan.aggregate_owner_expert_counts.tolist() == [[3, 2], [1, 1]]
    assert plan.aggregate_owner_expert_offsets.tolist() == [[0, 3, 5], [0, 1, 2]]
    assert plan.local_expert_counts.tolist() == [1, 1]
    assert plan.local_expert_offsets.tolist() == [0, 1, 2]
    assert workspace.rank_counts.tolist() == [1, 1]
    assert workspace.rank_offsets.tolist() == [0, 1, 2]


def test_owner_directed_dispatch_uses_aggregate_offsets_for_owner_rows() -> None:
    hidden_states = torch.tensor([[11.0, 12.0]])
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = _metadata([2, 0], [[0, 1], []], owner_expert_counts=[[1, 1], [0, 0]])
    workspace = _workspace(world_size=2, rank=0, hidden_size=2)
    dispatch_plan = EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros((2,), dtype=torch.int32),
        owner_expert_base_offsets=torch.tensor(
            [[2, 1], [0, 0]],
            dtype=torch.int32,
        ),
        aggregate_owner_expert_counts=torch.tensor(
            [[3, 2], [0, 0]],
            dtype=torch.int32,
        ),
        aggregate_owner_expert_offsets=torch.tensor(
            [[0, 3, 5], [0, 0, 0]],
            dtype=torch.int32,
        ),
        local_expert_counts=torch.tensor([3, 2], dtype=torch.int32),
        local_expert_offsets=torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    workspace.dispatch_buffer.zero_()

    step = owner_directed_dispatch(
        hidden_states,
        topk_ids,
        metadata,
        workspace,
        dispatch_plan=dispatch_plan,
    )

    expected = torch.zeros((5, 2), dtype=torch.float32)
    expected[2] = hidden_states[0]
    expected[4] = hidden_states[0]
    torch.testing.assert_close(step.dispatch_buffer, expected)


def test_owner_directed_dispatch_rejects_shape_mismatch() -> None:
    hidden_states = torch.empty((2, 4), dtype=torch.float32)
    topk_ids = torch.empty((3, 2), dtype=torch.int32)
    metadata = _metadata([0], [[]])
    workspace = _workspace(world_size=1, rank=0, hidden_size=4)

    with pytest.raises(EPWorkspaceError, match="topk_ids rows"):
        owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)


@pytest.mark.parametrize("required_world_size", [2, 4, 8])
def test_iris_owner_directed_dispatch_rank_filled_tensor(
    required_world_size: int,
) -> None:
    iris = pytest.importorskip("iris")
    if not dist.is_available() or not dist.is_initialized():
        pytest.skip("Iris dispatch test requires torch.distributed")
    if dist.get_world_size() != required_world_size:
        pytest.skip(f"requires world_size={required_world_size}")

    ctx = iris.iris(heap_size=1 << 27)
    try:
        rank = ctx.get_rank()
        world_size = ctx.get_num_ranks()
        hidden_size = 4
        hidden_states = torch.empty(
            (world_size, hidden_size),
            dtype=torch.float32,
            device=ctx.get_device(),
        )
        for token in range(world_size):
            hidden_states[token].fill_(rank * 100 + token)
        topk_ids = torch.arange(
            world_size,
            dtype=torch.int32,
            device=ctx.get_device(),
        ).reshape(world_size, 1)
        metadata = _metadata(
            [1 for _ in range(world_size)],
            [[owner] for owner in range(world_size)],
            device=ctx.get_device(),
        )
        workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=1,
            hidden_size=hidden_size,
            top_k=1,
            dtype=torch.float32,
            iris_mode="required",
            iris_context=ctx,
        )

        step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

        expected = torch.empty_like(step.dispatch_buffer)
        for source_rank in range(world_size):
            expected[source_rank].fill_(source_rank * 100 + rank)
        torch.testing.assert_close(step.dispatch_buffer, expected)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()


def test_iris_owner_directed_dispatch_one_rank_compile_smoke() -> None:
    iris = pytest.importorskip("iris")
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for Iris dispatch compile smoke")
    initialized_here = _ensure_one_rank_dist_initialized()
    if dist.get_world_size() != 1:
        pytest.skip("one-rank compile smoke requires world_size=1")

    ctx = iris.iris(heap_size=1 << 26)
    try:
        device = ctx.get_device()
        hidden_states = torch.arange(
            12,
            dtype=torch.float32,
            device=device,
        ).reshape(3, 4)
        topk_ids = torch.tensor(
            [[0, 0], [0, -1], [0, 0]],
            dtype=torch.int32,
            device=device,
        )
        metadata = _metadata([5], [[0, 1, 2, 4, 5]], device=device)
        workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=3,
            hidden_size=4,
            top_k=2,
            dtype=torch.float32,
            iris_mode="required",
            iris_context=ctx,
        )

        step = owner_directed_dispatch(hidden_states, topk_ids, metadata, workspace)

        expected_tokens = torch.tensor([0, 0, 1, 2, 2], device=device)
        torch.testing.assert_close(step.dispatch_buffer, hidden_states[expected_tokens])
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
