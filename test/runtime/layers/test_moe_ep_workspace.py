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
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import tokenspeed.runtime.layers.moe.backends.base as base_module
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)
from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec


class _DummyBackend(MoEBackend):
    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return True

    def create_layer_weights(self, layer, *, with_bias: bool = False) -> None:
        pass

    def forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        raise NotImplementedError


class _FakeIrisContext:
    def __init__(self, *, rank: int, world_size: int, device: str = "cpu") -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.barrier_count = 0
        self.heap_bases = torch.arange(world_size, dtype=torch.int64, device=device)
        self.device_context = torch.arange(world_size + 2, dtype=torch.int64, device=device)

    def empty(self, *shape, dtype=None):
        return torch.empty(_shape_tuple(shape), dtype=dtype, device=self.device)

    def zeros(self, *shape, dtype=None):
        return torch.zeros(_shape_tuple(shape), dtype=dtype, device=self.device)

    def get_rank(self):
        return self.rank

    def get_num_ranks(self):
        return self.world_size

    def get_device(self):
        return self.device

    def get_heap_bases(self):
        return self.heap_bases

    def get_device_context(self):
        return self.device_context

    def barrier(self):
        self.barrier_count += 1


def _shape_tuple(shape):
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return shape[0]
    return tuple(shape)


def _workspace(**overrides) -> EPCommunicationWorkspace:
    options = {
        "max_tokens_per_rank": 4,
        "hidden_size": 8,
        "top_k": 2,
        "world_size": 3,
        "rank": 1,
        "dtype": torch.bfloat16,
        "device": "cpu",
        "iris_mode": "disabled",
    }
    options.update(overrides)
    return EPCommunicationWorkspace.allocate(**options)


def test_ep_workspace_preallocates_and_returns_rank_views() -> None:
    workspace = _workspace()
    assert workspace.backend == "torch"
    assert workspace.max_dispatch_rows == 24
    assert workspace.dispatch_buffer.shape == (24, 8)
    assert workspace.combine_buffer.shape == (24, 8)
    assert workspace.rank_counts.shape == (3,)
    assert workspace.rank_offsets.shape == (4,)
    assert workspace.readiness_flags.shape == (3,)

    rank_counts = torch.tensor([2, 3, 1], dtype=torch.int32)
    rank_offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    step = workspace.prepare_step(rank_counts, rank_offsets)

    assert step.num_rows == 6
    assert step.dispatch_buffer.shape == (6, 8)
    assert step.combine_buffer.shape == (6, 8)
    assert torch.equal(step.rank_counts, rank_counts)
    assert torch.equal(step.rank_offsets, rank_offsets)

    local = workspace.rank_local_view()
    assert local.num_rows == 3
    assert local.dispatch_buffer.shape == (3, 8)


def test_ep_workspace_capacity_checks() -> None:
    workspace = _workspace(max_tokens_per_rank=1, top_k=2, world_size=2, rank=0)
    assert workspace.max_dispatch_rows == 4

    workspace.check_capacity(4)
    with pytest.raises(EPWorkspaceError, match="capacity exceeded"):
        workspace.check_capacity(5)

    rank_counts = torch.tensor([3, 2], dtype=torch.int32)
    rank_offsets = torch.tensor([0, 3, 5], dtype=torch.int32)
    with pytest.raises(EPWorkspaceError, match="capacity exceeded"):
        workspace.prepare_step(rank_counts, rank_offsets)


def test_ep_workspace_empty_token_step() -> None:
    workspace = _workspace(max_tokens_per_rank=0)
    step = workspace.prepare_step(
        torch.zeros((3,), dtype=torch.int32),
        torch.zeros((4,), dtype=torch.int32),
    )

    assert step.num_rows == 0
    assert step.dispatch_buffer.shape == (0, 8)
    assert step.combine_buffer.shape == (0, 8)
    workspace.readiness_flags.fill_(1)
    assert workspace.reset_readiness_flags().sum().item() == 0


def test_ep_workspace_shape_and_dtype_mismatch_errors() -> None:
    workspace = _workspace()
    rank_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    with pytest.raises(EPWorkspaceError, match="rank_counts shape"):
        workspace.prepare_step(torch.ones((2,), dtype=torch.int32), rank_offsets)
    with pytest.raises(EPWorkspaceError, match="rank_offsets shape"):
        workspace.prepare_step(
            torch.ones((3,), dtype=torch.int32),
            torch.ones((3,), dtype=torch.int32),
        )
    with pytest.raises(EPWorkspaceError, match="rank_counts must be torch.int32"):
        workspace.prepare_step(torch.ones((3,), dtype=torch.int64), rank_offsets)
    with pytest.raises(EPWorkspaceError, match="hidden_size mismatch"):
        workspace.check_capacity(1, hidden_size=7)
    with pytest.raises(EPWorkspaceError, match="payload dtype"):
        workspace.validate_payload(torch.empty((1, 8), dtype=torch.float32), name="payload")
    with pytest.raises(EPWorkspaceError, match="payload must be rank-2"):
        workspace.validate_payload(torch.empty((8,), dtype=torch.bfloat16), name="payload")


def test_ep_workspace_one_rank_self_dispatch_buffer_smoke() -> None:
    workspace = _workspace(
        max_tokens_per_rank=3,
        hidden_size=4,
        top_k=1,
        world_size=1,
        rank=0,
        dtype=torch.float32,
    )
    step = workspace.prepare_step(
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
    )
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    step.dispatch_buffer.copy_(source)
    step.combine_buffer.copy_(step.dispatch_buffer)

    assert torch.equal(step.combine_buffer, source)


def test_ep_workspace_one_rank_gpu_self_dispatch_buffer_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the one-rank EP workspace smoke test")
    device = torch.device("cuda")
    workspace = _workspace(
        max_tokens_per_rank=3,
        hidden_size=4,
        top_k=1,
        world_size=1,
        rank=0,
        dtype=torch.float32,
        device=device,
    )
    step = workspace.prepare_step(
        torch.tensor([3], dtype=torch.int32, device=device),
        torch.tensor([0, 3], dtype=torch.int32, device=device),
    )
    source = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)

    step.dispatch_buffer.copy_(source)
    step.combine_buffer.copy_(step.dispatch_buffer)

    torch.testing.assert_close(step.combine_buffer, source)


def test_ep_workspace_uses_injected_iris_context_without_exposing_iris_type() -> None:
    fake_iris = _FakeIrisContext(rank=1, world_size=2)
    workspace = EPCommunicationWorkspace.allocate(
        max_tokens_per_rank=2,
        hidden_size=4,
        top_k=2,
        dtype=torch.float32,
        iris_mode="required",
        iris_context=fake_iris,
    )

    assert workspace.backend == "iris"
    assert workspace.rank == 1
    assert workspace.world_size == 2
    assert workspace.handle.backend == "iris"
    assert workspace.handle.context_rank_start == 0
    assert workspace.handle.context_rank_stride == 1
    assert workspace.handle.heap_bases is fake_iris.heap_bases
    assert workspace.handle.device_context is fake_iris.device_context
    workspace.barrier()
    assert fake_iris.barrier_count == 1


def test_ep_workspace_supports_tp_strided_iris_subgroup() -> None:
    fake_iris = _FakeIrisContext(rank=3, world_size=4)
    workspace = EPCommunicationWorkspace.allocate(
        max_tokens_per_rank=2,
        hidden_size=4,
        top_k=2,
        world_size=2,
        rank=1,
        dtype=torch.float32,
        iris_mode="required",
        iris_context=fake_iris,
        context_rank_stride=2,
    )

    assert workspace.backend == "iris"
    assert workspace.rank == 1
    assert workspace.world_size == 2
    assert workspace.handle.context_rank_start == 1
    assert workspace.handle.context_rank_stride == 2
    assert workspace.handle.device_context is not None
    assert workspace.handle.device_context[:2].tolist() == [1, 2]
    assert torch.equal(
        workspace.handle.device_context[2:4],
        fake_iris.heap_bases[[1, 3]],
    )


def test_ep_workspace_rejects_invalid_iris_subgroup_mapping() -> None:
    fake_iris = _FakeIrisContext(rank=2, world_size=4)

    with pytest.raises(EPWorkspaceError, match="maps to Iris rank"):
        EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=2,
            hidden_size=4,
            top_k=2,
            world_size=2,
            rank=1,
            dtype=torch.float32,
            iris_mode="required",
            iris_context=fake_iris,
            context_rank_start=1,
            context_rank_stride=2,
        )


def test_moe_backend_reuses_ep_workspace_until_capacity_grows() -> None:
    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=4,
        hidden_size=16,
        intermediate_size=32,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=1,
        ep_size=2,
    )
    backend = _DummyBackend(
        key=BackendKey(arch="any", quant="test", impl="dummy"),
        spec=spec,
        quant_config=None,
    )

    first = backend.ensure_ep_workspace(
        max_tokens_per_rank=4,
        dtype=torch.float32,
        device="cpu",
        iris_mode="disabled",
    )
    second = backend.ensure_ep_workspace(
        max_tokens_per_rank=2,
        dtype=torch.float32,
        device="cpu",
        iris_mode="disabled",
    )
    larger = backend.ensure_ep_workspace(
        max_tokens_per_rank=5,
        dtype=torch.float32,
        device="cpu",
        iris_mode="disabled",
    )

    assert first is second
    assert larger is not first
    assert larger.max_dispatch_rows == 20
    assert larger.rank == spec.ep_rank
    assert larger.world_size == spec.ep_size


def test_moe_backend_reuses_shared_auto_iris_workspace_across_instances(
    monkeypatch,
) -> None:
    base_module._SHARED_IRIS_EP_WORKSPACES.clear()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=4,
        hidden_size=16,
        intermediate_size=32,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=1,
        ep_size=2,
    )
    first_backend = _DummyBackend(
        key=BackendKey(arch="any", quant="test", impl="dummy"),
        spec=spec,
        quant_config=None,
    )
    second_backend = _DummyBackend(
        key=BackendKey(arch="any", quant="test", impl="dummy"),
        spec=spec,
        quant_config=None,
    )
    calls = []

    def fake_allocate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            backend="iris",
            max_dispatch_rows=(
                kwargs["max_tokens_per_rank"]
                * kwargs["top_k"]
                * kwargs["world_size"]
            ),
            hidden_size=kwargs["hidden_size"],
            top_k=kwargs["top_k"],
            world_size=kwargs["world_size"],
            rank=kwargs["rank"],
            dtype=kwargs["dtype"],
            device=torch.device("cuda:0"),
            handle=SimpleNamespace(
                context_rank_start=kwargs.get("context_rank_start") or 0,
                context_rank_stride=kwargs.get("context_rank_stride", 1),
            ),
        )

    monkeypatch.setattr(
        EPCommunicationWorkspace,
        "allocate",
        staticmethod(fake_allocate),
    )

    try:
        first = first_backend.ensure_ep_workspace(
            max_tokens_per_rank=4,
            dtype=torch.float32,
            device="cuda:0",
            iris_mode="auto",
        )
        second = second_backend.ensure_ep_workspace(
            max_tokens_per_rank=2,
            dtype=torch.float32,
            device="cuda:0",
            iris_mode="auto",
        )
        larger = second_backend.ensure_ep_workspace(
            max_tokens_per_rank=5,
            dtype=torch.float32,
            device="cuda:0",
            iris_mode="auto",
        )
    finally:
        base_module._SHARED_IRIS_EP_WORKSPACES.clear()

    assert first is second
    assert larger is not first
    assert larger.max_dispatch_rows == 20
    assert len(calls) == 2


def test_moe_backend_does_not_cache_non_iris_auto_workspace(monkeypatch) -> None:
    base_module._SHARED_IRIS_EP_WORKSPACES.clear()
    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=4,
        hidden_size=16,
        intermediate_size=32,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=1,
        ep_size=2,
    )
    first_backend = _DummyBackend(
        key=BackendKey(arch="any", quant="test", impl="dummy"),
        spec=spec,
        quant_config=None,
    )
    second_backend = _DummyBackend(
        key=BackendKey(arch="any", quant="test", impl="dummy"),
        spec=spec,
        quant_config=None,
    )
    calls = []

    def fake_allocate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            backend="torch",
            max_dispatch_rows=(
                kwargs["max_tokens_per_rank"]
                * kwargs["top_k"]
                * kwargs["world_size"]
            ),
            hidden_size=kwargs["hidden_size"],
            top_k=kwargs["top_k"],
            world_size=kwargs["world_size"],
            rank=kwargs["rank"],
            dtype=kwargs["dtype"],
            device=torch.device("cuda:0"),
            handle=SimpleNamespace(
                context_rank_start=kwargs.get("context_rank_start") or 0,
                context_rank_stride=kwargs.get("context_rank_stride", 1),
            ),
        )

    monkeypatch.setattr(
        EPCommunicationWorkspace,
        "allocate",
        staticmethod(fake_allocate),
    )

    try:
        first = first_backend.ensure_ep_workspace(
            max_tokens_per_rank=4,
            dtype=torch.float32,
            device="cuda:0",
            iris_mode="auto",
        )
        second = second_backend.ensure_ep_workspace(
            max_tokens_per_rank=2,
            dtype=torch.float32,
            device="cuda:0",
            iris_mode="auto",
        )
    finally:
        base_module._SHARED_IRIS_EP_WORKSPACES.clear()

    assert first is not second
    assert len(calls) == 2


@pytest.mark.parametrize("required_world_size", [2, 4, 8])
def test_iris_workspace_can_exchange_rank_filled_tensor(required_world_size: int) -> None:
    iris = pytest.importorskip("iris")
    if not dist.is_available() or not dist.is_initialized():
        pytest.skip("Iris multi-rank workspace test requires torch.distributed")
    if dist.get_world_size() != required_world_size:
        pytest.skip(f"requires world_size={required_world_size}")

    from iris.ccl import Config

    ctx = iris.iris(heap_size=1 << 26)
    try:
        rank = ctx.get_rank()
        world_size = ctx.get_num_ranks()
        workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=1,
            hidden_size=world_size,
            top_k=1,
            dtype=torch.float32,
            iris_mode="required",
            iris_context=ctx,
        )
        step = workspace.view(1)
        step.dispatch_buffer.fill_(float(rank))
        step.combine_buffer.zero_()

        ctx.ccl.all_to_all(
            step.combine_buffer,
            step.dispatch_buffer,
            config=Config(use_gluon=True),
        )
        expected = torch.arange(
            world_size,
            device=step.combine_buffer.device,
            dtype=torch.float32,
        ).reshape(1, world_size)
        torch.testing.assert_close(step.combine_buffer, expected)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()
