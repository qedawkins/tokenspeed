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

"""Internal expert-parallel communication workspace.

This module intentionally stays below the MoE backend layer. Public MoE
contracts continue to see tensors and scalar sizes; Iris-specific state is
owned here and exposed to later EP kernels only through an opaque internal
handle made of tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist

EPWorkspaceBackend = Literal["torch", "iris"]
EPWorkspaceIrisMode = Literal["auto", "required", "disabled"]


class EPWorkspaceError(ValueError):
    """Raised when an EP workspace cannot satisfy a requested shape or step."""


class EPWorkspaceUnavailable(RuntimeError):
    """Raised when an Iris-backed workspace is required but unavailable."""


@dataclass(frozen=True)
class EPWorkspaceHandle:
    backend: EPWorkspaceBackend
    rank: int
    world_size: int
    context_rank_start: int = 0
    context_rank_stride: int = 1
    heap_bases: torch.Tensor | None = None
    device_context: torch.Tensor | None = None


@dataclass(frozen=True)
class EPWorkspaceStep:
    dispatch_buffer: torch.Tensor
    combine_buffer: torch.Tensor
    rank_counts: torch.Tensor
    rank_offsets: torch.Tensor
    readiness_flags: torch.Tensor
    handle: EPWorkspaceHandle
    num_rows: int


@dataclass
class EPCommunicationWorkspace:
    max_tokens_per_rank: int
    top_k: int
    max_dispatch_rows: int
    hidden_size: int
    world_size: int
    rank: int
    dtype: torch.dtype
    device: torch.device
    backend: EPWorkspaceBackend
    dispatch_buffer: torch.Tensor
    combine_buffer: torch.Tensor
    rank_counts: torch.Tensor
    rank_offsets: torch.Tensor
    readiness_flags: torch.Tensor
    _handle: EPWorkspaceHandle
    _iris_context: Any | None = None

    @classmethod
    def allocate(
        cls,
        *,
        max_tokens_per_rank: int,
        hidden_size: int,
        top_k: int,
        world_size: int | None = None,
        rank: int | None = None,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        iris_mode: EPWorkspaceIrisMode = "auto",
        iris_context: Any | None = None,
        iris_heap_size: int = 1 << 30,
        context_rank_start: int | None = None,
        context_rank_stride: int = 1,
    ) -> "EPCommunicationWorkspace":
        if max_tokens_per_rank < 0:
            raise EPWorkspaceError(
                f"max_tokens_per_rank must be non-negative, got {max_tokens_per_rank}"
            )
        if hidden_size <= 0:
            raise EPWorkspaceError(f"hidden_size must be positive, got {hidden_size}")
        if top_k <= 0:
            raise EPWorkspaceError(f"top_k must be positive, got {top_k}")
        if iris_mode not in {"auto", "required", "disabled"}:
            raise EPWorkspaceError(f"invalid iris_mode {iris_mode!r}")
        if context_rank_stride <= 0:
            raise EPWorkspaceError(
                f"context_rank_stride must be positive, got {context_rank_stride}"
            )

        iris_context = _resolve_iris_context(
            iris_mode=iris_mode,
            iris_context=iris_context,
            iris_heap_size=iris_heap_size,
            device=device,
        )
        if iris_context is not None:
            backend: EPWorkspaceBackend = "iris"
            ctx_rank = int(iris_context.get_rank())
            ctx_world_size = int(iris_context.get_num_ranks())
            if rank is None:
                rank = ctx_rank
            elif rank < 0:
                raise EPWorkspaceError(
                    f"rank must be non-negative, got {rank}"
                )
            if world_size is None:
                world_size = ctx_world_size
            elif world_size <= 0 or world_size > ctx_world_size:
                raise EPWorkspaceError(
                    f"world_size {world_size} is not a valid Iris subgroup of "
                    f"{ctx_world_size}"
                )
            if rank >= world_size:
                raise EPWorkspaceError(
                    f"rank must be in [0, {world_size}), got {rank}"
                )
            if context_rank_start is None:
                context_rank_start = ctx_rank - rank * context_rank_stride
            _validate_context_rank_mapping(
                context_rank_start=context_rank_start,
                context_rank_stride=context_rank_stride,
                rank=rank,
                world_size=world_size,
                ctx_rank=ctx_rank,
                ctx_world_size=ctx_world_size,
            )
            device = torch.device(iris_context.get_device())
        else:
            backend = "torch"
            if world_size is None:
                world_size = 1
            if rank is None:
                rank = 0
            if context_rank_start is None:
                context_rank_start = 0
            device = _normalize_device(device)

        _validate_rank_world_size(rank=rank, world_size=world_size)
        max_dispatch_rows = max_tokens_per_rank * top_k * world_size
        dispatch_buffer = _empty(
            (max_dispatch_rows, hidden_size),
            dtype=dtype,
            device=device,
            iris_context=iris_context,
        )
        combine_buffer = _empty(
            (max_dispatch_rows, hidden_size),
            dtype=dtype,
            device=device,
            iris_context=iris_context,
        )
        rank_counts = _zeros(
            (world_size,),
            dtype=torch.int32,
            device=device,
            iris_context=iris_context,
        )
        rank_offsets = _zeros(
            (world_size + 1,),
            dtype=torch.int32,
            device=device,
            iris_context=iris_context,
        )
        readiness_flags = _zeros(
            (world_size,),
            dtype=torch.int32,
            device=device,
            iris_context=iris_context,
        )
        heap_bases = _maybe_call(iris_context, "get_heap_bases")
        device_context = _make_ep_device_context(
            iris_context=iris_context,
            heap_bases=heap_bases,
            rank=rank,
            world_size=world_size,
            context_rank_start=context_rank_start,
            context_rank_stride=context_rank_stride,
        )
        handle = EPWorkspaceHandle(
            backend=backend,
            rank=rank,
            world_size=world_size,
            context_rank_start=context_rank_start,
            context_rank_stride=context_rank_stride,
            heap_bases=heap_bases,
            device_context=device_context,
        )
        return cls(
            max_tokens_per_rank=max_tokens_per_rank,
            top_k=top_k,
            max_dispatch_rows=max_dispatch_rows,
            hidden_size=hidden_size,
            world_size=world_size,
            rank=rank,
            dtype=dtype,
            device=device,
            backend=backend,
            dispatch_buffer=dispatch_buffer,
            combine_buffer=combine_buffer,
            rank_counts=rank_counts,
            rank_offsets=rank_offsets,
            readiness_flags=readiness_flags,
            _handle=handle,
            _iris_context=iris_context,
        )

    @property
    def handle(self) -> EPWorkspaceHandle:
        return self._handle

    def check_capacity(self, num_rows: int, hidden_size: int | None = None) -> None:
        if num_rows < 0:
            raise EPWorkspaceError(f"num_rows must be non-negative, got {num_rows}")
        if num_rows > self.max_dispatch_rows:
            raise EPWorkspaceError(
                f"EP workspace capacity exceeded: requested {num_rows} rows, "
                f"capacity is {self.max_dispatch_rows}"
            )
        if hidden_size is not None and hidden_size != self.hidden_size:
            raise EPWorkspaceError(
                f"hidden_size mismatch: got {hidden_size}, expected {self.hidden_size}"
            )

    def validate_payload(self, tensor: torch.Tensor, *, name: str = "tensor") -> None:
        if tensor.ndim != 2:
            raise EPWorkspaceError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
        self.check_capacity(tensor.shape[0], tensor.shape[1])
        if tensor.dtype != self.dtype:
            raise EPWorkspaceError(f"{name} dtype {tensor.dtype} != {self.dtype}")
        if tensor.device != self.device:
            raise EPWorkspaceError(f"{name} device {tensor.device} != {self.device}")

    def prepare_step(
        self,
        rank_counts: torch.Tensor,
        rank_offsets: torch.Tensor,
        *,
        num_rows: int | None = None,
    ) -> EPWorkspaceStep:
        self._validate_rank_metadata(rank_counts, rank_offsets)
        rows = int(rank_offsets[-1].item()) if num_rows is None else int(num_rows)
        self.check_capacity(rows)
        self.rank_counts.copy_(rank_counts)
        self.rank_offsets.copy_(rank_offsets)
        return self.view(rows)

    def view(self, num_rows: int) -> EPWorkspaceStep:
        self.check_capacity(num_rows)
        return EPWorkspaceStep(
            dispatch_buffer=self.dispatch_buffer[:num_rows],
            combine_buffer=self.combine_buffer[:num_rows],
            rank_counts=self.rank_counts,
            rank_offsets=self.rank_offsets,
            readiness_flags=self.readiness_flags,
            handle=self._handle,
            num_rows=num_rows,
        )

    def rank_local_view(self, num_rows: int | None = None) -> EPWorkspaceStep:
        rows = int(self.rank_counts[self.rank].item()) if num_rows is None else num_rows
        return self.view(rows)

    def reset_readiness_flags(self) -> torch.Tensor:
        self.readiness_flags.zero_()
        return self.readiness_flags

    def barrier(self) -> None:
        if self._iris_context is not None:
            self._iris_context.barrier()

    def _validate_rank_metadata(
        self,
        rank_counts: torch.Tensor,
        rank_offsets: torch.Tensor,
    ) -> None:
        expected_counts = (self.world_size,)
        expected_offsets = (self.world_size + 1,)
        if rank_counts.shape != expected_counts:
            raise EPWorkspaceError(
                f"rank_counts shape {tuple(rank_counts.shape)} != {expected_counts}"
            )
        if rank_offsets.shape != expected_offsets:
            raise EPWorkspaceError(
                f"rank_offsets shape {tuple(rank_offsets.shape)} != {expected_offsets}"
            )
        for name, tensor in (
            ("rank_counts", rank_counts),
            ("rank_offsets", rank_offsets),
        ):
            if tensor.dtype != torch.int32:
                raise EPWorkspaceError(f"{name} must be torch.int32, got {tensor.dtype}")
            if tensor.device != self.device:
                raise EPWorkspaceError(
                    f"{name} device {tensor.device} != workspace device {self.device}"
                )


def _resolve_iris_context(
    *,
    iris_mode: EPWorkspaceIrisMode,
    iris_context: Any | None,
    iris_heap_size: int,
    device: torch.device | str | None,
) -> Any | None:
    if iris_mode == "disabled":
        if iris_context is not None:
            raise EPWorkspaceError("iris_context was provided with iris_mode='disabled'")
        return None
    if iris_context is not None:
        return iris_context

    if not _should_auto_create_iris(device):
        if iris_mode == "required":
            raise EPWorkspaceUnavailable(
                "Iris workspace requires CUDA/ROCm device and initialized "
                "torch.distributed process group"
            )
        return None

    try:
        _ensure_tokenspeed_iris_bridge()
        import iris
    except ImportError as exc:
        if iris_mode == "required":
            raise EPWorkspaceUnavailable("Iris is not importable") from exc
        return None

    return iris.iris(heap_size=iris_heap_size)


def _ensure_tokenspeed_iris_bridge() -> None:
    """Load Iris through TokenSpeed's Triton bridge before EP kernels compile."""

    import tokenspeed_kernel.ops.communication.iris  # noqa: F401


def _should_auto_create_iris(device: torch.device | str | None) -> bool:
    if not torch.cuda.is_available() or not dist.is_available() or not dist.is_initialized():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def _normalize_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        normalized = torch.device(device)
        if normalized.type == "cuda" and normalized.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return normalized
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _validate_rank_world_size(*, rank: int, world_size: int) -> None:
    if world_size <= 0:
        raise EPWorkspaceError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise EPWorkspaceError(f"rank must be in [0, {world_size}), got {rank}")


def _validate_context_rank_mapping(
    *,
    context_rank_start: int,
    context_rank_stride: int,
    rank: int,
    world_size: int,
    ctx_rank: int,
    ctx_world_size: int,
) -> None:
    if context_rank_start < 0:
        raise EPWorkspaceError(
            f"context_rank_start must be non-negative, got {context_rank_start}"
        )
    last_context_rank = context_rank_start + (world_size - 1) * context_rank_stride
    if last_context_rank >= ctx_world_size:
        raise EPWorkspaceError(
            "Iris subgroup ranks exceed context world size: "
            f"start={context_rank_start}, stride={context_rank_stride}, "
            f"world_size={world_size}, context_world_size={ctx_world_size}"
        )
    expected_ctx_rank = context_rank_start + rank * context_rank_stride
    if expected_ctx_rank != ctx_rank:
        raise EPWorkspaceError(
            f"rank {rank} maps to Iris rank {expected_ctx_rank}, "
            f"but context rank is {ctx_rank}"
        )


def _make_ep_device_context(
    *,
    iris_context: Any | None,
    heap_bases: torch.Tensor | None,
    rank: int,
    world_size: int,
    context_rank_start: int,
    context_rank_stride: int,
) -> torch.Tensor | None:
    if iris_context is None:
        return None
    full_device_context = _maybe_call(iris_context, "get_device_context")
    if heap_bases is None or full_device_context is None:
        return full_device_context
    full_world_size = int(heap_bases.numel())
    if (
        rank == int(iris_context.get_rank())
        and world_size == full_world_size
        and context_rank_start == 0
        and context_rank_stride == 1
    ):
        return full_device_context
    context_ranks = (
        torch.arange(world_size, device=heap_bases.device, dtype=torch.long)
        * context_rank_stride
        + context_rank_start
    )
    selected_heap_bases = heap_bases.index_select(0, context_ranks)
    device_context = full_device_context.clone()
    device_context[0] = rank
    device_context[1] = world_size
    device_context[2 : 2 + world_size] = selected_heap_bases.to(
        full_device_context.dtype
    )
    return device_context


def _empty(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    iris_context: Any | None,
) -> torch.Tensor:
    if iris_context is not None:
        return iris_context.empty(shape, dtype=dtype)
    return torch.empty(shape, dtype=dtype, device=device)


def _zeros(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    iris_context: Any | None,
) -> torch.Tensor:
    if iris_context is not None:
        return iris_context.zeros(shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def _maybe_call(obj: Any | None, method_name: str) -> torch.Tensor | None:
    if obj is None:
        return None
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    return method()


__all__ = [
    "EPCommunicationWorkspace",
    "EPWorkspaceError",
    "EPWorkspaceHandle",
    "EPWorkspaceStep",
    "EPWorkspaceUnavailable",
]
