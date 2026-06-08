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
            elif rank != ctx_rank:
                raise EPWorkspaceError(
                    f"rank {rank} does not match Iris rank {ctx_rank}"
                )
            if world_size is None:
                world_size = ctx_world_size
            elif world_size != ctx_world_size:
                raise EPWorkspaceError(
                    f"world_size {world_size} does not match Iris world size "
                    f"{ctx_world_size}"
                )
            device = torch.device(iris_context.get_device())
        else:
            backend = "torch"
            if world_size is None:
                world_size = 1
            if rank is None:
                rank = 0
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
        handle = EPWorkspaceHandle(
            backend=backend,
            rank=rank,
            world_size=world_size,
            heap_bases=_maybe_call(iris_context, "get_heap_bases"),
            device_context=_maybe_call(iris_context, "get_device_context"),
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
