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

"""Communication ops for distributed communication.

All ops require explicit group (tuple of ranks) and rank parameters.
Groups are looked up from pg_manager internally via comm_backend.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.distributed
from tokenspeed_kernel.ops.communication import (
    allgather_dual_rmsnorm,
)
from tokenspeed_kernel.ops.communication import (
    allreduce_lane_latent_norm as kernel_allreduce_lane_latent_norm,
)
from tokenspeed_kernel.ops.communication import (
    allreduce_lane_latent_norm_supported,
    allreduce_residual_attnres_combine,
    allreduce_residual_attnres_combine_supported,
    allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication import (
    prepare_allreduce_fusion as kernel_prepare_allreduce_fusion,
)
from tokenspeed_kernel.ops.communication import (
    reducescatter_residual_rmsnorm,
)

from tokenspeed.runtime.distributed.comm_backend import (
    CommBackend,
    Group,
    get_global_backend,
)

# Re-exported for reduce-strategy callers (e.g. kimi3_join_reduce_moe):
# tensors past the one-shot admission window always take an NCCL path.
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (  # noqa: F401
    MAX_ONESHOT_BYTES as COMM_ONESHOT_MAX_BYTES,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils.pdl import pdl_enabled


def _get_process_group(group: Group):
    return pg_manager.get_process_group("nccl", group)


# ---------------------------------------------------------------------------
# Fusion parameters
# ---------------------------------------------------------------------------


class FusionOp(IntEnum):
    """What post-communication fusion to apply."""

    NONE = 0
    # all_reduce + residual_add + RMSNorm
    RESIDUAL_RMS_NORM = 1
    # reduce_scatter + residual_add + RMSNorm
    RS_RESIDUAL_RMS_NORM = 2
    # all_gather + dual RMSNorm (for MLA)
    AG_DUAL_RMS_NORM = 3


@dataclass
class FusionParams:
    """Optional fusion context passed to fused comm_ops functions.

    Not all fields are used by every ``FusionOp``. Only the relevant
    subset is accessed.
    """

    fusion_op: FusionOp = FusionOp.NONE

    # --- For RESIDUAL_RMS_NORM / RS_RESIDUAL_RMS_NORM ---
    residual: torch.Tensor | None = None
    norm_weight: torch.Tensor | None = None
    eps: float = 1e-6

    # --- For AG_DUAL_RMS_NORM ---
    norm_weight_2: torch.Tensor | None = None
    eps_2: float = 1e-6

    # --- For reduce-scatter fusion ---
    add_in: torch.Tensor | None = None
    residual_reduce_scattered: bool = False
    has_partial_norm_out: bool = False

    # --- Shared by RESIDUAL_RMS_NORM / RS_RESIDUAL_RMS_NORM / AG_DUAL_RMS_NORM ---
    max_token_num: int = 0

    # --- For FP8 block quantization ---
    block_quant_fp8: bool = False

    # --- General ---
    total_num_tokens: int = 0
    trigger_completion_at_end: bool = False
    fp32_acc: bool = False
    max_sm_to_use: int | None = None


@dataclass(frozen=True)
class ResidualRMSNormEpilogue:
    """Operands and options for residual addition followed by RMSNorm."""

    residual: torch.Tensor
    weight: torch.Tensor
    eps: float = 1e-6
    max_token_num: int = 2048
    block_quant_fp8: bool = False
    residual_reduce_scattered: bool = False
    has_partial_norm_out: bool = False
    trigger_completion_at_end: bool = False
    fp32_acc: bool = False
    max_sm_to_use: int | None = None


@dataclass(frozen=True)
class LatentRMSNormEpilogue:
    """RMSNorm parameters for the routed prefix of a reduced latent lane."""

    weight: torch.Tensor
    latent_width: int
    eps: float
    max_token_num: int
    prepared: bool = False


@dataclass(frozen=True)
class AttnResEpilogue:
    """Operands for the Kimi-K3 AttnRes combine after an all-reduce."""

    residual: torch.Tensor
    res_weight: torch.Tensor
    rms_weight: torch.Tensor
    combined_score_weight: torch.Tensor
    output_weight: torch.Tensor
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    eps: float
    max_token_num: int
    local_world_size: int
    enabled: bool = True
    prepared: bool = False


AllReduceEpilogue = ResidualRMSNormEpilogue | LatentRMSNormEpilogue | AttnResEpilogue


# ---------------------------------------------------------------------------
# Basic primitives
# ---------------------------------------------------------------------------


def all_reduce(
    tensor: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> torch.Tensor:
    """All-reduce the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.all_reduce(tensor, group, op=op)


def all_reduce_two(
    first: torch.Tensor,
    second: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce two tensors, using a fused backend primitive when available.

    Args:
        first: First tensor to reduce.
        second: Second tensor to reduce.
        group: Global ranks participating in both reductions.
        backend: Optional communication backend override.
        op: Reduction operation.

    Returns:
        The two reduced tensors in input order.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.all_reduce_two(first, second, group, op=op)


def prepare_all_reduce_lane(
    group: Group,
    hidden_dim: int,
    backend: CommBackend | None = None,
) -> bool:
    """Prepare a backend-owned one-shot lane for a wider fused reduction."""

    if backend is None:
        backend = get_global_backend()
    # No try/except: "backend can't do it" is already the base-class default
    # (returns False), and this call is COLLECTIVE — swallowing a real error
    # on one rank while peers succeed would leave the group disagreeing on
    # the lane width. Let real failures propagate loudly.
    return backend.prepare_all_reduce_lane(group, hidden_dim)


def prepare_all_reduce_fusion(
    group: Group,
    hidden_dim: int,
    max_token_num: int,
) -> bool:
    """Prepare fused all-reduce kernels before graph capture."""

    try:
        process_group = _get_process_group(group)
        return kernel_prepare_allreduce_fusion(
            rank=process_group.rank(),
            group=process_group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
        )
    except Exception:
        return False


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value_fp32 = value.float()
    normalized = value_fp32 * torch.rsqrt(
        value_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(value.dtype)


def _residual_rmsnorm_fallback(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend,
    epilogue: ResidualRMSNormEpilogue,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if epilogue.residual_reduce_scattered:
        raise RuntimeError(
            "residual_reduce_scattered requires a fused all-reduce implementation"
        )
    reduced = backend.all_reduce(tensor, group)
    residual_fp32 = reduced.float() + epilogue.residual.float()
    residual_out = residual_fp32.to(tensor.dtype)
    normalized = residual_fp32 * torch.rsqrt(
        residual_fp32.square().mean(dim=-1, keepdim=True) + epilogue.eps
    )
    norm_out = (normalized * epilogue.weight.float()).to(tensor.dtype)

    partial_norm_out = None
    if epilogue.has_partial_norm_out:
        world_size = len(group)
        base, remainder = divmod(norm_out.shape[0], world_size)
        counts = [base + (index < remainder) for index in range(world_size)]
        start = sum(counts[:rank])
        partial_norm_out = norm_out[start : start + counts[rank]].contiguous()

    if not epilogue.block_quant_fp8:
        return norm_out, residual_out, None, partial_norm_out

    from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

    quant_out, scale_out = per_token_group_quant_fp8(
        norm_out,
        group_size=128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
    )
    return quant_out, residual_out, scale_out, partial_norm_out


def all_reduce_with_epilogue(
    tensor: torch.Tensor,
    group: Group,
    epilogue: AllReduceEpilogue,
    backend: CommBackend | None = None,
) -> torch.Tensor | tuple:
    """All-reduce ``tensor`` and apply one typed, backend-fusible epilogue.

    Platform kernels may fuse the collective and epilogue. Unsupported calls
    preserve the same operation through an ordinary all-reduce followed by the
    descriptor's epilogue.

    Args:
        tensor: Per-rank partial to all-reduce.
        group: Global ranks participating in the reduction.
        epilogue: Typed operands and options for the post-reduction operation.
        backend: Optional ordinary-collective backend used by fallbacks.

    Returns:
        The result contract defined by ``epilogue``.
    """

    if backend is None:
        backend = get_global_backend()
    process_group = _get_process_group(group)
    rank = process_group.rank()

    if isinstance(epilogue, ResidualRMSNormEpilogue):
        result = allreduce_residual_rmsnorm(
            input_tensor=tensor,
            residual=epilogue.residual,
            weight=epilogue.weight,
            rank=rank,
            group=process_group,
            eps=epilogue.eps,
            max_token_num=epilogue.max_token_num,
            block_quant_fp8=epilogue.block_quant_fp8,
            residual_reduce_scattered=epilogue.residual_reduce_scattered,
            has_partial_norm_out=epilogue.has_partial_norm_out,
            trigger_completion_at_end=epilogue.trigger_completion_at_end,
            fp32_acc=epilogue.fp32_acc,
            max_sm_to_use=epilogue.max_sm_to_use,
            launch_with_pdl=pdl_enabled(),
        )
        if result[0] is not None:
            return result
        return _residual_rmsnorm_fallback(tensor, rank, group, backend, epilogue)

    if isinstance(epilogue, LatentRMSNormEpilogue):
        if allreduce_lane_latent_norm_supported(
            tensor,
            group=process_group,
            max_token_num=epilogue.max_token_num,
            prepared=epilogue.prepared,
        ):
            return kernel_allreduce_lane_latent_norm(
                tensor,
                epilogue.weight,
                epilogue.latent_width,
                rank=rank,
                group=process_group,
                eps=epilogue.eps,
                max_token_num=epilogue.max_token_num,
                launch_with_pdl=pdl_enabled(),
                trigger_completion_at_end=True,
            )
        reduced = backend.all_reduce(tensor, group)
        return torch.cat(
            (
                _rmsnorm(
                    reduced[:, : epilogue.latent_width],
                    epilogue.weight,
                    epilogue.eps,
                ),
                reduced[:, epilogue.latent_width :],
            ),
            dim=-1,
        )

    if isinstance(epilogue, AttnResEpilogue):
        if allreduce_residual_attnres_combine_supported(
            tensor,
            epilogue.residual,
            epilogue.combined_score_weight,
            epilogue.output_weight,
            epilogue.scratch,
            rank=rank,
            group=process_group,
            local_world_size=epilogue.local_world_size,
            max_token_num=epilogue.max_token_num,
            enabled=epilogue.enabled,
            prepared=epilogue.prepared,
        ):
            return allreduce_residual_attnres_combine(
                tensor,
                epilogue.residual,
                epilogue.res_weight,
                epilogue.rms_weight,
                epilogue.combined_score_weight,
                epilogue.output_weight,
                epilogue.scratch,
                rank=rank,
                group=process_group,
                local_world_size=epilogue.local_world_size,
                eps=epilogue.eps,
                max_token_num=epilogue.max_token_num,
                enabled=epilogue.enabled,
                prepared=epilogue.prepared,
                launch_with_pdl=pdl_enabled(),
            )

        from tokenspeed_kernel.ops.activation.triton import attnres_combine

        residual_out = epilogue.residual + backend.all_reduce(tensor, group)
        hidden = attnres_combine(
            residual_out,
            epilogue.combined_score_weight,
            epilogue.output_weight,
            epilogue.eps,
            epilogue.scratch,
            torch.empty_like(residual_out),
        )
        return hidden, residual_out

    raise TypeError(f"Unsupported all-reduce epilogue: {type(epilogue).__name__}")


def all_gather(
    tensor: torch.Tensor,
    group: Group,
    dim: int = -1,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """All-gather the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.all_gather(tensor, group, dim)


def all_gather_into_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> None:
    """All-gather input into a pre-allocated output buffer."""
    if backend is None:
        backend = get_global_backend()
    backend.all_gather_into_tensor(output, input, group)


def reduce_scatter(
    tensor: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Reduce-scatter the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.reduce_scatter(tensor, group)


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> None:
    """Even-split all_to_all into a pre-allocated output buffer."""
    if backend is None:
        backend = get_global_backend()
    backend.all_to_all_single(output, input, group)


# ---------------------------------------------------------------------------
# Fused ops (comm + residual + norm)
# ---------------------------------------------------------------------------


def fused_all_reduce(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """All-reduce with optional fused residual + RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.all_reduce(tensor, group)

    if fusion_params.fusion_op == FusionOp.RESIDUAL_RMS_NORM:
        return all_reduce_with_epilogue(
            tensor,
            group,
            ResidualRMSNormEpilogue(
                residual=fusion_params.residual,
                weight=fusion_params.norm_weight,
                eps=fusion_params.eps,
                max_token_num=fusion_params.max_token_num or 2048,
                fp32_acc=fusion_params.fp32_acc,
                block_quant_fp8=fusion_params.block_quant_fp8,
                residual_reduce_scattered=fusion_params.residual_reduce_scattered,
                has_partial_norm_out=fusion_params.has_partial_norm_out,
                trigger_completion_at_end=fusion_params.trigger_completion_at_end,
                max_sm_to_use=fusion_params.max_sm_to_use,
            ),
            backend=backend,
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_all_reduce"
    )


def fused_reduce_scatter(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce-scatter with optional fused residual + RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.reduce_scatter(tensor, group)

    if fusion_params.fusion_op == FusionOp.RS_RESIDUAL_RMS_NORM:
        return reducescatter_residual_rmsnorm(
            input_tensor=tensor,
            weight=fusion_params.norm_weight,
            residual=fusion_params.residual,
            eps=fusion_params.eps,
            rank=rank,
            group=_get_process_group(group),
            add_in=fusion_params.add_in,
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            max_token_num=fusion_params.max_token_num or tensor.shape[0],
            launch_with_pdl=pdl_enabled(),
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_reduce_scatter"
    )


def fused_all_gather(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    dim: int = -1,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """All-gather with optional fused dual-RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.all_gather(tensor, group, dim)

    if fusion_params.fusion_op == FusionOp.AG_DUAL_RMS_NORM:
        return allgather_dual_rmsnorm(
            qkv=tensor,
            weight_q_a=fusion_params.norm_weight,
            eps_q=fusion_params.eps,
            weight_kv_a=fusion_params.norm_weight_2,
            eps_kv=fusion_params.eps_2,
            rank=rank,
            group=_get_process_group(group),
            total_num_tokens=fusion_params.total_num_tokens,
            max_token_num=fusion_params.max_token_num
            or max(tensor.shape[0], fusion_params.total_num_tokens),
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            launch_with_pdl=pdl_enabled(),
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_all_gather"
    )


# ---------------------------------------------------------------------------
# Token-aware ops (uneven token distribution via TritonRSAG)
# ---------------------------------------------------------------------------


def token_all_gather(
    tensor: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """All-gather with token-aware distribution (TritonRSAG).

    Args:
        scattered_num_tokens: Number of tokens on each rank in the group,
            e.g. [50, 50, 51, 49] for 4 ranks with 200 total tokens.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.token_all_gather(tensor, group, scattered_num_tokens)


def token_reduce_scatter(
    tensor: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """Reduce-scatter with token-aware distribution (TritonRSAG).

    Args:
        scattered_num_tokens: Number of tokens on each rank in the group,
            e.g. [50, 50, 51, 49] for 4 ranks with 200 total tokens.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.token_reduce_scatter(tensor, group, scattered_num_tokens)
