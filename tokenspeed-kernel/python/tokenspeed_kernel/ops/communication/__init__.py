"""Public communication-kernel interfaces."""

from __future__ import annotations

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.trtllm import (
    allgather_dual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_lane_latent_norm as _allreduce_lane_latent_norm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_residual_attnres_combine as _trtllm_allreduce_residual_attnres_combine,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_residual_rmsnorm as _trtllm_allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    reducescatter_residual_rmsnorm,
)
from tokenspeed_kernel.platform import current_platform

_ALLREDUCE_FUSION_LANE: torch.Tensor | None = None


def allreduce_fusion_lane(
    like: torch.Tensor,
    width: int,
    *,
    enabled: bool = True,
) -> torch.Tensor | None:
    """Return a persistent one-row lane when fused all-reduce can use it.

    Args:
        like: Tensor providing the row count, dtype, and device.
        width: Width of the fused reduction lane.
        enabled: Whether the caller prepared fused all-reduce support.

    Returns:
        A zero-initialized ``[1, width]`` lane, or ``None`` when this invocation
        should use the ordinary reduction path.
    """

    if not enabled or like.ndim != 2 or like.shape[0] != 1:
        return None
    global _ALLREDUCE_FUSION_LANE
    lane = _ALLREDUCE_FUSION_LANE
    if (
        lane is None
        or lane.dtype != like.dtype
        or lane.device != like.device
        or lane.shape != (1, width)
    ):
        lane = torch.zeros(1, width, dtype=like.dtype, device=like.device)
        _ALLREDUCE_FUSION_LANE = lane
    return lane


def allreduce_lane_latent_norm_supported(
    lane: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    max_token_num: int,
    prepared: bool,
) -> bool:
    """Return whether this invocation can use the fused lane-norm epilogue."""

    return (
        current_platform().is_nvidia
        and prepared
        and group.size() > 1
        and lane.ndim == 2
        and 0 < lane.shape[0] <= max_token_num
    )


def prepare_allreduce_fusion(
    *,
    rank: int,
    group: dist.ProcessGroup,
    max_token_num: int,
    hidden_dim: int,
    use_fp32_lamport: bool = False,
) -> bool:
    """Prepare the selected fused-all-reduce implementation for graph capture."""

    if not current_platform().is_nvidia:
        return False
    from tokenspeed_kernel.ops.communication.trtllm import (
        ensure_workspace_initialized,
    )

    return bool(
        ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=use_fp32_lamport,
        )
    )


def allreduce_lane_latent_norm(
    lane: torch.Tensor,
    gamma: torch.Tensor,
    latent_width: int,
    *,
    rank: int,
    group: dist.ProcessGroup,
    eps: float,
    max_token_num: int,
    launch_with_pdl: bool = False,
    trigger_completion_at_end: bool = False,
) -> torch.Tensor:
    """Reduce a routed/shared lane and normalize its routed prefix."""

    return _allreduce_lane_latent_norm(
        lane,
        gamma,
        latent_width,
        rank=rank,
        group=group,
        eps=eps,
        max_token_num=max_token_num,
        launch_with_pdl=launch_with_pdl,
        trigger_completion_at_end=trigger_completion_at_end,
    )


def allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    rank: int,
    group: dist.ProcessGroup,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    residual_reduce_scattered: bool = False,
    has_partial_norm_out: bool = False,
    max_sm_to_use: int | None = None,
    launch_with_pdl: bool = False,
) -> tuple:
    """Run the platform implementation of all-reduce + residual + RMSNorm.

    Args:
        input_tensor: Per-rank partial to reduce.
        residual: Residual tensor added after reduction.
        weight: RMSNorm weight for the reduced residual.
        rank: Rank within ``group``.
        group: Process group participating in the all-reduce.
        eps: RMSNorm epsilon.
        max_token_num: Prepared fused-kernel token capacity.
        trigger_completion_at_end: Complete the collective epoch in this call.
        fp32_acc: Accumulate the fused reduction in FP32 when supported.
        block_quant_fp8: Quantize the normalized output to FP8 when supported.
        residual_reduce_scattered: Treat the residual as reduce-scattered.
        has_partial_norm_out: Also return this rank's normalized token slice.
        max_sm_to_use: Optional fused-kernel SM limit.
        launch_with_pdl: Enable programmatic dependent launch when supported.

    Returns:
        Normalized output, updated residual, optional FP8 scales, and optional
        rank-local normalized output. Unsupported platform kernels return a
        tuple whose first element is ``None``.
    """

    platform = current_platform()
    if platform.is_amd:
        from tokenspeed_kernel.ops.communication.triton import (
            allreduce_residual_rmsnorm as triton_allreduce_residual_rmsnorm,
        )

        return triton_allreduce_residual_rmsnorm(
            input_tensor=input_tensor,
            residual=residual,
            weight=weight,
            rank=rank,
            group=group,
            eps=eps,
            max_token_num=max_token_num,
            trigger_completion_at_end=trigger_completion_at_end,
            fp32_acc=fp32_acc,
            block_quant_fp8=block_quant_fp8,
            residual_reduce_scattered=residual_reduce_scattered,
            has_partial_norm_out=has_partial_norm_out,
            max_sm_to_use=max_sm_to_use,
            launch_with_pdl=launch_with_pdl,
        )
    if (
        not platform.is_nvidia
        or group.size() <= 1
        or input_tensor.ndim != 2
        or input_tensor.shape[0] > max_token_num
    ):
        return None, None, None, None

    from tokenspeed_kernel.ops.communication.trtllm import (
        ensure_workspace_initialized,
    )

    if not ensure_workspace_initialized(
        rank=rank,
        group=group,
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        return None, None, None, None
    return _trtllm_allreduce_residual_rmsnorm(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        rank=rank,
        group=group,
        eps=eps,
        max_token_num=max_token_num,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        block_quant_fp8=block_quant_fp8,
        residual_reduce_scattered=residual_reduce_scattered,
        has_partial_norm_out=has_partial_norm_out,
        max_sm_to_use=max_sm_to_use,
        launch_with_pdl=launch_with_pdl,
    )


def allreduce_residual_attnres_combine_supported(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rank: int,
    group: dist.ProcessGroup,
    local_world_size: int,
    max_token_num: int,
    enabled: bool,
    prepared: bool,
) -> bool:
    """Return whether the platform can fuse the AttnRes all-reduce epilogue.

    ``enabled`` carries caller policy, while ``prepared`` records
    whether a backend requiring explicit graph-safe setup completed it.

    Args:
        input_tensor: Per-rank attention projection partial.
        residual: Running AttnRes residual stream.
        score_weight: Precombined AttnRes score weight used by AMD.
        output_weight: Output RMSNorm weight.
        scratch: AttnRes max, denominator, and weighted-sum partials.
        rank: Rank within ``group``.
        group: Process group participating in the all-reduce.
        local_world_size: Number of processes on each node.
        max_token_num: Prepared fused-kernel token capacity.
        enabled: Whether caller policy permits a specialized collective.
        prepared: Whether explicit backend setup completed.

    Returns:
        Whether the selected platform implementation supports the call.
    """

    if not enabled:
        return False
    platform = current_platform()
    if platform.is_amd:
        from tokenspeed_kernel.ops.communication.triton import (
            allreduce_residual_attnres_combine_supported as triton_supported,
        )

        return triton_supported(
            input_tensor,
            residual,
            score_weight,
            output_weight,
            scratch,
            rank=rank,
            group=group,
            local_world_size=local_world_size,
        )
    return (
        platform.is_nvidia
        and prepared
        and group.size() > 1
        and input_tensor.ndim == 2
        and input_tensor.shape[0] <= max_token_num
    )


def allreduce_residual_attnres_combine(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rank: int,
    group: dist.ProcessGroup,
    local_world_size: int,
    eps: float,
    max_token_num: int,
    enabled: bool,
    prepared: bool,
    launch_with_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the platform implementation of all-reduce + AttnRes combine.

    NVIDIA consumes separate AttnRes projection and RMS weights. AMD consumes
    their precombined ``score_weight``. The common interface carries both so
    model code does not select a platform implementation.

    Args:
        input_tensor: Per-rank attention projection partial.
        residual: Running AttnRes residual stream.
        res_weight: AttnRes projection weight used by NVIDIA.
        rms_weight: AttnRes RMS weight used by NVIDIA.
        score_weight: Precombined AttnRes score weight used by AMD.
        output_weight: Output RMSNorm weight.
        scratch: AttnRes max, denominator, and weighted-sum partials.
        rank: Rank within ``group``.
        group: Process group participating in the all-reduce.
        local_world_size: Number of processes on each node.
        eps: AttnRes and output RMSNorm epsilon.
        max_token_num: Prepared fused-kernel token capacity.
        enabled: Whether caller policy permits a specialized collective.
        prepared: Whether explicit backend setup completed.
        launch_with_pdl: Enable programmatic dependent launch when supported.

    Returns:
        The completed AttnRes hidden state and accumulated residual.
    """

    assert allreduce_residual_attnres_combine_supported(
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        rank=rank,
        group=group,
        local_world_size=local_world_size,
        max_token_num=max_token_num,
        enabled=enabled,
        prepared=prepared,
    )
    if current_platform().is_amd:
        from tokenspeed_kernel.ops.communication.triton import (
            allreduce_residual_attnres_combine as triton_attnres_combine,
        )

        return triton_attnres_combine(
            input_tensor,
            residual,
            score_weight,
            output_weight,
            scratch,
            rank=rank,
            group=group,
            local_world_size=local_world_size,
            eps=eps,
        )
    return _trtllm_allreduce_residual_attnres_combine(
        input_tensor,
        residual,
        res_weight,
        rms_weight,
        output_weight,
        scratch=scratch,
        rank=rank,
        group=group,
        eps=eps,
        max_token_num=max_token_num,
        launch_with_pdl=launch_with_pdl,
    )


__all__ = [
    "allgather_dual_rmsnorm",
    "allreduce_fusion_lane",
    "allreduce_lane_latent_norm",
    "allreduce_lane_latent_norm_supported",
    "allreduce_residual_attnres_combine",
    "allreduce_residual_attnres_combine_supported",
    "allreduce_residual_rmsnorm",
    "prepare_allreduce_fusion",
    "reducescatter_residual_rmsnorm",
]
