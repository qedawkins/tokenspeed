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

"""Internal pre-routed fused EP down GEMM plus weighted combine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.layers.moe.backends.ep_combine import owner_directed_combine
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import EPOwnerDispatchPlan
from tokenspeed.runtime.layers.moe.backends.ep_experts import (
    build_owner_expert_metadata,
    ep_expert_config,
    owner_rank_fp8_expert_gemm,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    PreRoutedFusedEPMetadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_reduce import ep_weighted_reduce
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
    EPWorkspaceUnavailable,
)


try:
    import triton
    from iris.gluon import IrisDeviceCtx
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    triton = None
    IrisDeviceCtx = None
    gluon = None
    gl = None

if gluon is None:
    _pre_routed_fused_down_combine_kernel = None


@dataclass(frozen=True)
class PreRoutedFusedDownCombineResult:
    output: torch.Tensor
    dispatch_plan: EPOwnerDispatchPlan


def pre_routed_fused_down_combine(
    owner_intermediate: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    fused_metadata: PreRoutedFusedEPMetadata,
    local_down_weight: torch.Tensor,
    local_down_weight_scale: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    block_size: int = 16,
    routed_scaling_factor: float = 1.0,
    out: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
    expected_gemm_kernel_name: str | None = None,
    expected_reduce_kernel_name: str | None = None,
) -> PreRoutedFusedDownCombineResult:
    _validate_pre_routed_inputs(
        owner_intermediate,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        fused_metadata,
    )
    dispatch_plan = _dispatch_plan_from_fused_metadata(fused_metadata)
    _validate_owner_intermediate_rows(owner_intermediate, dispatch_plan)
    if workspace.backend == "iris":
        return _pre_routed_fused_down_combine_with_iris_gluon(
            owner_intermediate,
            topk_ids,
            topk_weights,
            workspace,
            fused_metadata,
            local_down_weight,
            local_down_weight_scale,
            dispatch_plan,
            block_shape=block_shape,
            block_size=block_size,
            routed_scaling_factor=routed_scaling_factor,
            out=out,
            config=config,
        )

    owner_outputs = owner_rank_fp8_expert_gemm(
        owner_intermediate,
        local_down_weight,
        local_down_weight_scale,
        dispatch_plan.local_expert_counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
        expected_kernel_name=expected_gemm_kernel_name,
    )
    workspace.combine_buffer[: owner_outputs.shape[0]].copy_(owner_outputs)
    returned_slots = owner_directed_combine(
        workspace.combine_buffer,
        topk_ids,
        ep_metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )
    output = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        out=out,
        expected_kernel_name=expected_reduce_kernel_name,
    )
    return PreRoutedFusedDownCombineResult(
        output=output,
        dispatch_plan=dispatch_plan,
    )


def self_routing_fused_down_combine(
    owner_intermediate: torch.Tensor,
    gate_up_result: Any,
    workspace: EPCommunicationWorkspace,
    local_down_weight: torch.Tensor,
    local_down_weight_scale: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    block_size: int = 16,
    routed_scaling_factor: float = 1.0,
    out: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
    expected_gemm_kernel_name: str | None = None,
    expected_reduce_kernel_name: str | None = None,
) -> PreRoutedFusedDownCombineResult:
    """Reuse the pre-routed fused down+combine path for self-routing output."""

    _validate_self_routing_gate_up_result(gate_up_result)
    topk_output = gate_up_result.topk_output
    topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_output.topk_weights
    return pre_routed_fused_down_combine(
        owner_intermediate,
        topk_ids,
        topk_weights,
        gate_up_result.ep_metadata,
        workspace,
        gate_up_result.fused_metadata,
        local_down_weight,
        local_down_weight_scale,
        expert_owner,
        local_expert_id,
        block_shape=block_shape,
        block_size=block_size,
        routed_scaling_factor=routed_scaling_factor,
        out=out,
        config=config,
        expected_gemm_kernel_name=expected_gemm_kernel_name,
        expected_reduce_kernel_name=expected_reduce_kernel_name,
    )


def _dispatch_plan_from_fused_metadata(
    fused_metadata: PreRoutedFusedEPMetadata,
) -> EPOwnerDispatchPlan:
    return EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros_like(fused_metadata.owner_counts),
        owner_expert_base_offsets=fused_metadata.owner_expert_base_offsets,
        aggregate_owner_expert_counts=fused_metadata.aggregate_owner_expert_counts,
        aggregate_owner_expert_offsets=fused_metadata.aggregate_owner_expert_offsets,
        local_expert_counts=fused_metadata.local_expert_counts,
        local_expert_offsets=fused_metadata.local_expert_offsets,
    )


if gluon is not None:

    @gluon.jit
    def _pre_routed_fused_down_combine_kernel(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        owner_intermediate_ptr,
        accum_out_ptr,
        row_source_rank_ptr,
        row_source_token_ptr,
        row_source_slot_ptr,
        row_weight_ptr,
        row_ready_ptr,
        source_owner_ranks_ptr,
        source_owner_rows_ptr,
        source_token_ids_ptr,
        source_topk_slots_ptr,
        source_weights_ptr,
        weight_ptr,
        weight_scale_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        INTERMEDIATE_STRIDE_M: gl.constexpr,
        ACCUM_STRIDE_M: gl.constexpr,
        WEIGHT_STRIDE_E: gl.constexpr,
        WEIGHT_STRIDE_N: gl.constexpr,
        WEIGHT_STRIDE_K: gl.constexpr,
        WEIGHT_SCALE_STRIDE_E: gl.constexpr,
        WEIGHT_SCALE_STRIDE_N: gl.constexpr,
        WEIGHT_SCALE_STRIDE_K: gl.constexpr,
        WORKSPACE_RANK: gl.constexpr,
        WORLD_SIZE: gl.constexpr,
        MAX_TOKENS_PER_RANK: gl.constexpr,
        NUM_SOURCE_SLOTS: gl.constexpr,
        MAX_DISPATCH_ROWS: gl.constexpr,
        NUM_LOCAL_ROWS: gl.constexpr,
        HIDDEN_SIZE: gl.constexpr,
        INTERMEDIATE_SIZE: gl.constexpr,
        NUM_PROGRAMS: gl.constexpr,
        TOTAL_GEMM_TILES: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        WEIGHT_BLOCK_N: gl.constexpr,
        WEIGHT_BLOCK_K: gl.constexpr,
        GEMM_BLOCK_K: gl.constexpr,
        GEMM_BLOCK_K_ELEMS_PER_THREAD: gl.constexpr,
        ROUTED_SCALING_FACTOR: gl.constexpr,
    ):
        ctx = IrisDeviceCtx.initialize(context_tensor)
        pid = gl.program_id(0)

        for flat_slot in range(pid, NUM_SOURCE_SLOTS, NUM_PROGRAMS):
            owner = gl.load(source_owner_ranks_ptr + flat_slot).to(gl.int32)
            owner_row = gl.load(source_owner_rows_ptr + flat_slot).to(gl.int32)
            valid = (
                (owner >= 0)
                & (owner < WORLD_SIZE)
                & (owner_row >= 0)
                & (owner_row < MAX_DISPATCH_ROWS)
            )
            if valid:
                token = gl.load(source_token_ids_ptr + flat_slot).to(gl.int32)
                slot = gl.load(source_topk_slots_ptr + flat_slot).to(gl.int32)
                weight = gl.load(source_weights_ptr + flat_slot).to(gl.float32)
                ctx.store(row_source_rank_ptr + owner_row, WORKSPACE_RANK, owner)
                ctx.store(row_source_token_ptr + owner_row, token, owner)
                ctx.store(row_source_slot_ptr + owner_row, slot, owner)
                ctx.store(row_weight_ptr + owner_row, weight, owner)
                ctx.atomic_cas(
                    row_ready_ptr + owner_row,
                    0,
                    1,
                    owner,
                    sem="release",
                    scope="sys",
                )

        for gemm_tile in range(pid, TOTAL_GEMM_TILES, NUM_PROGRAMS):
            sorted_row = gemm_tile // HIDDEN_SIZE
            out_n = gemm_tile - sorted_row * HIDDEN_SIZE
            num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr).to(gl.int32)
            row_active = sorted_row < num_tokens_post_padded
            sorted_slot = gl.load(
                sorted_token_ids_ptr + sorted_row,
                mask=row_active,
                other=NUM_LOCAL_ROWS,
            ).to(gl.int32)
            slot_valid = sorted_slot < NUM_LOCAL_ROWS
            expert_id = gl.load(
                expert_ids_ptr + sorted_row // BLOCK_SIZE_M,
                mask=row_active,
                other=-1,
            ).to(gl.int32)
            expert_valid = expert_id >= 0
            source_row = sorted_slot

            if row_active & slot_valid & expert_valid:
                done = gl.full((), 0, dtype=gl.int32)
                while done != 1:
                    done = ctx.atomic_cas(
                        row_ready_ptr + source_row,
                        1,
                        1,
                        WORKSPACE_RANK,
                        sem="acquire",
                        scope="sys",
                    ).to(gl.int32)

            target_rank = gl.load(
                row_source_rank_ptr + source_row,
                mask=slot_valid,
                other=WORKSPACE_RANK,
            ).to(gl.int32)
            target_token = gl.load(
                row_source_token_ptr + source_row,
                mask=slot_valid,
                other=0,
            ).to(gl.int32)
            routed_weight = gl.load(
                row_weight_ptr + source_row,
                mask=slot_valid,
                other=0.0,
            ).to(gl.float32)

            gemm_layout: gl.constexpr = gl.BlockedLayout(
                [GEMM_BLOCK_K_ELEMS_PER_THREAD], [64], [1], [0]
            )
            offs_k = gl.arange(0, GEMM_BLOCK_K, layout=gemm_layout)
            k_valid = offs_k < INTERMEDIATE_SIZE
            load_valid = row_active & slot_valid & expert_valid & k_valid
            activations = gl.load(
                owner_intermediate_ptr
                + source_row * INTERMEDIATE_STRIDE_M
                + offs_k,
                mask=load_valid,
                other=0.0,
            ).to(gl.float32)
            weight = gl.load(
                weight_ptr
                + expert_id * WEIGHT_STRIDE_E
                + out_n * WEIGHT_STRIDE_N
                + offs_k * WEIGHT_STRIDE_K,
                mask=load_valid,
                other=0.0,
            ).to(gl.float32)
            scale = gl.load(
                weight_scale_ptr
                + expert_id * WEIGHT_SCALE_STRIDE_E
                + (out_n // WEIGHT_BLOCK_N) * WEIGHT_SCALE_STRIDE_N
                + (offs_k // WEIGHT_BLOCK_K) * WEIGHT_SCALE_STRIDE_K,
                mask=load_valid,
                other=0.0,
            ).to(gl.float32)
            acc = gl.sum(activations * weight * scale, axis=0)
            acc *= routed_weight * ROUTED_SCALING_FACTOR
            output_valid = (
                row_active
                & slot_valid
                & expert_valid
                & (target_rank >= 0)
                & (target_rank < WORLD_SIZE)
                & (target_token >= 0)
                & (target_token < MAX_TOKENS_PER_RANK)
            )
            ctx.atomic_add(
                accum_out_ptr + target_token * ACCUM_STRIDE_M + out_n,
                acc,
                target_rank,
                mask=output_valid,
                sem="acq_rel",
                scope="sys",
            )


def _pre_routed_fused_down_combine_with_iris_gluon(
    owner_intermediate: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    workspace: EPCommunicationWorkspace,
    fused_metadata: PreRoutedFusedEPMetadata,
    local_down_weight: torch.Tensor,
    local_down_weight_scale: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
    *,
    block_shape: tuple[int, int],
    block_size: int,
    routed_scaling_factor: float,
    out: torch.Tensor | None,
    config: dict[str, Any] | None,
) -> PreRoutedFusedDownCombineResult:
    if _pre_routed_fused_down_combine_kernel is None or IrisDeviceCtx is None:
        raise EPWorkspaceUnavailable(
            "Iris Gluon fused down+combine support is not importable"
        )
    if triton is None:
        raise EPWorkspaceUnavailable("Triton is not importable")
    if workspace.handle.device_context is None or workspace._iris_context is None:
        raise EPWorkspaceUnavailable("Iris-backed workspace has no device context")

    output_size = local_down_weight.shape[1]
    if out is None:
        out = torch.empty(
            (topk_ids.shape[0], output_size),
            dtype=owner_intermediate.dtype,
            device=owner_intermediate.device,
        )
    expert_metadata = build_owner_expert_metadata(
        dispatch_plan.local_expert_counts,
        block_size=block_size,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
    )
    _validate_down_kernel_inputs(
        owner_intermediate,
        topk_ids,
        topk_weights,
        local_down_weight,
        local_down_weight_scale,
        out,
        dispatch_plan,
        block_shape=block_shape,
    )

    if workspace.world_size == 1:
        rank_counts = torch.tensor(
            [owner_intermediate.shape[0]],
            dtype=torch.int32,
            device=workspace.device,
        )
        rank_offsets = torch.tensor(
            [0, owner_intermediate.shape[0]],
            dtype=torch.int32,
            device=workspace.device,
        )
        workspace.prepare_step(
            rank_counts,
            rank_offsets,
            num_rows=owner_intermediate.shape[0],
        )

    accum_out = workspace._iris_context.zeros(
        (workspace.max_tokens_per_rank, output_size),
        dtype=torch.float32,
    )
    row_source_rank = workspace._iris_context.full(
        (workspace.max_dispatch_rows,),
        -1,
        dtype=torch.int32,
    )
    row_source_token = workspace._iris_context.full(
        (workspace.max_dispatch_rows,),
        -1,
        dtype=torch.int32,
    )
    row_source_slot = workspace._iris_context.full(
        (workspace.max_dispatch_rows,),
        -1,
        dtype=torch.int32,
    )
    row_weight = workspace._iris_context.zeros(
        (workspace.max_dispatch_rows,),
        dtype=torch.float32,
    )
    row_ready = workspace._iris_context.zeros(
        (workspace.max_dispatch_rows,),
        dtype=torch.int32,
    )
    workspace.barrier()

    gemm_programs = expert_metadata.sorted_token_ids.numel() * output_size
    total_programs = max(fused_metadata.source_flat_slots.numel() + gemm_programs, 1)
    num_programs = min(
        torch.cuda.get_device_properties(owner_intermediate.device).multi_processor_count,
        total_programs,
    )
    if fused_metadata.source_flat_slots.numel() > 0 or gemm_programs > 0:
        if config is None:
            config = ep_expert_config(block_size=block_size, block_shape=block_shape)
        intermediate_size = owner_intermediate.shape[1]
        gemm_block_k = max(64, triton.next_power_of_2(intermediate_size))
        weight_block_n, weight_block_k = int(block_shape[0]), int(block_shape[1])
        flat_topk_weights = topk_weights.reshape(-1).contiguous()
        _pre_routed_fused_down_combine_kernel[(num_programs,)](
            IrisDeviceCtx,
            workspace.handle.device_context,
            owner_intermediate,
            accum_out,
            row_source_rank,
            row_source_token,
            row_source_slot,
            row_weight,
            row_ready,
            fused_metadata.owner_ranks,
            fused_metadata.owner_rows,
            fused_metadata.source_token_ids,
            fused_metadata.source_topk_slots,
            flat_topk_weights,
            local_down_weight,
            local_down_weight_scale,
            expert_metadata.sorted_token_ids,
            expert_metadata.expert_ids,
            expert_metadata.num_tokens_post_padded,
            owner_intermediate.stride(0),
            accum_out.stride(0),
            local_down_weight.stride(0),
            local_down_weight.stride(1),
            local_down_weight.stride(2),
            local_down_weight_scale.stride(0),
            local_down_weight_scale.stride(1),
            local_down_weight_scale.stride(2),
            workspace.rank,
            workspace.world_size,
            workspace.max_tokens_per_rank,
            fused_metadata.source_flat_slots.numel(),
            workspace.max_dispatch_rows,
            owner_intermediate.shape[0],
            output_size,
            intermediate_size,
            num_programs,
            gemm_programs,
            config["BLOCK_SIZE_M"],
            weight_block_n,
            weight_block_k,
            gemm_block_k,
            max(gemm_block_k // 64, 1),
            float(routed_scaling_factor),
            num_warps=1,
        )
    workspace.barrier()
    out.copy_(accum_out[: topk_ids.shape[0]].to(out.dtype))
    return PreRoutedFusedDownCombineResult(
        output=out,
        dispatch_plan=dispatch_plan,
    )


def _validate_pre_routed_inputs(
    owner_intermediate: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    fused_metadata: PreRoutedFusedEPMetadata,
) -> None:
    if owner_intermediate.ndim != 2:
        raise EPWorkspaceError(
            "owner_intermediate must be rank-2, got "
            f"{tuple(owner_intermediate.shape)}"
        )
    if owner_intermediate.shape[0] > workspace.max_dispatch_rows:
        raise EPWorkspaceError(
            "owner_intermediate rows exceed workspace capacity: "
            f"{owner_intermediate.shape[0]} > {workspace.max_dispatch_rows}"
        )
    if owner_intermediate.dtype != workspace.dtype:
        raise EPWorkspaceError(
            f"owner_intermediate dtype {owner_intermediate.dtype} != {workspace.dtype}"
        )
    if owner_intermediate.device != workspace.device:
        raise EPWorkspaceError(
            f"owner_intermediate device {owner_intermediate.device} != "
            f"workspace device {workspace.device}"
        )
    if topk_ids.shape != topk_weights.shape:
        raise EPWorkspaceError(
            f"topk_weights shape {tuple(topk_weights.shape)} != topk_ids shape "
            f"{tuple(topk_ids.shape)}"
        )
    if topk_ids.dtype != torch.int32:
        raise EPWorkspaceError(f"topk_ids must be torch.int32, got {topk_ids.dtype}")
    if not topk_weights.is_floating_point():
        raise EPWorkspaceError(
            f"topk_weights must be floating point, got {topk_weights.dtype}"
        )
    for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
        if tensor.device != workspace.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != workspace device "
                f"{workspace.device}"
            )
    if fused_metadata.source_rank != workspace.rank:
        raise EPWorkspaceError(
            f"fused source rank {fused_metadata.source_rank} != workspace rank "
            f"{workspace.rank}"
        )
    if fused_metadata.world_size != workspace.world_size:
        raise EPWorkspaceError(
            f"fused world size {fused_metadata.world_size} != workspace world size "
            f"{workspace.world_size}"
        )
    if fused_metadata.num_tokens != topk_ids.shape[0]:
        raise EPWorkspaceError(
            f"fused num_tokens {fused_metadata.num_tokens} != topk rows "
            f"{topk_ids.shape[0]}"
        )
    if topk_ids.shape != (fused_metadata.num_tokens, fused_metadata.top_k):
        raise EPWorkspaceError(
            f"topk_ids shape {tuple(topk_ids.shape)} != "
            f"{(fused_metadata.num_tokens, fused_metadata.top_k)}"
        )
    _require_same_metadata_tensor(
        "owner_counts",
        fused_metadata.owner_counts,
        ep_metadata.owner_counts,
        workspace,
    )
    _require_same_metadata_tensor(
        "owner_expert_counts",
        fused_metadata.owner_expert_counts,
        ep_metadata.owner_expert_counts,
        workspace,
    )
    _require_same_metadata_tensor(
        "owner_expert_offsets",
        fused_metadata.owner_expert_offsets,
        ep_metadata.owner_expert_offsets,
        workspace,
    )
    for name, tensor in (
        ("source_flat_slots", fused_metadata.source_flat_slots),
        ("source_token_ids", fused_metadata.source_token_ids),
        ("source_topk_slots", fused_metadata.source_topk_slots),
        ("owner_ranks", fused_metadata.owner_ranks),
        ("owner_rows", fused_metadata.owner_rows),
        ("owner_expert_base_offsets", fused_metadata.owner_expert_base_offsets),
        ("aggregate_owner_expert_counts", fused_metadata.aggregate_owner_expert_counts),
        (
            "aggregate_owner_expert_offsets",
            fused_metadata.aggregate_owner_expert_offsets,
        ),
        ("local_expert_counts", fused_metadata.local_expert_counts),
        ("local_expert_offsets", fused_metadata.local_expert_offsets),
    ):
        _validate_int32_device_tensor(name, tensor, workspace)
    if fused_metadata.topk_weights.device != workspace.device:
        raise EPWorkspaceError(
            "fused topk_weights device "
            f"{fused_metadata.topk_weights.device} != workspace device "
            f"{workspace.device}"
        )


def _validate_self_routing_gate_up_result(gate_up_result: Any) -> None:
    for name in ("topk_output", "ep_metadata", "fused_metadata"):
        if not hasattr(gate_up_result, name):
            raise EPWorkspaceError(
                "self_routing_fused_down_combine expects "
                "SelfRoutingFusedGateUpResult"
            )
    topk_output = gate_up_result.topk_output
    if not hasattr(topk_output, "topk_ids") or not hasattr(topk_output, "topk_weights"):
        raise EPWorkspaceError(
            "self_routing_fused_down_combine expects resolved standard top-k output"
        )


def _validate_owner_intermediate_rows(
    owner_intermediate: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
) -> None:
    expected_rows = int(dispatch_plan.local_expert_offsets[-1].item())
    if owner_intermediate.shape[0] != expected_rows:
        raise EPWorkspaceError(
            "owner_intermediate rows must match the local dispatch plan rows: "
            f"{owner_intermediate.shape[0]} != {expected_rows}"
        )


def _validate_down_kernel_inputs(
    owner_intermediate: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_down_weight: torch.Tensor,
    local_down_weight_scale: torch.Tensor,
    out: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
    *,
    block_shape: tuple[int, int],
) -> None:
    del topk_weights
    if local_down_weight.ndim != 3:
        raise EPWorkspaceError(
            "local_down_weight must be rank-3, got "
            f"{tuple(local_down_weight.shape)}"
        )
    if local_down_weight_scale.ndim != 3:
        raise EPWorkspaceError(
            "local_down_weight_scale must be rank-3, got "
            f"{tuple(local_down_weight_scale.shape)}"
        )
    if local_down_weight.dtype not in _fp8_e4m3_dtypes():
        raise EPWorkspaceError(
            f"local_down_weight must be FP8 E4M3, got {local_down_weight.dtype}"
        )
    if local_down_weight_scale.dtype != torch.float32:
        raise EPWorkspaceError(
            "local_down_weight_scale must be torch.float32, got "
            f"{local_down_weight_scale.dtype}"
        )
    if len(block_shape) != 2 or block_shape[0] <= 0 or block_shape[1] <= 0:
        raise EPWorkspaceError(f"invalid block_shape {block_shape!r}")
    if local_down_weight.shape[0] != dispatch_plan.local_expert_counts.numel():
        raise EPWorkspaceError(
            f"local_down_weight experts {local_down_weight.shape[0]} != "
            f"{dispatch_plan.local_expert_counts.numel()}"
        )
    if local_down_weight.shape[2] != owner_intermediate.shape[1]:
        raise EPWorkspaceError(
            f"local_down_weight K {local_down_weight.shape[2]} != intermediate "
            f"{owner_intermediate.shape[1]}"
        )
    expected_scale_shape = (
        local_down_weight.shape[0],
        _ceil_div(local_down_weight.shape[1], block_shape[0]),
        _ceil_div(local_down_weight.shape[2], block_shape[1]),
    )
    if local_down_weight_scale.shape != expected_scale_shape:
        raise EPWorkspaceError(
            "local_down_weight_scale shape "
            f"{tuple(local_down_weight_scale.shape)} != {expected_scale_shape}"
        )
    expected_out_shape = (topk_ids.shape[0], local_down_weight.shape[1])
    if out.shape != expected_out_shape:
        raise EPWorkspaceError(f"out shape {tuple(out.shape)} != {expected_out_shape}")
    if out.dtype != owner_intermediate.dtype:
        raise EPWorkspaceError(
            f"out dtype {out.dtype} != intermediate dtype {owner_intermediate.dtype}"
        )
    for name, tensor in (
        ("local_down_weight", local_down_weight),
        ("local_down_weight_scale", local_down_weight_scale),
        ("out", out),
    ):
        if tensor.device != owner_intermediate.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != intermediate device "
                f"{owner_intermediate.device}"
            )


def _require_same_metadata_tensor(
    name: str,
    fused: torch.Tensor,
    source: torch.Tensor,
    workspace: EPCommunicationWorkspace,
) -> None:
    _validate_int32_device_tensor(name, fused, workspace)
    _validate_int32_device_tensor(name, source, workspace)
    if fused.shape != source.shape:
        raise EPWorkspaceError(
            f"fused {name} shape {tuple(fused.shape)} != source shape "
            f"{tuple(source.shape)}"
        )
    if not torch.equal(fused, source):
        raise EPWorkspaceError(f"fused {name} does not match S4 EP metadata")


def _validate_int32_device_tensor(
    name: str,
    tensor: torch.Tensor,
    workspace: EPCommunicationWorkspace,
) -> None:
    if tensor.dtype != torch.int32:
        raise EPWorkspaceError(f"{name} must be torch.int32, got {tensor.dtype}")
    if tensor.device != workspace.device:
        raise EPWorkspaceError(
            f"{name} device {tensor.device} != workspace device {workspace.device}"
        )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _fp8_e4m3_dtypes() -> tuple[torch.dtype, ...]:
    return tuple(
        dtype
        for name in ("float8_e4m3fn", "float8_e4m3fnuz")
        if (dtype := getattr(torch, name, None)) is not None
    )


__all__ = [
    "PreRoutedFusedDownCombineResult",
    "pre_routed_fused_down_combine",
    "self_routing_fused_down_combine",
]
