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

"""Internal pre-routed fused EP dispatch plus gate/up GEMM.

This module is part of the existing ``moe.fused`` ``pre_routed`` provider path.
It does not register a new public kernel mode or expose Iris-specific objects
through the MoE backend contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    EPOwnerDispatchPlan,
    owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_experts import (
    build_owner_expert_metadata,
    ep_expert_config,
    owner_rank_fp8_expert_gemm,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    PreRoutedFusedEPMetadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
    EPWorkspaceStep,
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
    _pre_routed_fused_dispatch_gate_up_kernel = None


@dataclass(frozen=True)
class PreRoutedFusedGateUpResult:
    gate_up: torch.Tensor
    owner_tokens: torch.Tensor
    dispatch_step: EPWorkspaceStep
    dispatch_plan: EPOwnerDispatchPlan


def pre_routed_fused_dispatch_gate_up(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    fused_metadata: PreRoutedFusedEPMetadata,
    local_gate_up_weight: torch.Tensor,
    local_gate_up_weight_scale: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    block_size: int = 16,
    out: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
    expected_gemm_kernel_name: str | None = None,
) -> PreRoutedFusedGateUpResult:
    """Produce owner-local gate/up activations for pre-routed EP tokens.

    Iris-backed workspaces use one Gluon producer-consumer kernel: dispatch
    programs publish hidden-state blocks to expert owners and release readiness
    flags, while GEMM programs acquire those flags before reading each owner
    row. The torch fallback preserves the S4 sequence for CPU and non-Iris
    smoke coverage.
    """

    _validate_pre_routed_inputs(
        hidden_states,
        topk_ids,
        ep_metadata,
        workspace,
        fused_metadata,
    )
    dispatch_plan = _dispatch_plan_from_fused_metadata(fused_metadata)
    if workspace.backend == "iris":
        return _pre_routed_fused_dispatch_gate_up_with_iris_gluon(
            hidden_states,
            topk_ids,
            ep_metadata,
            workspace,
            local_gate_up_weight,
            local_gate_up_weight_scale,
            dispatch_plan,
            block_shape=block_shape,
            block_size=block_size,
            out=out,
            config=config,
        )

    dispatch_step = owner_directed_dispatch(
        hidden_states,
        topk_ids,
        ep_metadata,
        workspace,
        dispatch_plan=dispatch_plan,
        synchronize=True,
    )
    gate_up = owner_rank_fp8_expert_gemm(
        dispatch_step.dispatch_buffer,
        local_gate_up_weight,
        local_gate_up_weight_scale,
        dispatch_plan.local_expert_counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
        out=out,
        config=config,
        expected_kernel_name=expected_gemm_kernel_name,
    )
    return PreRoutedFusedGateUpResult(
        gate_up=gate_up,
        owner_tokens=dispatch_step.dispatch_buffer,
        dispatch_step=dispatch_step,
        dispatch_plan=dispatch_plan,
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
    def _pre_routed_fused_dispatch_gate_up_kernel(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        dispatch_buffer_ptr,
        hidden_states_ptr,
        readiness_flags_ptr,
        output_ptr,
        weight_ptr,
        weight_scale_ptr,
        combine_offsets_ptr,
        owner_counts_ptr,
        owner_expert_offsets_ptr,
        owner_expert_base_offsets_ptr,
        aggregate_owner_expert_offsets_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        DISPATCH_STRIDE_M: gl.constexpr,
        HIDDEN_STRIDE_M: gl.constexpr,
        WEIGHT_STRIDE_E: gl.constexpr,
        WEIGHT_STRIDE_N: gl.constexpr,
        WEIGHT_STRIDE_K: gl.constexpr,
        OUTPUT_STRIDE_M: gl.constexpr,
        OUTPUT_STRIDE_N: gl.constexpr,
        WEIGHT_SCALE_STRIDE_E: gl.constexpr,
        WEIGHT_SCALE_STRIDE_N: gl.constexpr,
        WEIGHT_SCALE_STRIDE_K: gl.constexpr,
        COMBINE_STRIDE_OWNER: gl.constexpr,
        COMBINE_STRIDE_ROW: gl.constexpr,
        OWNER_EXPERT_STRIDE_OWNER: gl.constexpr,
        OWNER_EXPERT_STRIDE_LOCAL: gl.constexpr,
        OWNER_EXPERT_BASE_STRIDE_OWNER: gl.constexpr,
        OWNER_EXPERT_BASE_STRIDE_LOCAL: gl.constexpr,
        AGGREGATE_OWNER_EXPERT_STRIDE_OWNER: gl.constexpr,
        AGGREGATE_OWNER_EXPERT_STRIDE_LOCAL: gl.constexpr,
        WORKSPACE_RANK: gl.constexpr,
        WORLD_SIZE: gl.constexpr,
        HIDDEN_SIZE: gl.constexpr,
        OUTPUT_SIZE: gl.constexpr,
        TOPK: gl.constexpr,
        MAX_OWNER_ROWS: gl.constexpr,
        NUM_LOCAL_EXPERTS: gl.constexpr,
        NUM_PROGRAMS: gl.constexpr,
        TOTAL_DISPATCH_TILES: gl.constexpr,
        TOTAL_GEMM_TILES: gl.constexpr,
        DISPATCH_BLOCK_N: gl.constexpr,
        DISPATCH_BLOCK_N_ELEMS_PER_THREAD: gl.constexpr,
        NUM_READY_BLOCKS: gl.constexpr,
        NUM_VALID_ROWS: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        WEIGHT_BLOCK_N: gl.constexpr,
        WEIGHT_BLOCK_K: gl.constexpr,
        GEMM_BLOCK_K: gl.constexpr,
        GEMM_BLOCK_K_ELEMS_PER_THREAD: gl.constexpr,
    ):
        ctx = IrisDeviceCtx.initialize(context_tensor)
        pid = gl.program_id(0)

        for dispatch_tile in range(pid, TOTAL_DISPATCH_TILES, NUM_PROGRAMS):
            hidden_block = dispatch_tile % NUM_READY_BLOCKS
            owner_row_pid = dispatch_tile // NUM_READY_BLOCKS
            owner_row = owner_row_pid % MAX_OWNER_ROWS
            owner = owner_row_pid // MAX_OWNER_ROWS
            dispatch_layout: gl.constexpr = gl.BlockedLayout(
                [DISPATCH_BLOCK_N_ELEMS_PER_THREAD], [64], [1], [0]
            )
            offsets = hidden_block * DISPATCH_BLOCK_N + gl.arange(
                0,
                DISPATCH_BLOCK_N,
                layout=dispatch_layout,
            )
            mask_n = offsets < HIDDEN_SIZE
            owner_count = gl.load(owner_counts_ptr + owner).to(gl.int32)

            if owner_row < owner_count:
                flat_slot = gl.load(
                    combine_offsets_ptr
                    + owner * COMBINE_STRIDE_OWNER
                    + owner_row * COMBINE_STRIDE_ROW,
                ).to(gl.int32)
                if flat_slot >= 0:
                    local_id = gl.full((), 0, dtype=gl.int32)
                    local_start = gl.full((), 0, dtype=gl.int32)
                    for candidate in range(0, NUM_LOCAL_EXPERTS):
                        start = gl.load(
                            owner_expert_offsets_ptr
                            + owner * OWNER_EXPERT_STRIDE_OWNER
                            + candidate * OWNER_EXPERT_STRIDE_LOCAL,
                        ).to(gl.int32)
                        end = gl.load(
                            owner_expert_offsets_ptr
                            + owner * OWNER_EXPERT_STRIDE_OWNER
                            + (candidate + 1) * OWNER_EXPERT_STRIDE_LOCAL,
                        ).to(gl.int32)
                        if (owner_row >= start) & (owner_row < end):
                            local_id = candidate
                            local_start = start
                    token = flat_slot // TOPK
                    expert_base = gl.load(
                        aggregate_owner_expert_offsets_ptr
                        + owner * AGGREGATE_OWNER_EXPERT_STRIDE_OWNER
                        + local_id * AGGREGATE_OWNER_EXPERT_STRIDE_LOCAL,
                    ).to(gl.int32)
                    source_expert_base = gl.load(
                        owner_expert_base_offsets_ptr
                        + owner * OWNER_EXPERT_BASE_STRIDE_OWNER
                        + local_id * OWNER_EXPERT_BASE_STRIDE_LOCAL,
                    ).to(gl.int32)
                    dst_row = (
                        expert_base
                        + source_expert_base
                        + owner_row
                        - local_start
                    )
                    values = gl.load(
                        hidden_states_ptr + token * HIDDEN_STRIDE_M + offsets,
                        mask=mask_n,
                        other=0.0,
                    )
                    ctx.store(
                        dispatch_buffer_ptr + dst_row * DISPATCH_STRIDE_M + offsets,
                        values,
                        owner,
                        mask=mask_n,
                    )
                    ctx.atomic_cas(
                        readiness_flags_ptr + dst_row * NUM_READY_BLOCKS + hidden_block,
                        0,
                        1,
                        owner,
                        sem="release",
                        scope="sys",
                    )

        for gemm_tile in range(pid, TOTAL_GEMM_TILES, NUM_PROGRAMS):
            sorted_row = gemm_tile // OUTPUT_SIZE
            out_n = gemm_tile - sorted_row * OUTPUT_SIZE
            num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr).to(gl.int32)
            row_active = sorted_row < num_tokens_post_padded
            sorted_slot = gl.load(
                sorted_token_ids_ptr + sorted_row,
                mask=row_active,
                other=NUM_VALID_ROWS,
            ).to(gl.int32)
            slot_valid = sorted_slot < NUM_VALID_ROWS
            expert_id = gl.load(
                expert_ids_ptr + sorted_row // BLOCK_SIZE_M,
                mask=row_active,
                other=-1,
            ).to(gl.int32)
            expert_valid = expert_id >= 0
            source_row = sorted_slot

            if row_active & slot_valid & expert_valid:
                for ready_block in range(0, NUM_READY_BLOCKS):
                    done = gl.full((), 0, dtype=gl.int32)
                    while done != 1:
                        done = ctx.atomic_cas(
                            readiness_flags_ptr
                            + source_row * NUM_READY_BLOCKS
                            + ready_block,
                            1,
                            1,
                            WORKSPACE_RANK,
                            sem="acquire",
                            scope="sys",
                        ).to(gl.int32)

            gemm_layout: gl.constexpr = gl.BlockedLayout(
                [GEMM_BLOCK_K_ELEMS_PER_THREAD], [64], [1], [0]
            )
            offs_k = gl.arange(0, GEMM_BLOCK_K, layout=gemm_layout)
            k_valid = offs_k < HIDDEN_SIZE
            load_valid = row_active & slot_valid & expert_valid & k_valid
            activations = gl.load(
                dispatch_buffer_ptr + source_row * DISPATCH_STRIDE_M + offs_k,
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
            gl.store(
                output_ptr + source_row * OUTPUT_STRIDE_M + out_n * OUTPUT_STRIDE_N,
                acc.to(output_ptr.dtype.element_ty),
                mask=row_active & slot_valid & expert_valid,
            )


def _pre_routed_fused_dispatch_gate_up_with_iris_gluon(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    local_gate_up_weight: torch.Tensor,
    local_gate_up_weight_scale: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
    *,
    block_shape: tuple[int, int],
    block_size: int,
    out: torch.Tensor | None,
    config: dict[str, Any] | None,
) -> PreRoutedFusedGateUpResult:
    if _pre_routed_fused_dispatch_gate_up_kernel is None or IrisDeviceCtx is None:
        raise EPWorkspaceUnavailable(
            "Iris Gluon fused dispatch+gate/up support is not importable"
        )
    if triton is None:
        raise EPWorkspaceUnavailable("Triton is not importable")
    if workspace.handle.device_context is None or workspace._iris_context is None:
        raise EPWorkspaceUnavailable("Iris-backed workspace has no device context")

    num_rows = int(dispatch_plan.local_expert_offsets[-1].item())
    output_size = local_gate_up_weight.shape[1]
    if out is None:
        out = torch.empty(
            (num_rows, output_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    expert_metadata = build_owner_expert_metadata(
        dispatch_plan.local_expert_counts,
        block_size=block_size,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
    )
    _validate_gate_up_kernel_inputs(
        hidden_states,
        local_gate_up_weight,
        local_gate_up_weight_scale,
        out,
        dispatch_plan,
        block_shape=block_shape,
        num_rows=num_rows,
    )

    if workspace.world_size == 1:
        rank_counts = torch.tensor(
            [num_rows],
            dtype=torch.int32,
            device=workspace.device,
        )
        rank_offsets = torch.tensor(
            [0, num_rows],
            dtype=torch.int32,
            device=workspace.device,
        )
        workspace.prepare_step(rank_counts, rank_offsets, num_rows=num_rows)
    dispatch_step = workspace.view(num_rows)

    hidden_size = hidden_states.shape[1]
    dispatch_block_n = min(triton.next_power_of_2(hidden_size), 512)
    num_ready_blocks = triton.cdiv(hidden_size, dispatch_block_n)
    readiness_flags = workspace._iris_context.zeros(
        (workspace.max_dispatch_rows * num_ready_blocks,),
        dtype=torch.int32,
    )
    workspace.barrier()
    max_owner_rows = int(ep_metadata.owner_counts.max().item())
    max_owner_rows_arg = max(max_owner_rows, 1)
    dispatch_programs = workspace.world_size * max_owner_rows * num_ready_blocks
    gemm_programs = expert_metadata.sorted_token_ids.numel() * output_size
    if dispatch_programs > 0 or gemm_programs > 0:
        if config is None:
            config = ep_expert_config(block_size=block_size, block_shape=block_shape)
        gemm_block_k = max(64, triton.next_power_of_2(hidden_size))
        weight_block_n, weight_block_k = int(block_shape[0]), int(block_shape[1])
        num_programs = min(
            torch.cuda.get_device_properties(hidden_states.device).multi_processor_count,
            max(dispatch_programs + gemm_programs, 1),
        )
        _pre_routed_fused_dispatch_gate_up_kernel[(num_programs,)](
            IrisDeviceCtx,
            workspace.handle.device_context,
            workspace.dispatch_buffer,
            hidden_states,
            readiness_flags,
            out,
            local_gate_up_weight,
            local_gate_up_weight_scale,
            ep_metadata.combine_offsets,
            ep_metadata.owner_counts,
            ep_metadata.owner_expert_offsets,
            dispatch_plan.owner_expert_base_offsets,
            dispatch_plan.aggregate_owner_expert_offsets,
            expert_metadata.sorted_token_ids,
            expert_metadata.expert_ids,
            expert_metadata.num_tokens_post_padded,
            workspace.dispatch_buffer.stride(0),
            hidden_states.stride(0),
            local_gate_up_weight.stride(0),
            local_gate_up_weight.stride(1),
            local_gate_up_weight.stride(2),
            out.stride(0),
            out.stride(1),
            local_gate_up_weight_scale.stride(0),
            local_gate_up_weight_scale.stride(1),
            local_gate_up_weight_scale.stride(2),
            ep_metadata.combine_offsets.stride(0),
            ep_metadata.combine_offsets.stride(1),
            ep_metadata.owner_expert_offsets.stride(0),
            ep_metadata.owner_expert_offsets.stride(1),
            dispatch_plan.owner_expert_base_offsets.stride(0),
            dispatch_plan.owner_expert_base_offsets.stride(1),
            dispatch_plan.aggregate_owner_expert_offsets.stride(0),
            dispatch_plan.aggregate_owner_expert_offsets.stride(1),
            workspace.rank,
            workspace.world_size,
            hidden_size,
            output_size,
            topk_ids.shape[1],
            max_owner_rows_arg,
            dispatch_plan.local_expert_counts.numel(),
            num_programs,
            dispatch_programs,
            gemm_programs,
            dispatch_block_n,
            max(dispatch_block_n // 64, 1),
            num_ready_blocks,
            num_rows,
            config["BLOCK_SIZE_M"],
            weight_block_n,
            weight_block_k,
            gemm_block_k,
            max(gemm_block_k // 64, 1),
            num_warps=1,
        )

    return PreRoutedFusedGateUpResult(
        gate_up=out,
        owner_tokens=dispatch_step.dispatch_buffer,
        dispatch_step=dispatch_step,
        dispatch_plan=dispatch_plan,
    )


def _validate_gate_up_kernel_inputs(
    hidden_states: torch.Tensor,
    local_gate_up_weight: torch.Tensor,
    local_gate_up_weight_scale: torch.Tensor,
    out: torch.Tensor,
    dispatch_plan: EPOwnerDispatchPlan,
    *,
    block_shape: tuple[int, int],
    num_rows: int,
) -> None:
    if local_gate_up_weight.ndim != 3:
        raise EPWorkspaceError(
            "local_gate_up_weight must be rank-3, got "
            f"{tuple(local_gate_up_weight.shape)}"
        )
    if local_gate_up_weight_scale.ndim != 3:
        raise EPWorkspaceError(
            "local_gate_up_weight_scale must be rank-3, got "
            f"{tuple(local_gate_up_weight_scale.shape)}"
        )
    if local_gate_up_weight.dtype not in _fp8_e4m3_dtypes():
        raise EPWorkspaceError(
            f"local_gate_up_weight must be FP8 E4M3, got {local_gate_up_weight.dtype}"
        )
    if local_gate_up_weight_scale.dtype != torch.float32:
        raise EPWorkspaceError(
            "local_gate_up_weight_scale must be torch.float32, got "
            f"{local_gate_up_weight_scale.dtype}"
        )
    if len(block_shape) != 2 or block_shape[0] <= 0 or block_shape[1] <= 0:
        raise EPWorkspaceError(f"invalid block_shape {block_shape!r}")
    if local_gate_up_weight.shape[0] != dispatch_plan.local_expert_counts.numel():
        raise EPWorkspaceError(
            f"local_gate_up_weight experts {local_gate_up_weight.shape[0]} != "
            f"{dispatch_plan.local_expert_counts.numel()}"
        )
    if local_gate_up_weight.shape[2] != hidden_states.shape[1]:
        raise EPWorkspaceError(
            f"local_gate_up_weight K {local_gate_up_weight.shape[2]} != hidden "
            f"{hidden_states.shape[1]}"
        )
    expected_scale_shape = (
        local_gate_up_weight.shape[0],
        _ceil_div(local_gate_up_weight.shape[1], block_shape[0]),
        _ceil_div(local_gate_up_weight.shape[2], block_shape[1]),
    )
    if local_gate_up_weight_scale.shape != expected_scale_shape:
        raise EPWorkspaceError(
            "local_gate_up_weight_scale shape "
            f"{tuple(local_gate_up_weight_scale.shape)} != {expected_scale_shape}"
        )
    expected_out_shape = (num_rows, local_gate_up_weight.shape[1])
    if out.shape != expected_out_shape:
        raise EPWorkspaceError(f"out shape {tuple(out.shape)} != {expected_out_shape}")
    if out.dtype != hidden_states.dtype:
        raise EPWorkspaceError(
            f"out dtype {out.dtype} != hidden dtype {hidden_states.dtype}"
        )
    for name, tensor in (
        ("local_gate_up_weight", local_gate_up_weight),
        ("local_gate_up_weight_scale", local_gate_up_weight_scale),
        ("out", out),
    ):
        if tensor.device != hidden_states.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != hidden device "
                f"{hidden_states.device}"
            )


def _validate_pre_routed_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_metadata: Any,
    workspace: EPCommunicationWorkspace,
    fused_metadata: PreRoutedFusedEPMetadata,
) -> None:
    workspace.validate_payload(hidden_states, name="hidden_states")
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
    if fused_metadata.num_tokens != hidden_states.shape[0]:
        raise EPWorkspaceError(
            f"fused num_tokens {fused_metadata.num_tokens} != hidden rows "
            f"{hidden_states.shape[0]}"
        )
    if fused_metadata.hidden_size != hidden_states.shape[1]:
        raise EPWorkspaceError(
            f"fused hidden_size {fused_metadata.hidden_size} != hidden size "
            f"{hidden_states.shape[1]}"
        )
    if topk_ids.shape != (hidden_states.shape[0], fused_metadata.top_k):
        raise EPWorkspaceError(
            f"topk_ids shape {tuple(topk_ids.shape)} != "
            f"{(hidden_states.shape[0], fused_metadata.top_k)}"
        )
    if topk_ids.dtype != torch.int32:
        raise EPWorkspaceError(f"topk_ids must be torch.int32, got {topk_ids.dtype}")
    if topk_ids.device != workspace.device:
        raise EPWorkspaceError(
            f"topk_ids device {topk_ids.device} != workspace device "
            f"{workspace.device}"
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
        ("owner_expert_base_offsets", fused_metadata.owner_expert_base_offsets),
        ("aggregate_owner_expert_counts", fused_metadata.aggregate_owner_expert_counts),
        (
            "aggregate_owner_expert_offsets",
            fused_metadata.aggregate_owner_expert_offsets,
        ),
        ("local_expert_counts", fused_metadata.local_expert_counts),
        ("local_expert_offsets", fused_metadata.local_expert_offsets),
        ("owner_row_to_source", fused_metadata.owner_row_to_source),
    ):
        _validate_int32_device_tensor(name, tensor, workspace)
    if (
        fused_metadata.owner_row_to_source.ndim != 3
        or fused_metadata.owner_row_to_source.shape[0] != workspace.world_size
        or fused_metadata.owner_row_to_source.shape[2] != 3
    ):
        raise EPWorkspaceError(
            "owner_row_to_source must have shape "
            f"({workspace.world_size}, rows, 3), got "
            f"{tuple(fused_metadata.owner_row_to_source.shape)}"
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
    "PreRoutedFusedGateUpResult",
    "pre_routed_fused_dispatch_gate_up",
]
