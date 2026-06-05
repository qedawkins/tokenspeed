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
import os
import socket
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    owner_directed_dispatch,
    prepare_owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_experts import (
    owner_rank_fp8_expert_gemm,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_gate_up import (
    pre_routed_fused_dispatch_gate_up,
    self_routing_fused_dispatch_gate_up,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    build_pre_routed_fused_ep_metadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for fused dispatch+gate/up validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for fused dispatch+gate/up validation")


def _uniform_owner_maps(
    world_size: int,
    num_local_experts: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = world_size * num_local_experts
    experts = torch.arange(num_experts, dtype=torch.int32, device=device)
    return experts // num_local_experts, experts % num_local_experts


def _workspace(
    *,
    world_size: int,
    rank: int,
    hidden_size: int,
    max_tokens_per_rank: int,
    top_k: int,
    device: torch.device | str,
    dtype: torch.dtype,
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


def _reference_ep_metadata(
    topk_ids: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_local_experts: int,
) -> SimpleNamespace:
    topk_cpu = topk_ids.detach().cpu()
    owner_cpu = expert_owner.detach().cpu()
    local_cpu = local_expert_id.detach().cpu()
    flat = topk_cpu.reshape(-1)
    owner_expert_counts = torch.zeros(
        (world_size, num_local_experts),
        dtype=torch.int32,
    )
    for expert_tensor in flat:
        expert = int(expert_tensor.item())
        if expert < 0 or expert >= owner_cpu.numel():
            continue
        owner = int(owner_cpu[expert].item())
        local_id = int(local_cpu[expert].item())
        if 0 <= owner < world_size and 0 <= local_id < num_local_experts:
            owner_expert_counts[owner, local_id] += 1

    owner_counts = owner_expert_counts.sum(dim=1).to(torch.int32)
    owner_offsets = torch.zeros((world_size + 1,), dtype=torch.int32)
    owner_offsets[1:] = owner_counts.cumsum(dim=0)
    owner_expert_offsets = torch.zeros(
        (world_size, num_local_experts + 1),
        dtype=torch.int32,
    )
    owner_expert_offsets[:, 1:] = owner_expert_counts.cumsum(dim=1)
    dispatch_offsets = torch.full(topk_cpu.shape, -1, dtype=torch.int32)
    combine_offsets = torch.full(
        (world_size, flat.numel()),
        -1,
        dtype=torch.int32,
    )
    write_offsets = owner_expert_offsets[:, :-1].clone()
    for flat_slot, expert_tensor in enumerate(flat):
        expert = int(expert_tensor.item())
        if expert < 0 or expert >= owner_cpu.numel():
            continue
        owner = int(owner_cpu[expert].item())
        local_id = int(local_cpu[expert].item())
        if not (0 <= owner < world_size and 0 <= local_id < num_local_experts):
            continue
        row = int(write_offsets[owner, local_id].item())
        write_offsets[owner, local_id] += 1
        token = flat_slot // topk_cpu.shape[1]
        slot = flat_slot - token * topk_cpu.shape[1]
        dispatch_offsets[token, slot] = row
        combine_offsets[owner, row] = flat_slot

    return SimpleNamespace(
        owner_counts=owner_counts.to(topk_ids.device),
        owner_offsets=owner_offsets.to(topk_ids.device),
        owner_expert_counts=owner_expert_counts.to(topk_ids.device),
        owner_expert_offsets=owner_expert_offsets.to(topk_ids.device),
        dispatch_offsets=dispatch_offsets.to(topk_ids.device),
        combine_offsets=combine_offsets.to(topk_ids.device),
    )


def _make_fp8_weight(
    dense: torch.Tensor,
    block_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.platform import current_platform

    fp8 = current_platform().fp8e4m3fn
    block_n, block_k = block_shape
    num_experts, n_size, k_size = dense.shape
    scale = torch.empty(
        (
            num_experts,
            math.ceil(n_size / block_n),
            math.ceil(k_size / block_k),
        ),
        device=dense.device,
        dtype=torch.float32,
    )
    quantized = torch.empty(dense.shape, device=dense.device, dtype=fp8.dtype)

    for expert in range(num_experts):
        for n_block, n_start in enumerate(range(0, n_size, block_n)):
            n_end = min(n_start + block_n, n_size)
            for k_block, k_start in enumerate(range(0, k_size, block_k)):
                k_end = min(k_start + block_k, k_size)
                block = dense[expert, n_start:n_end, k_start:k_end].float()
                block_scale = torch.clamp(block.abs().max() / fp8.max, min=1e-6)
                scale[expert, n_block, k_block] = block_scale
                quantized[expert, n_start:n_end, k_start:k_end] = torch.clamp(
                    block / block_scale,
                    min=fp8.min,
                    max=fp8.max,
                ).to(fp8.dtype)

    return quantized, scale


def test_pre_routed_fused_dispatch_gate_up_handles_zero_token_rank() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for zero-row gate/up validation")

    world_size = 4
    rank = 3
    num_local_experts = 2
    hidden_size = 32
    output_size = 32
    top_k = 2
    block_shape = (16, 16)
    device = torch.device("cpu")
    topk_ids = torch.empty((0, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.empty((0, top_k), dtype=torch.float32, device=device)
    hidden_states = torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device)
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
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
        max_tokens_per_rank=0,
        top_k=top_k,
        device=device,
        dtype=hidden_states.dtype,
    )
    fused_metadata = build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )
    fused_metadata = replace(
        fused_metadata,
        owner_row_to_source=torch.full_like(fused_metadata.owner_row_to_source, -1),
    )
    local_gate_up_weight = torch.empty(
        (num_local_experts, output_size, hidden_size),
        dtype=fp8_dtype,
        device=device,
    )
    local_gate_up_scale = torch.empty(
        (num_local_experts, 2, 2),
        dtype=torch.float32,
        device=device,
    )

    result = pre_routed_fused_dispatch_gate_up(
        hidden_states,
        topk_ids,
        ep_metadata,
        workspace,
        fused_metadata,
        local_gate_up_weight,
        local_gate_up_scale,
        block_shape=block_shape,
    )

    assert result.owner_tokens.shape == (0, hidden_size)
    assert result.gate_up.shape == (0, output_size)
    assert result.dispatch_step.num_rows == 0
    assert result.dispatch_plan.local_expert_counts.tolist() == [0, 0]


def test_pre_routed_fused_dispatch_gate_up_handles_all_remote_rank() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for zero-row gate/up validation")

    world_size = 2
    rank = 1
    num_local_experts = 2
    hidden_size = 32
    output_size = 32
    top_k = 2
    block_shape = (16, 16)
    device = torch.device("cpu")
    topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    hidden_states = torch.arange(
        topk_ids.shape[0] * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(topk_ids.shape[0], hidden_size)
    hidden_states = hidden_states.to(torch.bfloat16)
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
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
        top_k=top_k,
        device=device,
        dtype=hidden_states.dtype,
    )
    fused_metadata = build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )
    local_gate_up_weight = torch.empty(
        (num_local_experts, output_size, hidden_size),
        dtype=fp8_dtype,
        device=device,
    )
    local_gate_up_scale = torch.empty(
        (num_local_experts, 2, 2),
        dtype=torch.float32,
        device=device,
    )

    result = pre_routed_fused_dispatch_gate_up(
        hidden_states,
        topk_ids,
        ep_metadata,
        workspace,
        fused_metadata,
        local_gate_up_weight,
        local_gate_up_scale,
        block_shape=block_shape,
    )

    assert result.owner_tokens.shape == (0, hidden_size)
    assert result.gate_up.shape == (0, output_size)
    assert result.dispatch_plan.local_expert_counts.tolist() == [0, 0]


def test_pre_routed_fused_dispatch_gate_up_matches_s4_unfused_hot_empty_experts() -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel

    torch.manual_seed(8502)
    device = torch.device("cuda")
    world_size = 2
    rank = 0
    num_local_experts = 3
    hidden_size = 32
    intermediate_size = 16
    output_size = 2 * intermediate_size
    top_k = 2
    block_shape = (16, 16)
    block_size = 16
    topk_ids = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.linspace(
        0.2,
        0.9,
        steps=topk_ids.numel(),
        dtype=torch.float32,
        device=device,
    ).reshape(topk_ids.shape)
    hidden_states = (
        torch.randn(topk_ids.shape[0], hidden_size, device=device) * 0.11
    ).to(torch.bfloat16)
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
    )
    ep_metadata = tokenspeed_kernel.moe_dispatch(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank,
        world_size,
        num_local_experts,
        dtype=torch.int32,
        traits={"comm_strategy": "ep_metadata"},
        expected_kernel_name="gluon_ep_metadata_gfx950",
    )
    torch.cuda.synchronize()
    ref_workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=top_k,
        device=device,
        dtype=hidden_states.dtype,
    )
    fused_workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=topk_ids.shape[0],
        top_k=top_k,
        device=device,
        dtype=hidden_states.dtype,
    )
    ref_plan = prepare_owner_directed_dispatch(ep_metadata, ref_workspace)
    fused_metadata = build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        fused_workspace,
        dispatch_plan=ref_plan,
    )
    dense_weight = (
        torch.randn(num_local_experts, output_size, hidden_size, device=device) * 0.13
    )
    local_gate_up_weight, local_gate_up_scale = _make_fp8_weight(
        dense_weight,
        block_shape,
    )

    ref_step = owner_directed_dispatch(
        hidden_states,
        topk_ids,
        ep_metadata,
        ref_workspace,
        dispatch_plan=ref_plan,
    )
    ref_gate_up = owner_rank_fp8_expert_gemm(
        ref_step.dispatch_buffer,
        local_gate_up_weight,
        local_gate_up_scale,
        ref_plan.local_expert_counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=ref_plan.local_expert_offsets,
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    result = pre_routed_fused_dispatch_gate_up(
        hidden_states,
        topk_ids,
        ep_metadata,
        fused_workspace,
        fused_metadata,
        local_gate_up_weight,
        local_gate_up_scale,
        block_shape=block_shape,
        block_size=block_size,
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()

    assert ref_plan.local_expert_counts.tolist() == [6, 2, 0]
    torch.testing.assert_close(result.owner_tokens, ref_step.dispatch_buffer)
    torch.testing.assert_close(result.gate_up, ref_gate_up)


def test_self_routing_fused_dispatch_gate_up_matches_pre_routed_same_logits() -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel
    from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutputFormat

    torch.manual_seed(8602)
    device = torch.device("cuda")
    world_size = 2
    rank = 0
    num_local_experts = 3
    num_experts = world_size * num_local_experts
    hidden_size = 32
    output_size = 32
    top_k = 2
    num_expert_group = 2
    topk_group = 1
    block_shape = (16, 16)
    block_size = 16
    hidden_states = (torch.randn(4, hidden_size, device=device) * 0.11).to(
        torch.bfloat16
    )
    router_logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, -4.0, -5.0, -6.0],
            [4.0, 5.0, 3.0, -4.0, -5.0, -6.0],
            [3.0, 4.0, 5.0, -4.0, -5.0, -6.0],
            [5.0, 3.0, 4.0, -4.0, -5.0, -6.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    routing_bias = torch.zeros(num_experts, device=device)
    topk_kwargs = dict(
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        correction_bias=routing_bias,
        routed_scaling_factor=1.0,
    )
    pre_routed_topk = TopK(top_k, **topk_kwargs)(hidden_states, router_logits)
    bypassed_topk = TopK(
        top_k,
        **topk_kwargs,
        output_format=TopKOutputFormat.BYPASSED,
    )(hidden_states, router_logits)
    torch.cuda.synchronize()

    topk_ids = pre_routed_topk.topk_ids.to(torch.int32)
    topk_weights = pre_routed_topk.topk_weights
    expert_owner, local_expert_id = _uniform_owner_maps(
        world_size,
        num_local_experts,
        device=device,
    )
    ep_metadata = tokenspeed_kernel.moe_dispatch(
        topk_ids,
        expert_owner,
        local_expert_id,
        rank,
        world_size,
        num_local_experts,
        dtype=torch.int32,
        traits={"comm_strategy": "ep_metadata"},
        expected_kernel_name="gluon_ep_metadata_gfx950",
    )
    torch.cuda.synchronize()
    ref_workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=hidden_states.shape[0],
        top_k=top_k,
        device=device,
        dtype=hidden_states.dtype,
    )
    self_workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=hidden_states.shape[0],
        top_k=top_k,
        device=device,
        dtype=hidden_states.dtype,
    )
    ref_plan = prepare_owner_directed_dispatch(ep_metadata, ref_workspace)
    ref_fused_metadata = build_pre_routed_fused_ep_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        ep_metadata,
        ref_workspace,
        dispatch_plan=ref_plan,
    )
    dense_weight = (
        torch.randn(num_local_experts, output_size, hidden_size, device=device) * 0.13
    )
    local_gate_up_weight, local_gate_up_scale = _make_fp8_weight(
        dense_weight,
        block_shape,
    )

    ref_result = pre_routed_fused_dispatch_gate_up(
        hidden_states,
        topk_ids,
        ep_metadata,
        ref_workspace,
        ref_fused_metadata,
        local_gate_up_weight,
        local_gate_up_scale,
        block_shape=block_shape,
        block_size=block_size,
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    self_result = self_routing_fused_dispatch_gate_up(
        hidden_states,
        bypassed_topk,
        expert_owner,
        local_expert_id,
        self_workspace,
        local_gate_up_weight,
        local_gate_up_scale,
        block_shape=block_shape,
        block_size=block_size,
        expected_metadata_kernel_name="gluon_ep_metadata_gfx950",
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()

    assert ref_plan.local_expert_counts.sum().item() > 0
    torch.testing.assert_close(self_result.topk_output.topk_ids, topk_ids)
    torch.testing.assert_close(self_result.topk_output.topk_weights, topk_weights)
    torch.testing.assert_close(
        self_result.fused_metadata.owner_rows,
        ref_fused_metadata.owner_rows,
    )
    torch.testing.assert_close(
        self_result.dispatch_plan.local_expert_counts,
        ref_plan.local_expert_counts,
    )
    torch.testing.assert_close(self_result.owner_tokens, ref_result.owner_tokens)
    torch.testing.assert_close(self_result.gate_up, ref_result.gate_up)


def test_self_routing_fused_dispatch_gate_up_rejects_standard_topk_output() -> None:
    with pytest.raises(EPWorkspaceError, match="BypassedTopKOutput"):
        self_routing_fused_dispatch_gate_up(
            torch.empty(0, 0),
            SimpleNamespace(format=None),
            None,
            None,
            None,
            None,
            None,
            block_shape=(16, 16),
        )


def test_iris_pre_routed_fused_dispatch_gate_up_matches_s4_unfused_one_rank() -> None:
    iris = pytest.importorskip("iris")
    _require_cdna4_gpu()
    import tokenspeed_kernel

    initialized_here = _ensure_one_rank_dist_initialized()
    if dist.get_world_size() != 1:
        pytest.skip("one-rank fused gate/up compile smoke requires world_size=1")

    ctx = iris.iris(heap_size=1 << 28)
    try:
        torch.manual_seed(8503)
        device = ctx.get_device()
        world_size = 1
        rank = 0
        num_local_experts = 3
        hidden_size = 32
        output_size = 32
        top_k = 2
        block_shape = (16, 16)
        block_size = 16
        topk_ids = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [0, 0],
                [1, 0],
            ],
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.linspace(
            0.2,
            0.9,
            steps=topk_ids.numel(),
            dtype=torch.float32,
            device=device,
        ).reshape(topk_ids.shape)
        hidden_states = (
            torch.randn(topk_ids.shape[0], hidden_size, device=device) * 0.11
        ).to(torch.bfloat16)
        expert_owner, local_expert_id = _uniform_owner_maps(
            world_size,
            num_local_experts,
            device=device,
        )
        ep_metadata = tokenspeed_kernel.moe_dispatch(
            topk_ids,
            expert_owner,
            local_expert_id,
            rank,
            world_size,
            num_local_experts,
            dtype=torch.int32,
            traits={"comm_strategy": "ep_metadata"},
            expected_kernel_name="gluon_ep_metadata_gfx950",
        )
        torch.cuda.synchronize()
        ref_workspace = _workspace(
            world_size=world_size,
            rank=rank,
            hidden_size=hidden_size,
            max_tokens_per_rank=topk_ids.shape[0],
            top_k=top_k,
            device=device,
            dtype=hidden_states.dtype,
        )
        fused_workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=topk_ids.shape[0],
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=hidden_states.dtype,
            iris_mode="required",
            iris_context=ctx,
        )
        ref_plan = prepare_owner_directed_dispatch(ep_metadata, ref_workspace)
        fused_metadata = build_pre_routed_fused_ep_metadata(
            hidden_states,
            topk_ids,
            topk_weights,
            ep_metadata,
            fused_workspace,
            dispatch_plan=ref_plan,
        )
        dense_weight = (
            torch.randn(num_local_experts, output_size, hidden_size, device=device)
            * 0.13
        )
        local_gate_up_weight, local_gate_up_scale = _make_fp8_weight(
            dense_weight,
            block_shape,
        )

        ref_step = owner_directed_dispatch(
            hidden_states,
            topk_ids,
            ep_metadata,
            ref_workspace,
            dispatch_plan=ref_plan,
        )
        ref_gate_up = owner_rank_fp8_expert_gemm(
            ref_step.dispatch_buffer,
            local_gate_up_weight,
            local_gate_up_scale,
            ref_plan.local_expert_counts,
            block_shape=block_shape,
            block_size=block_size,
            local_expert_offsets=ref_plan.local_expert_offsets,
            expected_kernel_name="gluon_fp8_local_experts_gfx950",
        )
        result = pre_routed_fused_dispatch_gate_up(
            hidden_states,
            topk_ids,
            ep_metadata,
            fused_workspace,
            fused_metadata,
            local_gate_up_weight,
            local_gate_up_scale,
            block_shape=block_shape,
            block_size=block_size,
        )
        torch.cuda.synchronize()

        assert result.dispatch_plan.local_expert_counts.tolist() == [6, 2, 0]
        torch.testing.assert_close(result.owner_tokens, ref_step.dispatch_buffer)
        torch.testing.assert_close(result.gate_up, ref_gate_up)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()
            if initialized_here:
                dist.destroy_process_group()


def test_iris_pre_routed_fused_dispatch_gate_up_remote_zero_token_rank() -> None:
    iris = pytest.importorskip("iris")
    _require_cdna4_gpu()
    import tokenspeed_kernel

    initialized_here = _ensure_torchrun_dist_initialized(required_world_size=2)

    ctx = iris.iris(heap_size=1 << 28)
    try:
        device = ctx.get_device()
        world_size = ctx.get_num_ranks()
        rank = ctx.get_rank()
        num_local_experts = 2
        hidden_size = 32
        output_size = 32
        top_k = 2
        block_shape = (16, 16)
        block_size = 16
        max_tokens_per_rank = 2
        if rank == 0:
            topk_ids = torch.tensor(
                [[2, 2], [3, 2]],
                dtype=torch.int32,
                device=device,
            )
            hidden_states = _rank0_remote_test_hidden(device, hidden_size)
        else:
            topk_ids = torch.empty((0, top_k), dtype=torch.int32, device=device)
            hidden_states = torch.empty(
                (0, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        expert_owner, local_expert_id = _uniform_owner_maps(
            world_size,
            num_local_experts,
            device=device,
        )
        ep_metadata = tokenspeed_kernel.moe_dispatch(
            topk_ids,
            expert_owner,
            local_expert_id,
            rank,
            world_size,
            num_local_experts,
            dtype=torch.int32,
            traits={"comm_strategy": "ep_metadata"},
            expected_kernel_name="gluon_ep_metadata_gfx950",
        )
        torch.cuda.synchronize()
        workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=hidden_states.dtype,
            iris_mode="required",
            iris_context=ctx,
        )
        dispatch_plan = prepare_owner_directed_dispatch(ep_metadata, workspace)
        fused_metadata = build_pre_routed_fused_ep_metadata(
            hidden_states,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
            dispatch_plan=dispatch_plan,
        )
        dense_weight = (
            torch.arange(
                num_local_experts * output_size * hidden_size,
                dtype=torch.float32,
                device=device,
            ).reshape(num_local_experts, output_size, hidden_size)
            * 0.0007
            - 0.11
        )
        local_gate_up_weight, local_gate_up_scale = _make_fp8_weight(
            dense_weight,
            block_shape,
        )

        result = pre_routed_fused_dispatch_gate_up(
            hidden_states,
            topk_ids,
            ep_metadata,
            workspace,
            fused_metadata,
            local_gate_up_weight,
            local_gate_up_scale,
            block_shape=block_shape,
            block_size=block_size,
        )
        torch.cuda.synchronize()

        if rank == 0:
            assert result.owner_tokens.shape == (0, hidden_size)
            assert result.gate_up.shape == (0, output_size)
        else:
            expected_owner_tokens = _rank0_remote_test_hidden(
                device,
                hidden_size,
            )[torch.tensor([0, 0, 1, 1], device=device)]
            local_counts = torch.tensor([3, 1], dtype=torch.int32, device=device)
            local_offsets = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
            expected_gate_up = owner_rank_fp8_expert_gemm(
                expected_owner_tokens,
                local_gate_up_weight,
                local_gate_up_scale,
                local_counts,
                block_shape=block_shape,
                block_size=block_size,
                local_expert_offsets=local_offsets,
                expected_kernel_name="gluon_fp8_local_experts_gfx950",
            )
            torch.cuda.synchronize()
            assert result.dispatch_plan.local_expert_counts.tolist() == [3, 1]
            torch.testing.assert_close(result.owner_tokens, expected_owner_tokens)
            torch.testing.assert_close(result.gate_up, expected_gate_up)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()
            if initialized_here:
                dist.destroy_process_group()


def _rank0_remote_test_hidden(
    device: torch.device | str,
    hidden_size: int,
) -> torch.Tensor:
    return (
        torch.arange(2 * hidden_size, dtype=torch.float32, device=device)
        .reshape(2, hidden_size)
        .mul(0.01)
        .to(torch.bfloat16)
    )


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


def _ensure_torchrun_dist_initialized(*, required_world_size: int) -> bool:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if dist.is_initialized():
        if dist.get_world_size() != required_world_size:
            pytest.skip(f"requires world_size={required_world_size}")
        return False
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("remote fused gate/up test requires torchrun")
    if int(os.environ["WORLD_SIZE"]) != required_world_size:
        pytest.skip(f"requires world_size={required_world_size}")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend="nccl", init_method="env://")
    return True
