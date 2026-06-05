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
import math
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.moe.backends.ep_combine import owner_directed_combine
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    prepare_owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_experts import (
    owner_rank_fp8_expert_gemm,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_down_combine import (
    pre_routed_fused_down_combine,
    self_routing_fused_down_combine,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_gate_up import (
    pre_routed_fused_dispatch_gate_up,
    self_routing_fused_dispatch_gate_up,
)
from tokenspeed.runtime.layers.moe.backends.ep_fused_metadata import (
    build_pre_routed_fused_ep_metadata,
)
from tokenspeed.runtime.layers.moe.backends.ep_reduce import ep_weighted_reduce
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for fused down+combine validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for fused down+combine validation")


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


def test_pre_routed_fused_down_combine_handles_zero_token_rank() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for zero-row down validation")

    world_size = 4
    rank = 3
    num_local_experts = 2
    hidden_size = 32
    intermediate_size = 16
    top_k = 2
    block_shape = (16, 16)
    device = torch.device("cpu")
    topk_ids = torch.empty((0, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.empty((0, top_k), dtype=torch.float32, device=device)
    source_hidden = torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device)
    owner_intermediate = torch.empty(
        (0, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
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
        dtype=source_hidden.dtype,
    )
    fused_metadata = build_pre_routed_fused_ep_metadata(
        source_hidden,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
    )
    local_down_weight = torch.empty(
        (num_local_experts, hidden_size, intermediate_size),
        dtype=fp8_dtype,
        device=device,
    )
    local_down_scale = torch.empty(
        (num_local_experts, 2, 1),
        dtype=torch.float32,
        device=device,
    )

    result = pre_routed_fused_down_combine(
        owner_intermediate,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        fused_metadata,
        local_down_weight,
        local_down_scale,
        expert_owner,
        local_expert_id,
        block_shape=block_shape,
    )

    assert result.output.shape == (0, hidden_size)
    assert result.dispatch_plan.local_expert_counts.tolist() == [0, 0]


def test_pre_routed_fused_down_combine_rejects_owner_row_mismatch() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for owner-row validation")

    world_size = 1
    rank = 0
    num_local_experts = 2
    hidden_size = 32
    intermediate_size = 16
    top_k = 2
    block_shape = (16, 16)
    device = torch.device("cpu")
    topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    source_hidden = torch.empty((2, hidden_size), dtype=torch.bfloat16, device=device)
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
        max_tokens_per_rank=2,
        top_k=top_k,
        device=device,
        dtype=source_hidden.dtype,
    )
    dispatch_plan = prepare_owner_directed_dispatch(ep_metadata, workspace)
    fused_metadata = build_pre_routed_fused_ep_metadata(
        source_hidden,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        dispatch_plan=dispatch_plan,
    )
    expected_rows = int(dispatch_plan.local_expert_offsets[-1].item())
    owner_intermediate = torch.empty(
        (expected_rows - 1, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    local_down_weight = torch.empty(
        (num_local_experts, hidden_size, intermediate_size),
        dtype=fp8_dtype,
        device=device,
    )
    local_down_scale = torch.empty(
        (num_local_experts, 2, 1),
        dtype=torch.float32,
        device=device,
    )

    with pytest.raises(EPWorkspaceError, match="owner_intermediate rows"):
        pre_routed_fused_down_combine(
            owner_intermediate,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
            fused_metadata,
            local_down_weight,
            local_down_scale,
            expert_owner,
            local_expert_id,
            block_shape=block_shape,
        )


def test_self_routing_fused_down_combine_rejects_invalid_gate_up_result() -> None:
    with pytest.raises(EPWorkspaceError, match="SelfRoutingFusedGateUpResult"):
        self_routing_fused_down_combine(
            torch.empty(0, 0),
            SimpleNamespace(),
            None,
            None,
            None,
            None,
            None,
            block_shape=(16, 16),
        )


def test_self_routing_fused_down_combine_matches_pre_routed_final_output() -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel
    from tokenspeed.runtime.layers.activation import silu_and_mul
    from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutputFormat

    torch.manual_seed(8703)
    device = torch.device("cuda")
    world_size = 2
    rank = 0
    num_local_experts = 3
    num_experts = world_size * num_local_experts
    hidden_size = 32
    intermediate_size = 16
    gate_up_size = 2 * intermediate_size
    top_k = 2
    num_expert_group = 2
    topk_group = 1
    block_shape = (16, 16)
    block_size = 16
    routed_scaling_factor = 0.75
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
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
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
    dense_gate_up_weight = (
        torch.randn(num_local_experts, gate_up_size, hidden_size, device=device) * 0.13
    )
    gate_up_weight, gate_up_scale = _make_fp8_weight(
        dense_gate_up_weight,
        block_shape,
    )
    dense_down_weight = (
        torch.randn(num_local_experts, hidden_size, intermediate_size, device=device)
        * 0.11
    )
    down_weight, down_scale = _make_fp8_weight(dense_down_weight, block_shape)

    ref_gate_up = pre_routed_fused_dispatch_gate_up(
        hidden_states,
        topk_ids,
        ep_metadata,
        ref_workspace,
        ref_fused_metadata,
        gate_up_weight,
        gate_up_scale,
        block_shape=block_shape,
        block_size=block_size,
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    self_gate_up = self_routing_fused_dispatch_gate_up(
        hidden_states,
        bypassed_topk,
        expert_owner,
        local_expert_id,
        self_workspace,
        gate_up_weight,
        gate_up_scale,
        block_shape=block_shape,
        block_size=block_size,
        expected_metadata_kernel_name="gluon_ep_metadata_gfx950",
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    ref_intermediate = torch.empty(
        (ref_gate_up.gate_up.shape[0], intermediate_size),
        dtype=ref_gate_up.gate_up.dtype,
        device=device,
    )
    self_intermediate = torch.empty_like(ref_intermediate)
    silu_and_mul(ref_gate_up.gate_up.view(-1, gate_up_size), ref_intermediate)
    silu_and_mul(self_gate_up.gate_up.view(-1, gate_up_size), self_intermediate)

    ref_result = pre_routed_fused_down_combine(
        ref_intermediate,
        topk_ids,
        topk_weights,
        ep_metadata,
        ref_workspace,
        ref_fused_metadata,
        down_weight,
        down_scale,
        expert_owner,
        local_expert_id,
        block_shape=block_shape,
        block_size=block_size,
        routed_scaling_factor=1.0,
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    self_result = self_routing_fused_down_combine(
        self_intermediate,
        self_gate_up,
        self_workspace,
        down_weight,
        down_scale,
        expert_owner,
        local_expert_id,
        block_shape=block_shape,
        block_size=block_size,
        routed_scaling_factor=1.0,
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()

    assert self_result.output.dtype == ref_result.output.dtype == torch.bfloat16
    torch.testing.assert_close(self_gate_up.topk_output.topk_ids, topk_ids)
    torch.testing.assert_close(self_gate_up.topk_output.topk_weights, topk_weights)
    torch.testing.assert_close(self_intermediate, ref_intermediate)
    torch.testing.assert_close(self_result.output, ref_result.output, atol=4e-2, rtol=5e-2)


@pytest.mark.parametrize("num_tokens", [4, 33])
def test_pre_routed_fused_down_combine_matches_s4_unfused_batches(
    num_tokens: int,
) -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel

    torch.manual_seed(8603 + num_tokens)
    device = torch.device("cuda")
    world_size = 1
    rank = 0
    num_local_experts = 3
    hidden_size = 32
    intermediate_size = 16
    top_k = 2
    block_shape = (16, 16)
    block_size = 16
    routed_scaling_factor = 0.75
    local_experts = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=device)
    pattern = local_experts[torch.arange(num_tokens * top_k, device=device) % 4]
    topk_ids = pattern.reshape(num_tokens, top_k).contiguous()
    topk_weights = torch.linspace(
        0.2,
        0.95,
        steps=num_tokens * top_k,
        dtype=torch.float32,
        device=device,
    ).reshape(topk_ids.shape)
    source_hidden = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
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
    workspace = _workspace(
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=num_tokens,
        top_k=top_k,
        device=device,
        dtype=source_hidden.dtype,
    )
    dispatch_plan = prepare_owner_directed_dispatch(ep_metadata, workspace)
    fused_metadata = build_pre_routed_fused_ep_metadata(
        source_hidden,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        dispatch_plan=dispatch_plan,
    )
    num_owner_rows = int(dispatch_plan.local_expert_offsets[-1].item())
    owner_intermediate = (
        torch.randn(num_owner_rows, intermediate_size, device=device) * 0.13
    ).to(torch.bfloat16)
    dense_weight = (
        torch.randn(num_local_experts, hidden_size, intermediate_size, device=device)
        * 0.11
    )
    local_down_weight, local_down_scale = _make_fp8_weight(dense_weight, block_shape)

    owner_outputs = owner_rank_fp8_expert_gemm(
        owner_intermediate,
        local_down_weight,
        local_down_scale,
        dispatch_plan.local_expert_counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
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
    expected = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
    )
    result = pre_routed_fused_down_combine(
        owner_intermediate,
        topk_ids,
        topk_weights,
        ep_metadata,
        workspace,
        fused_metadata,
        local_down_weight,
        local_down_scale,
        expert_owner,
        local_expert_id,
        block_shape=block_shape,
        block_size=block_size,
        routed_scaling_factor=routed_scaling_factor,
        expected_gemm_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()

    assert dispatch_plan.local_expert_counts[2].item() == 0
    torch.testing.assert_close(result.output, expected, atol=4e-2, rtol=5e-2)


@pytest.mark.parametrize("num_tokens", [5, 33])
def test_iris_pre_routed_fused_down_combine_matches_s4_unfused_one_rank(
    num_tokens: int,
) -> None:
    iris = pytest.importorskip("iris")
    _require_cdna4_gpu()
    import tokenspeed_kernel

    initialized_here = _ensure_one_rank_dist_initialized()
    if dist.get_world_size() != 1:
        pytest.skip("one-rank fused down+combine compile smoke requires world_size=1")

    ctx = iris.iris(heap_size=1 << 28)
    try:
        torch.manual_seed(8604 + num_tokens)
        device = ctx.get_device()
        world_size = 1
        rank = 0
        num_local_experts = 3
        hidden_size = 32
        intermediate_size = 16
        top_k = 2
        block_shape = (16, 16)
        block_size = 16
        routed_scaling_factor = 0.6
        expert_pattern = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
        flat_topk_ids = expert_pattern[
            torch.arange(num_tokens * top_k, device=device) % expert_pattern.numel()
        ]
        topk_ids = flat_topk_ids.reshape(num_tokens, top_k).contiguous()
        topk_weights = torch.linspace(
            0.2,
            0.95,
            steps=topk_ids.numel(),
            dtype=torch.float32,
            device=device,
        ).reshape(topk_ids.shape)
        source_hidden = torch.empty(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
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
            max_tokens_per_rank=num_tokens,
            top_k=top_k,
            device=device,
            dtype=source_hidden.dtype,
        )
        fused_workspace = EPCommunicationWorkspace.allocate(
            max_tokens_per_rank=num_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=source_hidden.dtype,
            iris_mode="required",
            iris_context=ctx,
        )
        ref_plan = prepare_owner_directed_dispatch(ep_metadata, ref_workspace)
        metadata_topk_weights = torch.zeros_like(topk_weights)
        fused_metadata = build_pre_routed_fused_ep_metadata(
            source_hidden,
            topk_ids,
            metadata_topk_weights,
            ep_metadata,
            fused_workspace,
            dispatch_plan=ref_plan,
        )
        owner_intermediate = (
            torch.randn(
                int(ref_plan.local_expert_offsets[-1].item()),
                intermediate_size,
                device=device,
            )
            * 0.13
        ).to(torch.bfloat16)
        dense_weight = (
            torch.randn(num_local_experts, hidden_size, intermediate_size, device=device)
            * 0.11
        )
        local_down_weight, local_down_scale = _make_fp8_weight(
            dense_weight,
            block_shape,
        )

        expected = _s4_down_combine_reference(
            owner_intermediate,
            topk_ids,
            topk_weights,
            ep_metadata,
            ref_workspace,
            ref_plan,
            local_down_weight,
            local_down_scale,
            expert_owner,
            local_expert_id,
            block_shape=block_shape,
            block_size=block_size,
            routed_scaling_factor=routed_scaling_factor,
        )
        result = pre_routed_fused_down_combine(
            owner_intermediate,
            topk_ids,
            topk_weights,
            ep_metadata,
            fused_workspace,
            fused_metadata,
            local_down_weight,
            local_down_scale,
            expert_owner,
            local_expert_id,
            block_shape=block_shape,
            block_size=block_size,
            routed_scaling_factor=routed_scaling_factor,
        )
        torch.cuda.synchronize()

        assert result.dispatch_plan.local_expert_counts[2].item() == 0
        torch.testing.assert_close(result.output, expected, atol=4e-2, rtol=5e-2)
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()
            if initialized_here:
                dist.destroy_process_group()


def test_iris_pre_routed_fused_down_combine_remote_zero_token_owner() -> None:
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
        intermediate_size = 16
        top_k = 2
        block_shape = (16, 16)
        block_size = 16
        max_tokens_per_rank = 2
        routed_scaling_factor = 0.5
        if rank == 0:
            topk_ids = torch.tensor(
                [[2, 2], [3, 2]],
                dtype=torch.int32,
                device=device,
            )
            topk_weights = torch.tensor(
                [[0.25, 0.5], [0.75, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            source_hidden = torch.empty(
                (2, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            owner_intermediate = torch.empty(
                (0, intermediate_size),
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            topk_ids = torch.empty((0, top_k), dtype=torch.int32, device=device)
            topk_weights = torch.empty((0, top_k), dtype=torch.float32, device=device)
            source_hidden = torch.empty(
                (0, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            owner_intermediate = _remote_owner_intermediate(device, intermediate_size)
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
            dtype=torch.bfloat16,
            iris_mode="required",
            iris_context=ctx,
        )
        dispatch_plan = prepare_owner_directed_dispatch(ep_metadata, workspace)
        fused_metadata = build_pre_routed_fused_ep_metadata(
            source_hidden,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
            dispatch_plan=dispatch_plan,
        )
        dense_weight = (
            torch.arange(
                num_local_experts * hidden_size * intermediate_size,
                dtype=torch.float32,
                device=device,
            ).reshape(num_local_experts, hidden_size, intermediate_size)
            * 0.0007
            - 0.11
        )
        local_down_weight, local_down_scale = _make_fp8_weight(
            dense_weight,
            block_shape,
        )

        result = pre_routed_fused_down_combine(
            owner_intermediate,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
            fused_metadata,
            local_down_weight,
            local_down_scale,
            expert_owner,
            local_expert_id,
            block_shape=block_shape,
            block_size=block_size,
            routed_scaling_factor=routed_scaling_factor,
        )
        torch.cuda.synchronize()

        if rank == 0:
            expected = _remote_down_combine_expected(
                local_down_weight,
                local_down_scale,
                block_shape=block_shape,
                block_size=block_size,
                routed_scaling_factor=routed_scaling_factor,
            )
            torch.testing.assert_close(result.output, expected, atol=4e-2, rtol=5e-2)
        else:
            assert result.output.shape == (0, hidden_size)
            assert result.dispatch_plan.local_expert_counts.tolist() == [3, 1]
    finally:
        try:
            ctx.barrier()
        finally:
            del ctx
            gc.collect()
            if initialized_here:
                dist.destroy_process_group()


def _s4_down_combine_reference(
    owner_intermediate: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ep_metadata,
    workspace: EPCommunicationWorkspace,
    dispatch_plan,
    local_down_weight: torch.Tensor,
    local_down_scale: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    block_size: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    owner_outputs = owner_rank_fp8_expert_gemm(
        owner_intermediate,
        local_down_weight,
        local_down_scale,
        dispatch_plan.local_expert_counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=dispatch_plan.local_expert_offsets,
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
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
    return ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
    )


def _remote_owner_intermediate(
    device: torch.device | str,
    intermediate_size: int,
) -> torch.Tensor:
    return (
        torch.arange(4 * intermediate_size, dtype=torch.float32, device=device)
        .reshape(4, intermediate_size)
        .mul(0.01)
        .to(torch.bfloat16)
    )


def _remote_down_combine_expected(
    local_down_weight: torch.Tensor,
    local_down_scale: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    block_size: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = local_down_weight.device
    counts = torch.tensor([3, 1], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
    owner_outputs = owner_rank_fp8_expert_gemm(
        _remote_owner_intermediate(device, local_down_weight.shape[2]),
        local_down_weight,
        local_down_scale,
        counts,
        block_shape=block_shape,
        block_size=block_size,
        local_expert_offsets=offsets,
        expected_kernel_name="gluon_fp8_local_experts_gfx950",
    )
    torch.cuda.synchronize()
    expected = torch.empty((2, local_down_weight.shape[1]), dtype=torch.bfloat16, device=device)
    expected[0] = (
        owner_outputs[0].float() * 0.25 + owner_outputs[1].float() * 0.5
    ).mul(routed_scaling_factor).to(torch.bfloat16)
    expected[1] = (
        owner_outputs[3].float() * 0.75 + owner_outputs[2].float() * 1.0
    ).mul(routed_scaling_factor).to(torch.bfloat16)
    return expected


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
        pytest.skip("remote fused down+combine test requires torchrun")
    if int(os.environ["WORLD_SIZE"]) != required_world_size:
        pytest.skip(f"requires world_size={required_world_size}")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend="nccl", init_method="env://")
    return True
