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

import math

import pytest
import torch

from tokenspeed.runtime.layers.moe.backends.ep_combine import owner_directed_combine
from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
    owner_directed_dispatch,
    prepare_owner_directed_dispatch,
)
from tokenspeed.runtime.layers.moe.backends.ep_experts import owner_rank_fp8_expert_gemm
from tokenspeed.runtime.layers.moe.backends.ep_reduce import ep_weighted_reduce
from tokenspeed.runtime.layers.moe.backends.ep_workspace import (
    EPCommunicationWorkspace,
    EPWorkspaceError,
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for EP weighted reduce validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for EP weighted reduce validation")


def _expected_weighted_reduce(
    returned_slots: torch.Tensor,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    weighted_slots = (
        returned_slots.float() * topk_weights.float().unsqueeze(-1)
    ).to(returned_slots.dtype)
    return weighted_slots.float().sum(dim=1) * routed_scaling_factor


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "hidden_size"),
    [
        (0, 2, 16),
        (8, 2, 32),
        (32, 4, 32),
        (257, 2, 16),
    ],
)
def test_ep_weighted_reduce_matches_torch_reference(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> None:
    returned_slots = (
        torch.arange(
            num_tokens * top_k * hidden_size,
            dtype=torch.float32,
        )
        .reshape(num_tokens, top_k, hidden_size)
        .remainder(19)
        .sub(9)
        .to(torch.bfloat16)
    )
    if num_tokens:
        topk_weights = torch.linspace(
            0.0,
            1.0,
            steps=num_tokens * top_k,
            dtype=torch.float32,
        ).reshape(num_tokens, top_k)
    else:
        topk_weights = torch.empty((0, top_k), dtype=torch.float32)
    if num_tokens:
        topk_weights[::3, -1] = 0.0
    routed_scaling_factor = 0.75

    out = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        use_kernel=False,
    )

    expected = _expected_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_ep_weighted_reduce_rejects_shape_mismatch() -> None:
    returned_slots = torch.empty((2, 2, 4), dtype=torch.bfloat16)
    topk_weights = torch.empty((2, 3), dtype=torch.float32)

    with pytest.raises(EPWorkspaceError, match="topk_weights shape"):
        ep_weighted_reduce(returned_slots, topk_weights, use_kernel=False)


def test_ep_weighted_reduce_rejects_already_weighted_slots() -> None:
    returned_slots = torch.empty((2, 2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(EPWorkspaceError, match="expects unweighted"):
        ep_weighted_reduce(
            returned_slots,
            topk_weights,
            use_kernel=False,
            returned_slots_are_weighted=True,
        )


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "hidden_size", "routed_scaling_factor"),
    [
        (0, 2, 16, 0.75),
        (16, 2, 32, 1.0),
        (32, 4, 32, 0.5),
        (257, 2, 16, 1.25),
    ],
)
def test_ep_weighted_reduce_kernel_matches_reference(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    routed_scaling_factor: float,
) -> None:
    _require_cdna4_gpu()
    device = "cuda"
    returned_slots = (
        torch.randn(
            num_tokens,
            top_k,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * 0.2
    ).to(torch.bfloat16)
    topk_weights = torch.linspace(
        0.1,
        0.9,
        steps=max(1, num_tokens * top_k),
        device=device,
        dtype=torch.float32,
    )[: num_tokens * top_k].reshape(num_tokens, top_k)
    expected_kernel_name = (
        "gluon_local_sum_reduce_gfx950" if 0 < num_tokens <= 32 else None
    )

    out = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        use_kernel=True,
        expected_kernel_name=expected_kernel_name,
    )
    torch.cuda.synchronize()

    expected = _expected_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)


def test_ep_weighted_reduce_two_owner_oracle_matches_dense_reference() -> None:
    hidden_size = 8
    hidden_states = (
        torch.arange(3 * hidden_size, dtype=torch.float32).reshape(3, hidden_size) * 0.01
    ).to(torch.bfloat16)
    dense_weight = (
        torch.arange(4 * hidden_size * hidden_size, dtype=torch.float32)
        .reshape(4, hidden_size, hidden_size)
        .remainder(23)
        .sub(11)
        * 0.01
    )
    topk_ids = torch.tensor([[0, 2], [3, 1], [-1, 2]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.50, 0.90], [0.00, 0.40]],
        dtype=torch.float32,
    )
    expert_owner = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    local_expert_id = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    routed_scaling_factor = 0.5

    returned_slots = _two_owner_returned_slots(
        hidden_states,
        dense_weight,
        topk_ids,
        expert_owner,
        local_expert_id,
        world_size=2,
    )
    out = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        use_kernel=False,
    )

    expected = _dense_reference(
        hidden_states,
        dense_weight,
        topk_ids,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-2)


def test_ep_fp8_owner_pipeline_matches_dense_weighted_reference() -> None:
    _require_cdna4_gpu()
    torch.manual_seed(8606)
    import tokenspeed_kernel

    device = "cuda"
    num_tokens = 8
    top_k = 2
    num_experts = 4
    hidden_size = 32
    block_shape = (16, 16)
    block_size = 16
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [-1, 1],
            [3, 2],
            [0, -1],
            [2, 2],
            [1, 3],
            [3, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.linspace(
        0.15,
        0.95,
        steps=num_tokens * top_k,
        dtype=torch.float32,
        device=device,
    ).reshape(num_tokens, top_k)
    topk_weights = torch.where(topk_ids.ge(0), topk_weights, torch.zeros_like(topk_weights))
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32) * 0.18
    ).to(torch.bfloat16)
    dense_weight = (
        torch.randn(num_experts, hidden_size, hidden_size, device=device) * 0.14
    )
    fp8_weight, fp8_scale = _make_fp8_weight(dense_weight, block_shape)
    dequant_weight = _dequantize_fp8_weight(fp8_weight, fp8_scale, block_shape)
    expert_owner = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    local_expert_id = torch.arange(num_experts, dtype=torch.int32, device=device)
    metadata = tokenspeed_kernel.moe_dispatch(
        topk_ids,
        expert_owner,
        local_expert_id,
        0,
        1,
        num_experts,
        dtype=torch.int32,
        traits={"comm_strategy": "ep_metadata"},
        expected_kernel_name="gluon_ep_metadata_gfx950",
    )
    workspace = EPCommunicationWorkspace.allocate(
        max_tokens_per_rank=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        world_size=1,
        rank=0,
        dtype=torch.bfloat16,
        device=device,
        iris_mode="disabled",
    )
    dispatch_plan = prepare_owner_directed_dispatch(metadata, workspace)
    dispatch_step = owner_directed_dispatch(
        hidden_states,
        topk_ids,
        metadata,
        workspace,
        dispatch_plan=dispatch_plan,
    )
    owner_outputs = owner_rank_fp8_expert_gemm(
        dispatch_step.dispatch_buffer,
        fp8_weight,
        fp8_scale,
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
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
    )
    out = ep_weighted_reduce(
        returned_slots,
        topk_weights,
        expected_kernel_name="gluon_local_sum_reduce_gfx950",
    )
    torch.cuda.synchronize()

    expected = _dense_reference(
        hidden_states,
        dequant_weight,
        topk_ids,
        topk_weights,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)


def _dense_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    out = torch.zeros(
        (hidden_states.shape[0], weight.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for token in range(topk_ids.shape[0]):
        for topk_idx in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, topk_idx].item())
            if expert < 0 or expert >= weight.shape[0]:
                continue
            out[token] += (
                hidden_states[token].float()
                @ weight[expert].float().T
                * topk_weights[token, topk_idx].float()
            )
    return out * routed_scaling_factor


def _two_owner_returned_slots(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_owner: torch.Tensor,
    local_expert_id: torch.Tensor,
    *,
    world_size: int,
) -> torch.Tensor:
    returned_slots = torch.zeros(
        (*topk_ids.shape, weight.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    owner_rows: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for token in range(topk_ids.shape[0]):
        for topk_idx in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, topk_idx].item())
            if expert < 0 or expert >= expert_owner.numel():
                continue
            owner = int(expert_owner[expert].item())
            local_id = int(local_expert_id[expert].item())
            owner_rows.setdefault((owner, local_id), []).append((token, topk_idx))

    for owner in range(world_size):
        for local_id in range(int(local_expert_id.max().item()) + 1):
            for token, topk_idx in owner_rows.get((owner, local_id), []):
                expert = int(topk_ids[token, topk_idx].item())
                returned_slots[token, topk_idx] = (
                    hidden_states[token].float() @ weight[expert].float().T
                ).to(hidden_states.dtype)
    return returned_slots


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


def _dequantize_fp8_weight(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_shape
    num_experts, n_size, k_size = quantized.shape
    dense = torch.empty(quantized.shape, device=quantized.device, dtype=torch.float32)
    for expert in range(num_experts):
        for n_block, n_start in enumerate(range(0, n_size, block_n)):
            n_end = min(n_start + block_n, n_size)
            for k_block, k_start in enumerate(range(0, k_size, block_k)):
                k_end = min(k_start + block_k, k_size)
                dense[expert, n_start:n_end, k_start:k_end] = (
                    quantized[expert, n_start:n_end, k_start:k_end].float()
                    * scale[expert, n_block, k_block]
                )
    return dense
