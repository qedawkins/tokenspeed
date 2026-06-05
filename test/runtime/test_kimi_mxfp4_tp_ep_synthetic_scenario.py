# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""4-rank synthetic Kimi MXFP4 TP/EP scenario coverage.

Run with four visible idle AMD GPUs, for example:

    ROCR_VISIBLE_DEVICES=1,2,4,5 \
    python -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m pytest test/runtime/test_kimi_mxfp4_tp_ep_synthetic_scenario.py -q
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

import pytest
import torch
import torch.distributed as dist


def _require_four_rank_cdna4_gpu() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the Kimi MXFP4 TP/EP synthetic scenario")
    if torch.cuda.device_count() < 4:
        pytest.skip(
            "four visible ROCm devices are required for the Kimi MXFP4 TP/EP "
            f"synthetic scenario, got {torch.cuda.device_count()}"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("run this scenario with torchrun --nproc_per_node=4")

    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi MXFP4 TP/EP scenario")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size != 4:
        pytest.skip(f"expected 4 distributed ranks, got {world_size}")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def _record_kernel_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    import tokenspeed_kernel

    kernel_calls: dict[str, list[dict]] = {
        "dispatch": [],
        "combine": [],
        "quantize_mxfp4": [],
    }
    original_dispatch = tokenspeed_kernel.moe_dispatch
    original_combine = tokenspeed_kernel.moe_combine
    original_quantize_mxfp4 = tokenspeed_kernel.quantize_mxfp4

    def _record_dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        kernel_calls["dispatch"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_combine(*args, **kwargs):
        result = original_combine(*args, **kwargs)
        kernel_calls["combine"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_quantize_mxfp4(*args, **kwargs):
        result = original_quantize_mxfp4(*args, **kwargs)
        kernel_calls["quantize_mxfp4"].append({"scale_layout": kwargs.get("scale_layout")})
        return result

    def _forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("MXFP4 TP/EP scenario used an unsupported MoE fallback")

    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", _record_dispatch)
    monkeypatch.setattr(tokenspeed_kernel, "moe_combine", _record_combine)
    monkeypatch.setattr(tokenspeed_kernel, "quantize_mxfp4", _record_quantize_mxfp4)
    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _forbidden_kernel)
    monkeypatch.setattr(tokenspeed_kernel, "moe_experts", _forbidden_kernel)
    if hasattr(tokenspeed_kernel, "quantize_fp8"):
        monkeypatch.setattr(tokenspeed_kernel, "quantize_fp8", _forbidden_kernel)
    return kernel_calls


def _make_layer(rank: int, world_size: int, device: torch.device):
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.layer import MoELayer
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Mxfp4Config
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkey_patch_backend = getattr(moe_utils, "MOE_BACKEND", None)
    moe_utils.MOE_BACKEND = MoeBackend.AUTO
    global_server_args_dict["ep_num_redundant_experts"] = 0
    try:
        layer = MoELayer(
            top_k=8,
            num_experts=16,
            hidden_size=64,
            intermediate_size=128,
            quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
            layer_index=1,
            prefix="language_model.model.layers.1.mlp.experts",
            tp_rank=rank,
            tp_size=world_size,
            ep_rank=rank,
            ep_size=world_size,
            activation="silu",
            with_bias=True,
        ).to(device)
    finally:
        moe_utils.MOE_BACKEND = monkey_patch_backend
    layer.process_weights_after_loading(layer)
    return layer


def _make_global_weights(device: torch.device) -> dict[str, torch.Tensor]:
    num_experts = 16
    hidden_size = 64
    intermediate_per_rank = 32
    gate_up_rows = 2 * intermediate_per_rank
    packed_patterns = torch.tensor([0x11, 0x12, 0x21, 0x22], dtype=torch.uint8)

    def _packed_weight(out_features: int, in_features: int, *, offset: int) -> torch.Tensor:
        weight = torch.empty(
            (num_experts, out_features, in_features // 2),
            dtype=torch.uint8,
        )
        for expert in range(num_experts):
            weight[expert].fill_(int(packed_patterns[(expert + offset) % 4].item()))
        return weight

    def _e8m0_scales(out_features: int, in_features: int) -> torch.Tensor:
        scales = torch.full(
            (num_experts, out_features, in_features // 32),
            125,
            dtype=torch.uint8,
        )
        scales += (torch.arange(num_experts, dtype=torch.uint8) % 2).view(-1, 1, 1)
        return scales

    w13_bias = torch.zeros(num_experts, gate_up_rows, dtype=torch.float32)
    w2_bias = torch.zeros(num_experts, hidden_size, dtype=torch.float32)
    for expert in range(num_experts):
        w13_bias[expert] = 0.001 * expert
        w2_bias[expert] = 0.01 * expert

    weights = {
        "w13_weight": _packed_weight(gate_up_rows, hidden_size, offset=0),
        "w13_weight_scale": _e8m0_scales(gate_up_rows, hidden_size),
        "w2_weight": _packed_weight(hidden_size, intermediate_per_rank, offset=1),
        "w2_weight_scale": _e8m0_scales(hidden_size, intermediate_per_rank),
        "w13_weight_bias": w13_bias.bfloat16(),
        "w2_weight_bias": w2_bias.bfloat16(),
    }
    return {name: tensor.to(device) for name, tensor in weights.items()}


def _copy_local_weights(layer, global_weights: dict[str, torch.Tensor]) -> None:
    start = layer.ep_rank * layer.num_local_experts
    end = start + layer.num_local_experts
    for name, tensor in global_weights.items():
        getattr(layer, name).data.copy_(tensor[start:end])


def _router_logits(num_tokens: int, rank: int, device: torch.device) -> torch.Tensor:
    num_experts = 16
    owner_interleaved = torch.tensor(
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
        dtype=torch.long,
        device=device,
    )
    logits = torch.full(
        (num_tokens, num_experts),
        -8.0,
        dtype=torch.float32,
        device=device,
    )
    descending = torch.linspace(8.0, -4.0, steps=num_experts, device=device)
    for token in range(num_tokens):
        order = owner_interleaved.roll(shifts=token + rank)
        logits[token, order] = descending
    return logits.to(torch.bfloat16)


def _expected_mxfp4_tp_ep_output(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    global_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
        dequantize_mxfp4_activation,
        quantize_mxfp4_activation_reference,
    )
    from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
        dequantize_mxfp4_expert_weight,
        kimi_swiglu_gate_up,
    )
    from tokenspeed.runtime.layers.moe.backends.mxfp4.routing import (
        select_kimi_sigmoid_noaux_topk,
    )

    topk_weights, topk_ids = select_kimi_sigmoid_noaux_topk(
        router_logits,
        top_k=8,
        correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=2.827,
        apply_routed_scaling_factor_on_output=True,
        topk_indices_dtype=torch.int32,
    )
    packed_hidden, hidden_scale = quantize_mxfp4_activation_reference(hidden_states)
    dequant_hidden = dequantize_mxfp4_activation(
        packed_hidden,
        hidden_scale,
        logical_shape=tuple(hidden_states.shape),
    )
    w13 = dequantize_mxfp4_expert_weight(
        global_weights["w13_weight"],
        global_weights["w13_weight_scale"],
    )
    w2 = dequantize_mxfp4_expert_weight(
        global_weights["w2_weight"],
        global_weights["w2_weight_scale"],
    )
    returned_slots = torch.zeros(
        (*topk_ids.shape, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for token in range(topk_ids.shape[0]):
        hidden = dequant_hidden[token : token + 1]
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            gate_up = (
                hidden @ w13[expert].T
                + global_weights["w13_weight_bias"][expert].float()
            )
            activated = kimi_swiglu_gate_up(
                gate_up,
                output_dtype=hidden_states.dtype,
            )
            packed_intermediate, intermediate_scale = quantize_mxfp4_activation_reference(
                activated
            )
            dequant_intermediate = dequantize_mxfp4_activation(
                packed_intermediate,
                intermediate_scale,
                logical_shape=tuple(activated.shape),
            )
            down = (
                dequant_intermediate @ w2[expert].T
                + global_weights["w2_weight_bias"][expert].float()
            )
            returned_slots[token, slot].copy_(down.to(returned_slots.dtype)[0])
    weighted_slots = (
        returned_slots.float() * topk_weights.to(hidden_states.device).unsqueeze(-1)
    ).to(returned_slots.dtype)
    return weighted_slots.float().sum(dim=1).to(hidden_states.dtype)


def _run_mxfp4_tp_ep_case(
    layer,
    global_weights: dict[str, torch.Tensor],
    *,
    num_tokens: int,
    rank: int,
    world_size: int,
    kernel_calls: MutableMapping[str, list[dict]],
) -> None:
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig

    torch.manual_seed(9300 + rank * 17 + num_tokens)
    hidden_states = (
        torch.randn(num_tokens, layer.hidden_size, device=layer.w13_weight.device) * 0.05
    ).bfloat16()
    router_logits = _router_logits(num_tokens, rank, hidden_states.device)
    correction_bias = torch.zeros(layer.num_experts, device=hidden_states.device)
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=8,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            renormalize=True,
            correction_bias=correction_bias,
            routed_scaling_factor=2.827,
            apply_routed_scaling_factor_on_output=True,
        ),
    )
    dispatch_start = len(kernel_calls["dispatch"])
    combine_start = len(kernel_calls["combine"])
    quantize_start = len(kernel_calls["quantize_mxfp4"])

    actual = layer(
        hidden_states,
        topk_output,
        num_global_tokens=num_tokens * world_size,
        max_num_tokens_per_gpu=num_tokens,
    )
    torch.cuda.synchronize()
    expected = _expected_mxfp4_tp_ep_output(
        hidden_states,
        router_logits,
        correction_bias,
        global_weights,
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == torch.bfloat16
    assert actual.isfinite().all()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.35,
        rtol=0.18,
        check_dtype=False,
    )

    assert any(
        call.get("expected_kernel_name") == "gluon_ep_metadata_gfx950"
        and call.get("traits", {}).get("comm_strategy") == "ep_metadata"
        for call in kernel_calls["dispatch"][dispatch_start:]
    )
    assert any(
        call.get("expected_kernel_name") == "gluon_local_sum_reduce_gfx950"
        for call in kernel_calls["combine"][combine_start:]
    )
    assert len(kernel_calls["quantize_mxfp4"][quantize_start:]) >= 2


def test_kimi_mxfp4_tp_ep_synthetic_prefill_decode_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("iris")
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    kernel_calls = _record_kernel_calls(monkeypatch)
    layer = _make_layer(rank, world_size, device)
    global_weights = _make_global_weights(device)
    _copy_local_weights(layer, global_weights)

    assert type(layer.backend).__name__ == "Mxfp4TritonKernelEPBackend"
    assert layer.backend.key.quant == "mxfp4"
    assert layer.backend.key.impl == "triton_kernel_ep"
    assert layer.tp_rank == rank
    assert layer.tp_size == 4
    assert layer.ep_rank == rank
    assert layer.ep_size == 4
    assert layer.top_k == 8
    assert layer.topk_output_format.is_bypassed()
    assert layer.expert_weight_format_signature.name == "mxfp4_e2m1_block32"
    assert tuple(layer.w13_weight.shape) == (4, 64, 32)
    assert tuple(layer.w13_weight_scale.shape) == (4, 64, 2)
    assert tuple(layer.w2_weight.shape) == (4, 64, 16)
    assert tuple(layer.w2_weight_scale.shape) == (4, 64, 1)
    assert not hasattr(layer, "w13_weight_triton_tensor")
    assert not hasattr(layer, "w2_weight_triton_tensor")

    _run_mxfp4_tp_ep_case(
        layer,
        global_weights,
        num_tokens=5,
        rank=rank,
        world_size=world_size,
        kernel_calls=kernel_calls,
    )
    _run_mxfp4_tp_ep_case(
        layer,
        global_weights,
        num_tokens=1,
        rank=rank,
        world_size=world_size,
        kernel_calls=kernel_calls,
    )

    workspace = layer.backend._ep_workspace
    assert workspace is not None
    assert workspace.backend == "iris"
    assert workspace.world_size == world_size
    assert workspace.rank == rank
    dist.barrier()
