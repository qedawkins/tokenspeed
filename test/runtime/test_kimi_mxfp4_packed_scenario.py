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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Synthetic MXFP4 packed scenario coverage.

This test stays out of baseline promotion until a small real packed MXFP4
checkpoint is available. It validates the current synthetic checkpoint-flow
status by loading checkpoint-format packed weights into the existing runtime
MoE layer, selecting the existing MoE route/experts contracts, and comparing
against a dequantized dense reference.
"""

from __future__ import annotations

import pytest
import torch


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the Kimi MXFP4 packed scenario")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi MXFP4 packed scenario")


def _init_packed_mxfp4_layer(layer) -> dict[str, torch.Tensor]:
    torch.manual_seed(9241)
    device = layer.w13_weight.device

    w13_weight = torch.randint(
        0,
        256,
        tuple(layer.w13_weight.shape),
        device=device,
        dtype=torch.uint8,
    )
    w2_weight = torch.randint(
        0,
        256,
        tuple(layer.w2_weight.shape),
        device=device,
        dtype=torch.uint8,
    )
    w13_scale = torch.randint(
        120,
        125,
        tuple(layer.w13_weight_scale.shape),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.randint(
        120,
        125,
        tuple(layer.w2_weight_scale.shape),
        device=device,
        dtype=torch.uint8,
    )
    w13_bias = (torch.randn_like(layer.w13_weight_bias.float()) * 0.01).bfloat16()
    w2_bias = (torch.randn_like(layer.w2_weight_bias.float()) * 0.01).bfloat16()

    layer.w13_weight.data.copy_(w13_weight)
    layer.w2_weight.data.copy_(w2_weight)
    layer.w13_weight_scale.data.copy_(w13_scale)
    layer.w2_weight_scale.data.copy_(w2_scale)
    layer.w13_weight_bias.data.copy_(w13_bias)
    layer.w2_weight_bias.data.copy_(w2_bias)

    return {
        "w13_weight": w13_weight.clone(),
        "w13_weight_scale": w13_scale.clone(),
        "w13_weight_bias": w13_bias.clone(),
        "w2_weight": w2_weight.clone(),
        "w2_weight_scale": w2_scale.clone(),
        "w2_weight_bias": w2_bias.clone(),
    }


def _reference_mxfp4_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    original_weights: dict[str, torch.Tensor],
    *,
    top_k: int,
) -> torch.Tensor:
    from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
        dequantize_mxfp4_expert_weight,
    )

    w13 = dequantize_mxfp4_expert_weight(
        original_weights["w13_weight"],
        original_weights["w13_weight_scale"],
        logical_shape=(
            original_weights["w13_weight"].shape[0],
            original_weights["w13_weight"].shape[1],
            hidden_states.shape[1],
        ),
    )
    w2 = dequantize_mxfp4_expert_weight(
        original_weights["w2_weight"],
        original_weights["w2_weight_scale"],
        logical_shape=(
            original_weights["w2_weight"].shape[0],
            original_weights["w2_weight"].shape[1],
            original_weights["w13_weight"].shape[1] // 2,
        ),
    )
    topk_weights, topk_ids = torch.topk(
        torch.softmax(router_logits.float(), dim=-1),
        top_k,
        dim=-1,
    )
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for token in range(hidden_states.shape[0]):
        hidden = hidden_states[token : token + 1].float()
        for slot in range(top_k):
            expert = int(topk_ids[token, slot].item())
            gate_up = (
                hidden @ w13[expert].T
                + original_weights["w13_weight_bias"][expert].float()
            )[0]
            gate = gate_up[0::2].clamp(max=7.0)
            up = gate_up[1::2].clamp(min=-7.0, max=7.0)
            activated = (
                gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
            ).reshape(1, w2.shape[-1])
            down = (
                activated @ w2[expert].T
                + original_weights["w2_weight_bias"][expert].float()
            )[0]
            output[token] += down * topk_weights[token, slot]
    return output.to(hidden_states.dtype)


def _select_existing_packed_mxfp4_contract_kernels() -> tuple[str, str, str]:
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    route = select_kernel(
        "moe",
        "route",
        format_signature(logits=dense_tensor_format(torch.bfloat16)),
        traits={"output_type": "ragged_metadata"},
    )
    gate_up = select_kernel(
        "moe",
        "experts",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=frozenset({"ragged_metadata", "dispatch_gemm"}),
    )
    down_combine = select_kernel(
        "moe",
        "experts",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=frozenset({"ragged_metadata", "gemm_combine"}),
    )
    return route.name, gate_up.name, down_combine.name


def test_kimi_mxfp4_packed_synthetic_scenario_matches_dequantized_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.layer import MoELayer
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Mxfp4Config
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)
    global_server_args_dict["ep_num_redundant_experts"] = 0
    assert _select_existing_packed_mxfp4_contract_kernels() == (
        "triton_kernels_routing",
        "triton_kernels_dispatch_gemm",
        "triton_kernels_gemm_combine",
    )

    device = torch.device("cuda", torch.cuda.current_device())
    layer = MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=64,
        intermediate_size=64,
        quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
        layer_index=0,
        prefix="layers.0.mlp",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        activation="silu",
        with_bias=True,
    ).to(device)
    original_weights = _init_packed_mxfp4_layer(layer)

    assert type(layer.backend).__name__ == "Mxfp4TritonKernelBackend"
    assert layer.backend.key.quant == "mxfp4"
    assert layer.backend.key.impl == "triton_kernel"
    assert layer.topk_output_format.is_bypassed()
    assert layer.expert_weight_format_signature.name == "mxfp4_e2m1_block32"

    layer.process_weights_after_loading(layer)
    assert hasattr(layer, "w13_weight_triton_tensor")
    assert hasattr(layer, "w2_weight_triton_tensor")

    torch.manual_seed(314159)
    hidden_states = (torch.randn(5, layer.hidden_size, device=device) * 0.15).bfloat16()
    router_logits = (
        torch.tensor(
            [
                [2.50, 1.50, -0.75, 0.25],
                [-0.50, 1.75, 0.10, 2.20],
                [0.90, -1.25, 2.40, 0.35],
                [1.15, 2.05, -0.35, 0.80],
                [-1.50, 0.40, 1.80, 2.10],
            ],
            device=device,
            dtype=torch.float32,
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(top_k=2),
    )

    actual = layer(
        hidden_states.contiguous(),
        topk_output,
        num_global_tokens=hidden_states.shape[0],
        max_num_tokens_per_gpu=hidden_states.shape[0],
    )
    torch.cuda.synchronize()
    expected = _reference_mxfp4_moe(
        hidden_states,
        router_logits,
        original_weights,
        top_k=2,
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == torch.bfloat16
    assert actual.isfinite().all()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.30,
        rtol=0.18,
        check_dtype=False,
    )
