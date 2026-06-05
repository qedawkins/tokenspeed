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

import pytest
import torch


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for W8A8 backend wiring validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for W8A8 backend wiring validation")


def _moe_spec(*, tp_rank: int = 0, tp_size: int = 1):
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    return MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4,
        hidden_size=32,
        intermediate_size=24,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=0,
        ep_size=1,
    )


def _per_channel_quantize_fp8(
    dense: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.platform import current_platform

    fp8 = current_platform().fp8e4m3fn
    scale = torch.clamp(
        dense.float().abs().amax(dim=2, keepdim=True) / fp8.max,
        min=1e-6,
    )
    quantized = torch.clamp(
        dense.float() / scale,
        min=fp8.min,
        max=fp8.max,
    ).to(fp8.dtype)
    return quantized, scale


def _w8a8_per_channel_matmul(
    A: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert: int,
) -> torch.Tensor:
    from tokenspeed_kernel.ops.gemm.fp8_utils import scaled_fp8_quant

    A_fp8, A_scale = scaled_fp8_quant(
        A.contiguous(),
        None,
        use_per_token_if_dynamic=True,
    )
    A_dequantized = A_fp8.float() * A_scale.float()
    weight_dequantized = weight[expert].float() * weight_scale[expert].float()
    return A_dequantized @ weight_dequantized.T


def _reference_w8a8_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate_size = w2_weight.shape[-1]
    for token in range(hidden_states.shape[0]):
        hidden = hidden_states[token : token + 1]
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            gate_up = _w8a8_per_channel_matmul(
                hidden,
                w13_weight,
                w13_weight_scale,
                expert,
            )[0].to(torch.bfloat16)
            activated = (
                torch.nn.functional.silu(gate_up[:intermediate_size])
                * gate_up[intermediate_size:]
            ).reshape(1, intermediate_size).to(torch.bfloat16)
            down = _w8a8_per_channel_matmul(
                activated,
                w2_weight,
                w2_weight_scale,
                expert,
            )[0]
            output[token] += (down * topk_weights[token, slot]).to(torch.bfloat16).float()
    return output.to(hidden_states.dtype)


def test_w8a8_moe_and_existing_contracts_select_without_new_backend_names() -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
    from tokenspeed.runtime.layers.moe.backends.unquantized.triton import (
        Bf16TritonBackend,
    )
    from tokenspeed.runtime.layers.moe.backends.w8a8_fp8.triton import (
        W8A8PerTokenPerChannelFp8TritonBackend,
    )
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.quantization import Fp8Config, W8A8Fp8Config
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    spec = _moe_spec()
    bf16_backend = select_backend(spec, None)
    fp8_backend = select_backend(
        spec,
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
    )
    w8a8_backend = select_backend(
        spec,
        W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
    )

    assert isinstance(bf16_backend, Bf16TritonBackend)
    assert bf16_backend.key.quant == "unquantized"
    assert bf16_backend.key.impl == "triton"
    assert isinstance(fp8_backend, Fp8TritonBackend)
    assert fp8_backend.key.quant == "fp8"
    assert fp8_backend.key.impl == "triton"
    assert isinstance(w8a8_backend, W8A8PerTokenPerChannelFp8TritonBackend)
    assert w8a8_backend.key.quant == "w8a8_fp8"
    assert w8a8_backend.key.impl == "triton"

    tp_spec = _moe_spec(tp_rank=1, tp_size=2)
    tp_w8a8_backend = select_backend(
        tp_spec,
        W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
    )
    assert isinstance(tp_w8a8_backend, W8A8PerTokenPerChannelFp8TritonBackend)
    assert tp_w8a8_backend.key.quant == "w8a8_fp8"
    assert tp_w8a8_backend.key.impl == "triton"
    assert tp_w8a8_backend.spec.tp_rank == 1
    assert tp_w8a8_backend.spec.tp_size == 2

    class _WeightProbe(torch.nn.Module):
        pass

    probe = _WeightProbe()
    tp_w8a8_backend.create_layer_weights(probe)
    intermediate_per_partition = tp_spec.intermediate_size // tp_spec.tp_size
    assert probe.w13_weight.shape == (
        tp_spec.num_local_experts,
        2 * intermediate_per_partition,
        tp_spec.hidden_size,
    )
    assert probe.w2_weight.shape == (
        tp_spec.num_local_experts,
        tp_spec.hidden_size,
        intermediate_per_partition,
    )
    assert probe.w13_weight_scale.shape == (
        tp_spec.num_local_experts,
        2 * intermediate_per_partition,
        1,
    )
    assert probe.w2_weight_scale.shape == (
        tp_spec.num_local_experts,
        tp_spec.hidden_size,
        1,
    )

    decode = select_kernel(
        "attention",
        "mla_decode_with_kvcache",
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            kv_cache=dense_tensor_format(torch.bfloat16),
        ),
        features=frozenset({"paged", "mla"}),
        traits={
            "num_q_heads": 4,
            "query_dim": 72,
            "kv_lora_rank": 64,
            "qk_rope_head_dim": 8,
            "value_head_dim": 32,
            "page_size": 64,
        },
    )
    prefill = select_kernel(
        "attention",
        "mla_prefill",
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k=dense_tensor_format(torch.bfloat16),
            v=dense_tensor_format(torch.bfloat16),
        ),
        features=frozenset({"mla"}),
        traits={
            "num_q_heads": 4,
            "num_kv_heads": 4,
            "query_dim": 72,
            "value_head_dim": 32,
            "is_causal": True,
            "return_lse": True,
        },
    )
    assert decode.name == "gluon_mla_decode_gfx950"
    assert prefill.name == "gluon_mla_prefill_gfx950"


def test_w8a8_tp_moe_sublayer_forward_matches_reference_with_decoder_topk() -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.layer import MoELayer
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
    from tokenspeed.runtime.layers.quantization import W8A8Fp8Config

    torch.manual_seed(8407)
    device = torch.device("cuda", torch.cuda.current_device())
    layer = MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=32,
        intermediate_size=24,
        quant_config=W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
        layer_index=0,
        prefix="layers.0.mlp",
        tp_rank=1,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
        activation="silu",
    ).to(device)

    intermediate_per_partition = layer.w2_weight.shape[-1]
    gate_up_channels = layer.w13_weight.shape[1]
    assert intermediate_per_partition == layer.intermediate_size // layer.tp_size
    assert gate_up_channels == 2 * intermediate_per_partition
    assert layer.w13_weight.shape == (
        layer.num_local_experts,
        2 * intermediate_per_partition,
        layer.hidden_size,
    )
    assert layer.w2_weight.shape == (
        layer.num_local_experts,
        layer.hidden_size,
        intermediate_per_partition,
    )
    assert layer.w13_weight_scale.shape == (
        layer.num_local_experts,
        2 * intermediate_per_partition,
        1,
    )
    assert layer.w2_weight_scale.shape == (
        layer.num_local_experts,
        layer.hidden_size,
        1,
    )

    hidden_states = (
        torch.randn(8, layer.hidden_size, device=device)
        * torch.linspace(0.04, 0.85, steps=8, device=device).view(8, 1)
    ).bfloat16()
    gate_up_channel_scale = torch.linspace(
        0.05,
        1.70,
        steps=gate_up_channels,
        device=device,
    ).view(1, gate_up_channels, 1)
    down_channel_scale = torch.linspace(
        0.03,
        1.40,
        steps=layer.hidden_size,
        device=device,
    ).view(1, layer.hidden_size, 1)
    w13_dense = (
        torch.randn(
            layer.num_experts,
            gate_up_channels,
            layer.hidden_size,
            device=device,
        )
        * gate_up_channel_scale
        * 0.16
    )
    w2_dense = (
        torch.randn(
            layer.num_experts,
            layer.hidden_size,
            intermediate_per_partition,
            device=device,
        )
        * down_channel_scale
        * 0.14
    )
    w13_weight, w13_weight_scale = _per_channel_quantize_fp8(w13_dense)
    w2_weight, w2_weight_scale = _per_channel_quantize_fp8(w2_dense)
    layer.w13_weight.data.copy_(w13_weight)
    layer.w13_weight_scale.data.copy_(w13_weight_scale)
    layer.w2_weight.data.copy_(w2_weight)
    layer.w2_weight_scale.data.copy_(w2_weight_scale)

    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [3, 0],
            [0, 3],
            [1, 0],
            [3, 1],
            [0, 1],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.2,
        0.9,
        steps=hidden_states.shape[0] * layer.top_k,
        device=device,
        dtype=torch.float32,
    ).view(hidden_states.shape[0], layer.top_k)
    topk_output = StandardTopKOutput(
        topk_weights,
        topk_ids,
        torch.empty(hidden_states.shape[0], layer.num_experts, device=device),
    )

    actual = layer(
        hidden_states.contiguous(),
        topk_output,
        num_global_tokens=hidden_states.shape[0],
        max_num_tokens_per_gpu=hidden_states.shape[0],
    )
    torch.cuda.synchronize()
    expected = _reference_w8a8_moe(
        hidden_states,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        topk_ids,
        topk_weights,
    )

    assert type(layer.backend).__name__ == "W8A8PerTokenPerChannelFp8TritonBackend"
    assert layer.backend.key.quant == "w8a8_fp8"
    assert layer.backend.key.impl == "triton"
    assert topk_ids.ne(2).all()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.12,
        rtol=0.08,
        check_dtype=False,
    )
