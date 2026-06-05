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

"""Synthetic S4 FP8 EP scenario coverage.

No local Kimi K2.5 checkpoint or multi-process launch artifact is available in
this environment, so the true 4/8 GPU checkpoint scenario remains
artifact-gated. This test keeps S4 runtime evidence moving by using Kimi's
language-only path with synthetic FP8 block-scaled local EP expert weights,
local-owner routing, runtime-shaped MLA prefill/decode probes, explicit S4
kernel call-site checks, and a dense dequantized MoE reference.

Iris D2D is intentionally not exercised here: without an initialized
torch.distributed process group, EPCommunicationWorkspace(auto) resolves to the
single-process torch workspace. Distributed Iris D2D coverage remains in the S4
backend tests.
"""

from __future__ import annotations

from collections.abc import MutableMapping

import pytest
import torch
import torch.nn.functional as F

from test.runtime.test_kimi_fp8_tp_scenario import (
    _NoCollectiveComm,
    _S1GluonMLAAttention as _GluonMLAAttention,
    _S1GluonMLABackend as _GluonMLABackend,
    _dequantize_fp8_weight,
    _expected_logits,
    _make_fp8_weight,
    _make_mla_kv_pool,
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the Kimi FP8 EP scenario")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi FP8 EP scenario")


def _make_tiny_kimi_ep_language_model(
    device: torch.device,
    *,
    ep_size: int,
    ep_rank: int,
):
    from transformers import DeepseekV3Config

    from tokenspeed.runtime.configs.kimi_k25_config import (
        KimiK25Config,
        KimiK25VisionConfig,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization import Fp8Config
    from tokenspeed.runtime.models.kimi_k25 import KimiK25ForConditionalGeneration
    from tokenspeed.runtime.utils.env import global_server_args_dict

    global_server_args_dict["ep_num_redundant_experts"] = 0
    text_config = DeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=16,
        q_lora_rank=None,
        n_routed_experts=ep_size * 2,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        hidden_act="silu",
        rms_norm_eps=1e-5,
        pad_token_id=0,
        tie_word_embeddings=False,
        disable_quant_module=["self_attn"],
    )
    object.__setattr__(text_config, "n_shared_experts", None)
    object.__setattr__(text_config, "rope_scaling", None)
    vision_config = KimiK25VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        text_hidden_size=32,
    )
    config = KimiK25Config(
        text_config=text_config,
        vision_config=vision_config,
        language_only=True,
    )
    model = KimiK25ForConditionalGeneration(
        config,
        mapping=Mapping(
            rank=ep_rank,
            world_size=ep_size,
            attn_tp_size=1,
            attn_dp_size=ep_size,
            dense_tp_size=1,
            dense_dp_size=ep_size,
            moe_tp_size=1,
            moe_ep_size=ep_size,
            moe_dp_size=1,
        ),
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
        is_multimodal_active=False,
    ).to(device)
    layer = model.language_model.model.layers[0]
    layer.self_attn = _GluonMLAAttention(
        hidden_size=32,
        num_heads=4,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        value_head_dim=8,
    ).to(device=device, dtype=torch.bfloat16)
    layer.input_layernorm.to(dtype=torch.bfloat16)
    layer.post_attention_layernorm.to(dtype=torch.bfloat16)
    model.language_model.model.norm.to(dtype=torch.bfloat16)
    model.language_model.lm_head.to(dtype=torch.bfloat16)
    layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.to(torch.bfloat16)
    layer.comm_manager = _NoCollectiveComm(layer)
    return model


def _init_fp8_ep_layer(layer) -> None:
    torch.manual_seed(5100 + layer.ep_rank + layer.ep_size)
    block_shape = tuple(layer.backend.quant_config.weight_block_size)
    device = layer.w13_weight.device
    local_offsets = torch.arange(
        layer.num_local_experts,
        device=device,
        dtype=torch.float32,
    ).view(layer.num_local_experts, 1, 1)
    global_offsets = local_offsets + float(layer.ep_rank * layer.num_local_experts)
    gate_up_channels = layer.w13_weight.shape[1]
    intermediate_size = layer.w2_weight.shape[-1]
    gate_up_scale = torch.linspace(
        0.05,
        1.35,
        steps=gate_up_channels,
        device=device,
    ).view(1, gate_up_channels, 1)
    down_scale = torch.linspace(
        0.04,
        1.15,
        steps=layer.hidden_size,
        device=device,
    ).view(1, layer.hidden_size, 1)
    w13_dense = (
        torch.randn(
            layer.num_local_experts,
            gate_up_channels,
            layer.hidden_size,
            device=device,
        )
        * gate_up_scale
        * 0.13
    ) + global_offsets * 0.002
    w2_dense = (
        torch.randn(
            layer.num_local_experts,
            layer.hidden_size,
            intermediate_size,
            device=device,
        )
        * down_scale
        * 0.11
    ) - global_offsets * 0.0015
    w13_weight, w13_scale = _make_fp8_weight(w13_dense, block_shape)
    w2_weight, w2_scale = _make_fp8_weight(w2_dense, block_shape)
    layer.w13_weight.data.copy_(w13_weight)
    layer.w13_weight_scale_inv.data.copy_(w13_scale)
    layer.w2_weight.data.copy_(w2_weight)
    layer.w2_weight_scale_inv.data.copy_(w2_scale)


def _local_expert_range(experts) -> tuple[int, int]:
    start = experts.ep_rank * experts.num_local_experts
    return start, start + experts.num_local_experts


def _force_local_router(layer) -> None:
    experts = layer.mlp.experts
    local_start, local_end = _local_expert_range(experts)
    bias = torch.full(
        (experts.num_experts,),
        -10.0,
        device=layer.mlp.gate.weight.device,
        dtype=torch.float32,
    )
    bias[local_start:local_end] = torch.linspace(
        11.0,
        10.0,
        steps=experts.num_local_experts,
        device=bias.device,
    )
    layer.mlp.gate.weight.zero_()
    assert layer.mlp.gate.e_score_correction_bias is not None
    layer.mlp.gate.e_score_correction_bias.copy_(bias)


def _init_kimi_ep_language_weights(model) -> None:
    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts
    _init_fp8_ep_layer(experts)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "mlp.experts" in name:
                continue
            if parameter.dim() >= 2:
                parameter.copy_((torch.randn_like(parameter) * 0.04).to(parameter.dtype))
            elif "e_score_correction_bias" in name:
                parameter.zero_()
            elif "bias" in name:
                parameter.zero_()
            else:
                parameter.fill_(1.0)
        _force_local_router(layer)


def _assert_local_topk(experts, topk_ids: torch.Tensor) -> None:
    local_start, local_end = _local_expert_range(experts)
    assert bool(topk_ids.ge(local_start).all().item())
    assert bool(topk_ids.lt(local_end).all().item())


def _reference_fp8_ep_moe(
    hidden_states: torch.Tensor,
    experts,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    _assert_local_topk(experts, topk_ids)
    block_shape = tuple(experts.backend.quant_config.weight_block_size)
    local_start, _ = _local_expert_range(experts)
    local_topk_ids = topk_ids - local_start
    w13 = _dequantize_fp8_weight(
        experts.w13_weight,
        experts.w13_weight_scale_inv,
        block_shape,
    )
    w2 = _dequantize_fp8_weight(
        experts.w2_weight,
        experts.w2_weight_scale_inv,
        block_shape,
    )
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate_size = w2.shape[-1]
    for token in range(local_topk_ids.shape[0]):
        hidden = hidden_states[token : token + 1].float()
        for slot in range(local_topk_ids.shape[1]):
            expert = int(local_topk_ids[token, slot].item())
            gate_up = (hidden @ w13[expert].T)[0].to(torch.bfloat16)
            activated = (
                F.silu(gate_up[:intermediate_size]) * gate_up[intermediate_size:]
            ).reshape(1, intermediate_size)
            down = (activated.float() @ w2[expert].T)[0]
            output[token] += down * topk_weights[token, slot].float()
    return output.to(hidden_states.dtype)


def _reference_kimi_ep_language_logits(
    model,
    input_embeds: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    is_prefill: bool,
    attention_hidden_states: torch.Tensor,
) -> torch.Tensor:
    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts

    hidden_states, residual = layer.comm_manager.input_reduce_norm(input_embeds, None)
    assert hidden_states.shape == attention_hidden_states.shape
    hidden_states = attention_hidden_states
    hidden_states, residual = layer.comm_manager.post_attn_reduce_norm(
        hidden_states,
        residual,
        None,
    )
    hidden_states = layer.comm_manager.pre_mlp_comm(hidden_states, None)
    router_logits = layer.mlp.gate(hidden_states)
    topk_output = layer.mlp.topk(hidden_states, router_logits)
    moe_hidden = _reference_fp8_ep_moe(
        hidden_states,
        experts,
        topk_output.topk_ids,
        topk_output.topk_weights,
    )
    final_hidden = layer.comm_manager.final_norm(
        moe_hidden,
        residual,
        None,
        model.language_model.model.norm,
    )
    return _expected_logits(
        final_hidden,
        input_lengths,
        model.language_model.lm_head,
        is_prefill=is_prefill,
    )


def _run_kimi_ep_language_logits_case(
    model,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    is_prefill: bool,
    kernel_calls: MutableMapping[str, list[dict]],
) -> torch.Tensor:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    device = input_lengths.device
    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    attn_backend = _GluonMLABackend(page_size=64, max_context_len=64)
    token_to_kv_pool = _make_mla_kv_pool(device)
    if is_prefill:
        attn_backend.init_prefill_metadata(input_lengths)
        out_cache_loc = torch.arange(total_tokens, device=device, dtype=torch.int32)
    else:
        page_table = torch.arange(
            input_lengths.numel(),
            device=device,
            dtype=torch.int32,
        ).view(-1, 1)
        attn_backend.init_decode_metadata(page_table, input_lengths)
        out_cache_loc = page_table.flatten() * 64
    ctx = ForwardContext(
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        bs=input_lengths.numel(),
        num_extends=input_lengths.numel() if is_prefill else 0,
        input_num_tokens=total_tokens,
        forward_mode=forward_mode,
    )
    input_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
    input_embeds = (
        torch.randn(total_tokens, model.config.hidden_size, device=device) * 0.10
    ).bfloat16()
    route_start = len(kernel_calls["route"])
    dispatch_start = len(kernel_calls["dispatch"])
    experts_start = len(kernel_calls["experts"])
    combine_start = len(kernel_calls["combine"])
    output = model(
        ctx,
        input_ids,
        torch.arange(total_tokens, device=device, dtype=torch.long),
        out_cache_loc,
        input_lengths,
        input_embeds=input_embeds,
    )
    torch.cuda.synchronize()

    actual_route_calls = kernel_calls["route"][route_start:]
    actual_dispatch_calls = kernel_calls["dispatch"][dispatch_start:]
    actual_expert_calls = kernel_calls["experts"][experts_start:]
    actual_combine_calls = kernel_calls["combine"][combine_start:]
    expected = _reference_kimi_ep_language_logits(
        model,
        input_embeds,
        input_lengths,
        is_prefill=is_prefill,
        attention_hidden_states=model.language_model.model.layers[
            0
        ].comm_manager.last_attention_output,
    )

    assert output.next_token_logits.shape == (input_lengths.numel(), 64)
    assert output.next_token_logits.dtype == torch.bfloat16
    assert output.next_token_logits.isfinite().all()
    torch.testing.assert_close(
        output.next_token_logits.float(),
        expected.float(),
        atol=0.90,
        rtol=0.18,
        check_dtype=False,
    )
    if total_tokens == 0:
        assert not actual_route_calls
        assert not actual_dispatch_calls
        assert not actual_expert_calls
        assert not actual_combine_calls
    else:
        assert any(
            call.get("expected_kernel_name") == "gluon_grouped_biased_topk_gfx950"
            and call.get("traits", {}).get("biased") is True
            and call.get("traits", {}).get("grouped") is True
            and call.get("traits", {}).get("ep") is True
            for call in actual_route_calls
        )
        assert any(
            call.get("expected_kernel_name") == "gluon_ep_metadata_gfx950"
            and call.get("traits", {}).get("comm_strategy") == "ep_metadata"
            for call in actual_dispatch_calls
        )
        assert sum(
            call.get("expected_kernel_name") == "gluon_fp8_local_experts_gfx950"
            for call in actual_expert_calls
        ) >= 2
        assert any(
            call.get("expected_kernel_name") == "gluon_local_sum_reduce_gfx950"
            and call.get("traits", {}).get("comm_strategy") is None
            for call in actual_combine_calls
        )
    if is_prefill:
        assert attn_backend.prefill_calls >= 1
    else:
        assert attn_backend.decode_calls >= 1

    experts = model.language_model.model.layers[0].mlp.experts
    workspace = experts.backend._ep_workspace
    if total_tokens == 0:
        assert workspace is None
    else:
        assert workspace is not None
        assert workspace.backend == "torch"
        assert workspace.world_size == experts.ep_size
        assert workspace.rank == experts.ep_rank
    return output.next_token_logits


def _record_kernel_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    import tokenspeed_kernel

    kernel_calls: dict[str, list[dict]] = {
        "route": [],
        "dispatch": [],
        "experts": [],
        "combine": [],
    }
    original_route = tokenspeed_kernel.moe_route
    original_dispatch = tokenspeed_kernel.moe_dispatch
    original_experts = tokenspeed_kernel.moe_experts
    original_combine = tokenspeed_kernel.moe_combine

    def _record_route(*args, **kwargs):
        result = original_route(*args, **kwargs)
        kernel_calls["route"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        kernel_calls["dispatch"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_experts(*args, **kwargs):
        result = original_experts(*args, **kwargs)
        kernel_calls["experts"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "features": set(kwargs.get("features") or set()),
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

    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _record_route)
    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", _record_dispatch)
    monkeypatch.setattr(tokenspeed_kernel, "moe_experts", _record_experts)
    monkeypatch.setattr(tokenspeed_kernel, "moe_combine", _record_combine)
    return kernel_calls


def _select_existing_s4_kernels() -> tuple[str, str, str, str, str, str]:
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

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
    route = select_kernel(
        "moe",
        "route",
        format_signature(logits=dense_tensor_format(torch.float32)),
        traits={
            "output_type": "topk",
            "biased": True,
            "grouped": True,
            "ep": True,
            "num_expert_group": 1,
            "topk_group": 1,
            "topk": 2,
            "num_fused_shared_experts": 0,
        },
    )
    ep_metadata = select_kernel(
        "moe",
        "dispatch",
        format_signature(indices=dense_tensor_format(torch.int32)),
        traits={"comm_strategy": "ep_metadata"},
    )
    experts = select_kernel(
        "moe",
        "experts",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=frozenset({"dispatch_sorted"}),
    )
    combine = select_kernel(
        "moe",
        "combine",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"num_tokens": 8, "comm_strategy": None},
    )
    return (
        decode.name,
        prefill.name,
        route.name,
        ep_metadata.name,
        experts.name,
        combine.name,
    )


@pytest.mark.parametrize(
    ("ep_size", "ep_rank"),
    [(4, 1), (8, 3)],
    ids=["4-rank-ep", "8-rank-ep"],
)
def test_kimi_fp8_ep_synthetic_prefill_decode_reaches_logits(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
    ep_rank: int,
) -> None:
    _require_cdna4_gpu()
    kernel_calls = _record_kernel_calls(monkeypatch)

    assert _select_existing_s4_kernels() == (
        "gluon_mla_decode_gfx950",
        "gluon_mla_prefill_gfx950",
        "gluon_grouped_biased_topk_gfx950",
        "gluon_ep_metadata_gfx950",
        "gluon_fp8_local_experts_gfx950",
        "gluon_local_sum_reduce_gfx950",
    )

    torch.manual_seed(4110 + ep_size + ep_rank)
    device = torch.device("cuda", torch.cuda.current_device())
    kimi_model = _make_tiny_kimi_ep_language_model(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    _init_kimi_ep_language_weights(kimi_model)
    kimi_experts = kimi_model.language_model.model.layers[0].mlp.experts

    assert type(kimi_model).__name__ == "KimiK25ForConditionalGeneration"
    assert type(kimi_model.language_model).__name__ == "DeepseekV3ForCausalLM"
    assert kimi_model.is_multimodal_active is False
    assert type(kimi_experts.backend).__name__ == "Fp8TritonBackend"
    assert kimi_experts.backend.key.quant == "fp8"
    assert kimi_experts.backend.key.impl == "triton"
    assert kimi_experts.tp_rank == 0
    assert kimi_experts.tp_size == 1
    assert kimi_experts.ep_rank == ep_rank
    assert kimi_experts.ep_size == ep_size
    assert kimi_experts.num_experts == ep_size * 2
    assert kimi_experts.num_local_experts == 2
    assert tuple(kimi_experts.backend.quant_config.weight_block_size) == (16, 16)
    assert kimi_experts.w13_weight.shape == (2, 32, 32)
    assert kimi_experts.w2_weight.shape == (2, 32, 16)

    _run_kimi_ep_language_logits_case(
        kimi_model,
        total_tokens=5,
        input_lengths=torch.tensor([3, 2], device=device, dtype=torch.int32),
        is_prefill=True,
        kernel_calls=kernel_calls,
    )
    _run_kimi_ep_language_logits_case(
        kimi_model,
        total_tokens=2,
        input_lengths=torch.ones(2, device=device, dtype=torch.int32),
        is_prefill=False,
        kernel_calls=kernel_calls,
    )
