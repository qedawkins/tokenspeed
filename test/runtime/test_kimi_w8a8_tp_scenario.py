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

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the Kimi W8A8 TP scenario")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi W8A8 TP scenario")


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


def _forbid_triton_experts_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenspeed_kernel.ops.moe.gluon.experts_fp8_gfx950 as experts_mod

    def _fail_fallback(*args, **kwargs):
        pytest.fail("W8A8 per-channel expert GEMM used Triton fallback")

    monkeypatch.setattr(experts_mod, "_triton_experts_fallback", _fail_fallback)


def _init_w8a8_tp_layer(layer, device: torch.device) -> None:
    torch.manual_seed(2718)
    intermediate_per_partition = layer.w2_weight.shape[-1]
    gate_up_channels = layer.w13_weight.shape[1]
    gate_up_channel_scale = torch.linspace(
        0.04,
        1.55,
        steps=gate_up_channels,
        device=device,
    ).view(1, gate_up_channels, 1)
    down_channel_scale = torch.linspace(
        0.03,
        1.25,
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
        * 0.15
    )
    w2_dense = (
        torch.randn(
            layer.num_experts,
            layer.hidden_size,
            intermediate_per_partition,
            device=device,
        )
        * down_channel_scale
        * 0.13
    )
    w13_weight, w13_weight_scale = _per_channel_quantize_fp8(w13_dense)
    w2_weight, w2_weight_scale = _per_channel_quantize_fp8(w2_dense)
    layer.w13_weight.data.copy_(w13_weight)
    layer.w13_weight_scale.data.copy_(w13_weight_scale)
    layer.w2_weight.data.copy_(w2_weight)
    layer.w2_weight_scale.data.copy_(w2_weight_scale)


def _topk_output(
    *,
    device: torch.device,
    num_tokens: int,
    num_experts: int,
):
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput

    base_ids = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [3, 0],
            [0, 3],
            [1, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_ids = base_ids[:num_tokens].contiguous()
    topk_weights = torch.linspace(
        0.25,
        0.85,
        steps=num_tokens * topk_ids.shape[1],
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, topk_ids.shape[1])
    router_logits = torch.empty(num_tokens, num_experts, device=device)
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


class _IdentityAttention(torch.nn.Module):
    def forward(
        self,
        *,
        positions,
        hidden_states,
        ctx,
        out_cache_loc,
        comm_manager,
    ):
        del positions, ctx, out_cache_loc, comm_manager
        return hidden_states


class _NoCollectiveComm:
    def __init__(self, layer) -> None:
        self.input_layernorm = layer.input_layernorm
        self.post_attn_layernorm = layer.post_attention_layernorm

    def get_num_tokens(self, ctx):
        return ctx.input_num_tokens, ctx.input_num_tokens

    def input_reduce_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_attn_reduce_norm(self, hidden_states, residual, ctx):
        del ctx
        return self.post_attn_layernorm(hidden_states, residual)

    def pre_mlp_comm(self, hidden_states, ctx):
        del ctx
        return hidden_states.to(torch.bfloat16)

    def post_mlp_fused(self, hidden_states, residual, ctx):
        del ctx
        return hidden_states, residual

    def final_norm(self, hidden_states, residual, ctx, norm):
        del ctx
        hidden_states, _ = norm(hidden_states, residual)
        return hidden_states


def _make_tiny_kimi_language_model(device: torch.device):
    from transformers import DeepseekV3Config

    from tokenspeed.runtime.configs.kimi_k25_config import (
        KimiK25Config,
        KimiK25VisionConfig,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization import W8A8Fp8Config
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
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        n_shared_experts=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        topk_method="greedy",
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
            rank=1,
            world_size=2,
            attn_tp_size=1,
            attn_dp_size=2,
            dense_tp_size=1,
            dense_dp_size=2,
            moe_tp_size=2,
            moe_ep_size=1,
            moe_dp_size=1,
        ),
        quant_config=W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
        is_multimodal_active=False,
    ).to(device)
    layer = model.language_model.model.layers[0]
    layer.self_attn = _IdentityAttention().to(device)
    layer.comm_manager = _NoCollectiveComm(layer)
    return model


def _reference_kimi_language_logits(
    model,
    input_embeds: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    is_prefill: bool,
) -> torch.Tensor:
    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts

    hidden_states, residual = layer.comm_manager.input_reduce_norm(input_embeds, None)
    hidden_states = layer.self_attn(
        positions=torch.empty(0, device=input_embeds.device, dtype=torch.long),
        hidden_states=hidden_states,
        ctx=None,
        out_cache_loc=torch.empty(0, device=input_embeds.device, dtype=torch.long),
        comm_manager=layer.comm_manager,
    )
    hidden_states, residual = layer.comm_manager.post_attn_reduce_norm(
        hidden_states,
        residual,
        None,
    )
    hidden_states = layer.comm_manager.pre_mlp_comm(hidden_states, None)
    router_logits = layer.mlp.gate(hidden_states)
    topk_output = layer.mlp.topk(hidden_states, router_logits)
    moe_hidden = _reference_w8a8_moe(
        hidden_states,
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
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


def _init_kimi_language_weights(model) -> None:
    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts
    _init_w8a8_tp_layer(experts, experts.w13_weight.device)
    layer.mlp.gate.weight.data = (
        torch.randn_like(layer.mlp.gate.weight) * 0.05
    ).bfloat16()
    model.language_model.lm_head.weight.data = (
        torch.randn_like(model.language_model.lm_head.weight) * 0.04
    ).bfloat16()


def _run_kimi_language_logits_case(
    model,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    is_prefill: bool,
):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    device = input_lengths.device
    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    ctx = ForwardContext(
        attn_backend=SimpleNamespace(spec_num_tokens=1),
        token_to_kv_pool=None,
        bs=input_lengths.numel(),
        num_extends=input_lengths.numel() if is_prefill else 0,
        input_num_tokens=total_tokens,
        forward_mode=forward_mode,
    )
    input_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
    input_embeds = (
        torch.randn(total_tokens, model.config.hidden_size, device=device) * 0.10
    ).bfloat16()
    output = model(
        ctx,
        input_ids,
        torch.arange(total_tokens, device=device, dtype=torch.long),
        torch.arange(total_tokens, device=device, dtype=torch.long),
        input_lengths,
        input_embeds=input_embeds,
    )
    torch.cuda.synchronize()
    assert output.next_token_logits.shape == (input_lengths.numel(), 64)
    assert output.next_token_logits.dtype == torch.bfloat16
    assert output.next_token_logits.isfinite().all()
    expected = _reference_kimi_language_logits(
        model,
        input_embeds,
        input_lengths,
        is_prefill=is_prefill,
    )
    torch.testing.assert_close(
        output.next_token_logits.float(),
        expected.float(),
        atol=0.90,
        rtol=0.18,
        check_dtype=False,
    )
    return output.next_token_logits


def _logits_from_hidden(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    is_prefill: bool,
):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor

    logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=lm_head.weight.shape[0], model_type="kimi_k25"),
        skip_all_gather=True,
    )
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE,
        extend_seq_lens=input_lengths,
    )
    return logits_processor(input_ids, hidden_states, lm_head, metadata)


def _expected_logits(
    hidden_states: torch.Tensor,
    input_lengths: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    is_prefill: bool,
) -> torch.Tensor:
    if is_prefill:
        indices = torch.cumsum(input_lengths, dim=0) - 1
        hidden_states = hidden_states[indices]
    return torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)


def _select_existing_mla_kernels() -> tuple[str, str]:
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
    return decode.name, prefill.name


def _select_existing_w8a8_experts_kernel() -> str:
    from tokenspeed_kernel.signature import (
        ScaleFormat,
        dense_tensor_format,
        format_signature,
        tensor_format,
    )
    from tokenspeed_kernel.selection import select_kernel

    fp8_scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None) or getattr(
        torch,
        "float8_e4m3fnuz",
    )
    selected = select_kernel(
        "moe",
        "experts",
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=tensor_format(
                "scaled-fp8",
                fp8_dtype,
                scale=fp8_scale,
            ),
        ),
        features=frozenset({"dispatch_sorted"}),
    )
    return selected.name


def _reference_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    value_head_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.empty(
        q.shape[0],
        q.shape[1],
        value_head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    for row in range(q.shape[0]):
        rows = []
        for token in range(int(cache_seqlens[row].item())):
            page_id = token // 64
            token_slot = token % 64
            rows.append(kv_cache[page_table[row, page_id], token_slot])
        kv_rows = torch.stack(rows).float()
        scores = torch.einsum("hd,sd->hs", q[row].float(), kv_rows) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum(
            "hs,sv->hv",
            probs,
            kv_rows[:, :value_head_dim],
        ).to(q.dtype)
    return output


def _run_mla_decode_smoke(device: torch.device) -> None:
    from tokenspeed_kernel.ops.attention import mla_decode_with_kvcache

    torch.manual_seed(1618)
    q = (torch.randn(2, 4, 72, device=device) * 0.12).bfloat16()
    kv_cache = (torch.randn(4, 64, 72, device=device) * 0.12).bfloat16()
    page_table = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([65, 33], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / (72**0.5)
    actual = mla_decode_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        65,
        kv_lora_rank=64,
        qk_rope_head_dim=8,
        value_head_dim=32,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()
    expected = _reference_mla_decode(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        value_head_dim=32,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.04,
        rtol=0.02,
        check_dtype=False,
    )


def _reference_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: tuple[int, ...],
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lses = []
    offset = 0
    for seq_len in seq_lens:
        cur_q = q[offset : offset + seq_len].float()
        cur_k = k[offset : offset + seq_len].float()
        cur_v = v[offset : offset + seq_len].float()
        seq_out = []
        seq_lse = []
        q_idx = torch.arange(seq_len, device=q.device).view(-1, 1)
        k_idx = torch.arange(seq_len, device=q.device).view(1, -1)
        for head in range(q.shape[1]):
            scores = torch.matmul(cur_q[:, head], cur_k[:, head].T) * softmax_scale
            scores = scores.masked_fill(k_idx > q_idx, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            seq_out.append(torch.matmul(probs, cur_v[:, head]))
            seq_lse.append(torch.logsumexp(scores, dim=-1) * math.log2(math.e))
        outputs.append(torch.stack(seq_out, dim=1))
        lses.append(torch.stack(seq_lse, dim=1))
        offset += seq_len
    return torch.cat(outputs, dim=0).to(q.dtype), torch.cat(lses, dim=0)


def _run_mla_prefill_smoke(device: torch.device) -> None:
    from tokenspeed_kernel.ops.attention import mla_prefill

    torch.manual_seed(1414)
    seq_lens = (3, 2)
    total = sum(seq_lens)
    q = (torch.randn(total, 4, 72, device=device) * 0.12).bfloat16()
    k = (torch.randn(total, 4, 72, device=device) * 0.12).bfloat16()
    v = (torch.randn(total, 4, 32, device=device) * 0.12).bfloat16()
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    cum_seq_lens = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / (72**0.5)
    actual, actual_lse = mla_prefill(
        q,
        k,
        v,
        seq_lens_t,
        cum_seq_lens,
        max(seq_lens),
        batch_size=len(seq_lens),
        softmax_scale=softmax_scale,
        is_causal=True,
        return_lse=True,
    )
    torch.cuda.synchronize()
    expected, expected_lse = _reference_mla_prefill(
        q,
        k,
        v,
        seq_lens,
        softmax_scale,
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.04,
        rtol=0.02,
        check_dtype=False,
    )
    torch.testing.assert_close(
        actual_lse,
        expected_lse,
        atol=0.04,
        rtol=0.02,
        check_dtype=False,
    )


def test_kimi_w8a8_tp_prefill_decode_reach_logits_without_kernel_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.layer import MoELayer
    from tokenspeed.runtime.layers.quantization import W8A8Fp8Config

    _forbid_triton_experts_fallback(monkeypatch)
    decode_kernel, prefill_kernel = _select_existing_mla_kernels()
    assert decode_kernel == "gluon_mla_decode_gfx950"
    assert prefill_kernel == "gluon_mla_prefill_gfx950"
    assert _select_existing_w8a8_experts_kernel() == "gluon_fp8_local_experts_gfx950"

    torch.manual_seed(31415)
    device = torch.device("cuda", torch.cuda.current_device())
    _run_mla_decode_smoke(device)
    _run_mla_prefill_smoke(device)

    kimi_model = _make_tiny_kimi_language_model(device)
    _init_kimi_language_weights(kimi_model)
    kimi_experts = kimi_model.language_model.model.layers[0].mlp.experts
    assert type(kimi_model).__name__ == "KimiK25ForConditionalGeneration"
    assert type(kimi_model.language_model).__name__ == "DeepseekV3ForCausalLM"
    assert kimi_model.is_multimodal_active is False
    assert kimi_experts.backend.key.quant == "w8a8_fp8"
    assert kimi_experts.backend.key.impl == "triton"
    assert kimi_experts.tp_rank == 1
    assert kimi_experts.tp_size == 2
    assert kimi_experts.w13_weight.shape == (4, 24, 32)
    assert kimi_experts.w2_weight.shape == (4, 32, 12)
    _run_kimi_language_logits_case(
        kimi_model,
        total_tokens=5,
        input_lengths=torch.tensor([3, 2], device=device, dtype=torch.int32),
        is_prefill=True,
    )
    _run_kimi_language_logits_case(
        kimi_model,
        total_tokens=2,
        input_lengths=torch.ones(2, device=device, dtype=torch.int32),
        is_prefill=False,
    )

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
    _init_w8a8_tp_layer(layer, device)

    assert type(layer.backend).__name__ == "W8A8PerTokenPerChannelFp8TritonBackend"
    assert layer.backend.key.quant == "w8a8_fp8"
    assert layer.backend.key.impl == "triton"
    assert layer.w13_weight.shape == (4, 24, 32)
    assert layer.w2_weight.shape == (4, 32, 12)

    lm_head = torch.nn.Linear(layer.hidden_size, 19, bias=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    lm_head.weight.data.copy_(
        (
            torch.randn_like(lm_head.weight.float())
            * torch.linspace(0.04, 0.70, steps=19, device=device).view(19, 1)
        ).bfloat16()
    )

    cases = (
        (
            "prefill",
            torch.tensor([3, 2], device=device, dtype=torch.int32),
            torch.arange(5, device=device, dtype=torch.long),
            True,
        ),
        (
            "decode",
            torch.ones(2, device=device, dtype=torch.int32),
            torch.arange(2, device=device, dtype=torch.long),
            False,
        ),
    )
    for name, input_lengths, input_ids, is_prefill in cases:
        hidden_states = (
            torch.randn(input_ids.numel(), layer.hidden_size, device=device)
            * torch.linspace(
                0.05,
                0.95,
                steps=input_ids.numel(),
                device=device,
            ).view(input_ids.numel(), 1)
        ).bfloat16()
        topk_output = _topk_output(
            device=device,
            num_tokens=input_ids.numel(),
            num_experts=layer.num_experts,
        )

        actual_hidden = layer(
            hidden_states.contiguous(),
            topk_output,
            num_global_tokens=input_ids.numel(),
            max_num_tokens_per_gpu=input_ids.numel(),
        )
        actual = _logits_from_hidden(
            actual_hidden,
            input_ids,
            input_lengths,
            lm_head,
            is_prefill=is_prefill,
        ).next_token_logits
        torch.cuda.synchronize()

        expected_hidden = _reference_w8a8_moe(
            hidden_states,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_output.topk_ids,
            topk_output.topk_weights,
        )
        expected = _expected_logits(
            expected_hidden,
            input_lengths,
            lm_head,
            is_prefill=is_prefill,
        )

        assert actual.shape == (input_lengths.numel(), 19), name
        assert actual.isfinite().all(), name
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=0.40,
            rtol=0.12,
            check_dtype=False,
            msg=name,
        )
