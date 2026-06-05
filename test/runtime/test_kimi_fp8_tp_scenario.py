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

"""Synthetic S1 FP8 TP scenario coverage.

No local Kimi K2.5 checkpoint is available in this environment, so the full
4/8-rank checkpoint scenario remains artifact-gated. This test keeps the S1
runtime evidence moving by using the existing Kimi language-only entry path with
synthetic FP8 block-scaled MoE weights, a real grouped-biased runtime route,
runtime-shaped MLA prefill/decode probes, explicit S1 kernel-selection checks,
and a dense dequantized MoE reference.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the Kimi FP8 TP scenario")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi FP8 TP scenario")


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


def _reference_fp8_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    w13 = _dequantize_fp8_weight(w13_weight, w13_weight_scale, block_shape)
    w2 = _dequantize_fp8_weight(w2_weight, w2_weight_scale, block_shape)
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate_size = w2.shape[-1]
    for token in range(topk_ids.shape[0]):
        hidden = hidden_states[token : token + 1].float()
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            gate_up = (hidden @ w13[expert].T)[0].to(torch.bfloat16)
            activated = (
                F.silu(gate_up[:intermediate_size]) * gate_up[intermediate_size:]
            ).reshape(1, intermediate_size)
            down = (activated.float() @ w2[expert].T)[0]
            output[token] += down * topk_weights[token, slot].float()
    return output.to(hidden_states.dtype)


def _init_fp8_tp_layer(layer) -> None:
    torch.manual_seed(2718)
    block_shape = tuple(layer.backend.quant_config.weight_block_size)
    device = layer.w13_weight.device
    intermediate_per_partition = layer.w2_weight.shape[-1]
    gate_up_channels = layer.w13_weight.shape[1]
    gate_up_scale = torch.linspace(
        0.04,
        1.55,
        steps=gate_up_channels,
        device=device,
    ).view(1, gate_up_channels, 1)
    down_scale = torch.linspace(
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
        * gate_up_scale
        * 0.14
    )
    w2_dense = (
        torch.randn(
            layer.num_experts,
            layer.hidden_size,
            intermediate_per_partition,
            device=device,
        )
        * down_scale
        * 0.12
    )
    w13_weight, w13_scale = _make_fp8_weight(w13_dense, block_shape)
    w2_weight, w2_scale = _make_fp8_weight(w2_dense, block_shape)
    layer.w13_weight.data.copy_(w13_weight)
    layer.w13_weight_scale_inv.data.copy_(w13_scale)
    layer.w2_weight.data.copy_(w2_weight)
    layer.w2_weight_scale_inv.data.copy_(w2_scale)


def _reference_kimi_language_logits(
    model,
    input_embeds: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    is_prefill: bool,
    attention_hidden_states: torch.Tensor,
) -> torch.Tensor:
    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts
    block_shape = tuple(experts.backend.quant_config.weight_block_size)

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
    moe_hidden = _reference_fp8_moe(
        hidden_states,
        experts.w13_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight,
        experts.w2_weight_scale_inv,
        topk_output.topk_ids,
        topk_output.topk_weights,
        block_shape=block_shape,
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


class _NoCollectiveComm:
    def __init__(self, layer) -> None:
        self.input_layernorm = layer.input_layernorm
        self.post_attn_layernorm = layer.post_attention_layernorm
        self.last_attention_output = None

    def get_num_tokens(self, ctx):
        return ctx.input_num_tokens, ctx.input_num_tokens

    def input_reduce_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def pre_attn_comm(self, hidden_states, ctx):
        del ctx
        return hidden_states

    def post_attn_reduce_norm(self, hidden_states, residual, ctx):
        del ctx
        self.last_attention_output = hidden_states
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


class _S1GluonMLABackend:
    """Test-local runtime-shaped MLA backend using existing S1 Gluon operators."""

    spec_num_tokens = 1

    def __init__(self, *, page_size: int, max_context_len: int) -> None:
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.chunked_prefill_metadata = None
        self.forward_decode_metadata = None
        self.prefill_calls = 0
        self.decode_calls = 0

    def init_prefill_metadata(self, seq_lens: torch.Tensor) -> None:
        cum_seq_lens = torch.zeros(
            seq_lens.numel() + 1,
            device=seq_lens.device,
            dtype=torch.int32,
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])
        self.chunked_prefill_metadata = SimpleNamespace(
            extend_seq_lens=seq_lens,
            extend_seq_lens_cpu=seq_lens.cpu(),
            extend_prefix_lens=torch.zeros_like(seq_lens),
            extend_prefix_lens_cpu=torch.zeros_like(seq_lens).cpu(),
            req_pool_indices=torch.arange(
                seq_lens.numel(),
                device=seq_lens.device,
                dtype=torch.int32,
            ),
            cum_extend_seq_lens=cum_seq_lens,
            max_extend_seq_len=int(seq_lens.max().item()) if seq_lens.numel() else 0,
            chunked_loop_num=0,
            chunk_kv_indices_list=[],
            chunked_seq_len=[],
            cu_chunked_seq_len=[],
            max_chunk_len_per_loop=[],
        )

    def init_decode_metadata(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        self.forward_decode_metadata = SimpleNamespace(
            block_kv_indices=page_table,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens,
            num_extends=0,
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        forward_mode,
        bs,
        save_kv_cache=True,
        **kwargs,
    ):
        del kwargs
        if forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
            )
        return self.forward_extend(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            save_kv_cache=save_kv_cache,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
    ):
        del layer, out_cache_loc, token_to_kv_pool, bs, save_kv_cache
        meta = self.chunked_prefill_metadata
        return self.forward_extend_chunked(
            q,
            k,
            v,
            scaling=1.0,
            logits_soft_cap=0.0,
            cum_seq_lens_q=meta.cum_extend_seq_lens,
            cum_seq_lens_kv=meta.cum_extend_seq_lens,
            max_q_len=meta.max_extend_seq_len,
            max_kv_len=meta.max_extend_seq_len,
            seq_lens=meta.extend_seq_lens,
            batch_size=meta.extend_seq_lens.numel(),
            causal=True,
        )[0].reshape(q.shape[0], q.shape[1] * v.shape[-1])

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
        out: torch.Tensor | None = None,
    ):
        del logits_soft_cap
        from tokenspeed_kernel.ops.attention import mla_prefill

        self.prefill_calls += 1
        return mla_prefill(
            q.reshape(q.shape[0], q.shape[1], q.shape[2]),
            k.reshape(k.shape[0], k.shape[1], k.shape[2]),
            v.reshape(v.shape[0], v.shape[1], v.shape[2]),
            seq_lens,
            cum_seq_lens_kv,
            max_kv_len,
            batch_size=batch_size,
            softmax_scale=scaling,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max_q_len,
            out=out,
        )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
    ):
        del v, bs
        from tokenspeed_kernel.ops.attention import mla_decode_with_kvcache

        if save_kv_cache and k is not None:
            kv_lora_rank = token_to_kv_pool.kv_lora_rank
            cache_k_nope = k[..., :kv_lora_rank]
            cache_k_rope = k[..., kv_lora_rank:]
            if cache_k_nope.dim() == 3 and cache_k_nope.shape[1] != 1:
                cache_k_nope = cache_k_nope[:, 0:1, :]
                cache_k_rope = cache_k_rope[:, 0:1, :]
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                cache_k_nope,
                cache_k_rope,
            )

        metadata = self.forward_decode_metadata
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.page_size,
            token_to_kv_pool.kv_cache_dim,
        )
        self.decode_calls += 1
        return mla_decode_with_kvcache(
            q,
            kv_cache,
            metadata.block_kv_indices,
            metadata.seq_lens_k,
            metadata.max_seq_len_k,
            kv_lora_rank=token_to_kv_pool.kv_lora_rank,
            qk_rope_head_dim=token_to_kv_pool.qk_rope_head_dim,
            value_head_dim=layer.v_head_dim,
            softmax_scale=layer.scaling,
        ).reshape(q.shape[0], q.shape[1] * layer.v_head_dim)


class _S1GluonMLAAttention(torch.nn.Module):
    """Small attention module that exercises S1 MLA kernels in Kimi's layer path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = value_head_dim
        self.query_dim = kv_lora_rank + qk_rope_head_dim
        self.layer_id = 0
        self.scaling = self.query_dim**-0.5
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.query_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * self.query_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * value_head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * value_head_dim, hidden_size, bias=False)

    def forward(
        self,
        *,
        positions,
        hidden_states,
        ctx,
        out_cache_loc,
        comm_manager,
    ):
        del positions, comm_manager
        q = self.q_proj(hidden_states).view(
            -1,
            self.num_heads,
            self.query_dim,
        )
        k = self.k_proj(hidden_states).view(
            -1,
            self.num_heads,
            self.query_dim,
        )
        v = self.v_proj(hidden_states).view(
            -1,
            self.num_heads,
            self.v_head_dim,
        )

        if ctx.forward_mode.is_decode():
            kv = k.clone()
            kv[..., : self.v_head_dim] = v
            attn = ctx.attn_backend.forward_decode(
                q,
                kv,
                None,
                self,
                out_cache_loc,
                ctx.token_to_kv_pool,
                ctx.bs,
                save_kv_cache=True,
            )
        else:
            meta = ctx.attn_backend.chunked_prefill_metadata
            attn, _ = ctx.attn_backend.forward_extend_chunked(
                q,
                k,
                v,
                self.scaling,
                0.0,
                cum_seq_lens_q=meta.cum_extend_seq_lens,
                cum_seq_lens_kv=meta.cum_extend_seq_lens,
                max_q_len=meta.max_extend_seq_len,
                max_kv_len=meta.max_extend_seq_len,
                seq_lens=meta.extend_seq_lens,
                batch_size=meta.extend_seq_lens.numel(),
                causal=True,
            )
            attn = attn.reshape(
                hidden_states.shape[0],
                self.num_heads * self.v_head_dim,
            )

        return self.o_proj(attn)


def _make_tiny_kimi_language_model(device: torch.device):
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
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_shared_experts=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=2,
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
            rank=1,
            world_size=4,
            attn_tp_size=1,
            attn_dp_size=4,
            dense_tp_size=1,
            dense_dp_size=4,
            moe_tp_size=4,
            moe_ep_size=1,
            moe_dp_size=1,
        ),
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
        is_multimodal_active=False,
    ).to(device)
    layer = model.language_model.model.layers[0]
    layer.self_attn = _S1GluonMLAAttention(
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


def _init_kimi_language_weights(model) -> None:
    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts
    _init_fp8_tp_layer(experts)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "mlp.experts" in name:
                continue
            if parameter.dim() >= 2:
                parameter.copy_((torch.randn_like(parameter) * 0.04).to(parameter.dtype))
            elif "e_score_correction_bias" in name:
                parameter.copy_(
                    torch.linspace(
                        -0.20,
                        0.20,
                        steps=parameter.numel(),
                        device=parameter.device,
                    )
                )
            elif "bias" in name:
                parameter.zero_()
            else:
                parameter.fill_(1.0)
        layer.mlp.gate.weight.copy_(
            (torch.randn_like(layer.mlp.gate.weight) * 0.05).to(
                layer.mlp.gate.weight.dtype
            )
        )
        if layer.mlp.gate.e_score_correction_bias is not None:
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.tensor(
                    [-0.35, 0.10, -0.05, 0.25],
                    device=layer.mlp.gate.e_score_correction_bias.device,
                )
            )


class _TinyMLAKVPool:
    def __init__(
        self,
        device: torch.device,
        *,
        total_slots: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> None:
        self.device = device
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.kv_buffer = [
            torch.zeros(
                total_slots,
                1,
                self.kv_cache_dim,
                device=device,
                dtype=torch.bfloat16,
            )
        ]

    def get_key_buffer(self, layer_id: int):
        return self.kv_buffer[layer_id]

    def set_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1).to(torch.bfloat16)
        self.kv_buffer[layer.layer_id][loc.long()] = cache_k


def _make_mla_kv_pool(device: torch.device, *, total_slots: int = 512):
    return _TinyMLAKVPool(
        device,
        total_slots=total_slots,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
    )


def _run_kimi_language_logits_case(
    model,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    is_prefill: bool,
    route_calls: list[dict],
):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    device = input_lengths.device
    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    attn_backend = _S1GluonMLABackend(page_size=64, max_context_len=64)
    token_to_kv_pool = _make_mla_kv_pool(device)
    if is_prefill:
        attn_backend.init_prefill_metadata(input_lengths)
        out_cache_loc = torch.arange(total_tokens, device=device, dtype=torch.int32)
    else:
        page_table = torch.arange(
            input_lengths.numel(), device=device, dtype=torch.int32
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
    route_start = len(route_calls)
    output = model(
        ctx,
        input_ids,
        torch.arange(total_tokens, device=device, dtype=torch.long),
        out_cache_loc,
        input_lengths,
        input_embeds=input_embeds,
    )
    torch.cuda.synchronize()
    assert output.next_token_logits.shape == (input_lengths.numel(), 64)
    assert output.next_token_logits.dtype == torch.bfloat16
    assert output.next_token_logits.isfinite().all()
    actual_route_calls = route_calls[route_start:]
    expected = _reference_kimi_language_logits(
        model,
        input_embeds,
        input_lengths,
        is_prefill=is_prefill,
        attention_hidden_states=model.language_model.model.layers[
            0
        ].comm_manager.last_attention_output,
    )
    torch.testing.assert_close(
        output.next_token_logits.float(),
        expected.float(),
        atol=0.90,
        rtol=0.18,
        check_dtype=False,
    )
    if total_tokens == 0:
        assert not actual_route_calls
    else:
        assert actual_route_calls, "Kimi runtime did not call MoE route"
        assert any(
            call.get("expected_kernel_name") == "gluon_grouped_biased_topk_gfx950"
            and call.get("traits", {}).get("biased") is True
            and call.get("traits", {}).get("grouped") is True
            for call in actual_route_calls
        )
    if is_prefill:
        assert attn_backend.prefill_calls >= 1
    else:
        assert attn_backend.decode_calls >= 1
    return output.next_token_logits


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


def _select_existing_s1_kernels() -> tuple[str, str, str, str, str, str]:
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
            "num_expert_group": 2,
            "topk_group": 1,
            "topk": 2,
            "num_fused_shared_experts": 0,
        },
    )
    dispatch = select_kernel(
        "moe",
        "dispatch",
        format_signature(indices=dense_tensor_format(torch.int32)),
        traits={"comm_strategy": "local"},
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
        dispatch.name,
        experts.name,
        combine.name,
    )


def test_kimi_fp8_tp_synthetic_prefill_decode_reaches_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel

    route_calls: list[dict] = []
    original_moe_route = tokenspeed_kernel.moe_route

    def _record_moe_route(*args, **kwargs):
        route_calls.append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return original_moe_route(*args, **kwargs)

    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _record_moe_route)

    assert _select_existing_s1_kernels() == (
        "gluon_mla_decode_gfx950",
        "gluon_mla_prefill_gfx950",
        "gluon_grouped_biased_topk_gfx950",
        "gluon_local_dispatch_gfx950",
        "gluon_fp8_local_experts_gfx950",
        "gluon_local_sum_reduce_gfx950",
    )

    torch.manual_seed(31415)
    device = torch.device("cuda", torch.cuda.current_device())
    kimi_model = _make_tiny_kimi_language_model(device)
    _init_kimi_language_weights(kimi_model)
    kimi_experts = kimi_model.language_model.model.layers[0].mlp.experts

    assert type(kimi_model).__name__ == "KimiK25ForConditionalGeneration"
    assert type(kimi_model.language_model).__name__ == "DeepseekV3ForCausalLM"
    assert kimi_model.is_multimodal_active is False
    assert type(kimi_experts.backend).__name__ == "Fp8TritonBackend"
    assert kimi_experts.backend.key.quant == "fp8"
    assert kimi_experts.backend.key.impl == "triton"
    assert kimi_experts.tp_rank == 1
    assert kimi_experts.tp_size == 4
    assert tuple(kimi_experts.backend.quant_config.weight_block_size) == (16, 16)
    assert kimi_experts.w13_weight.shape == (4, 32, 32)
    assert kimi_experts.w2_weight.shape == (4, 32, 16)

    _run_kimi_language_logits_case(
        kimi_model,
        total_tokens=5,
        input_lengths=torch.tensor([3, 2], device=device, dtype=torch.int32),
        is_prefill=True,
        route_calls=route_calls,
    )
    _run_kimi_language_logits_case(
        kimi_model,
        total_tokens=2,
        input_lengths=torch.ones(2, device=device, dtype=torch.int32),
        is_prefill=False,
        route_calls=route_calls,
    )
