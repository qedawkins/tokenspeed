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

"""Synthetic Kimi W4A8 Quark scenario coverage.

This is the W4A8 analog of the synthetic W8A8 Kimi TP scenario. The W4A8
runtime intentionally has no registered expert kernel yet, so the scenario
asserts that model construction rejects runtime backend selection before serving.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_w4a8_quantization_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU is required for the Kimi W4A8 Quark synthetic scenario",
)


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the Kimi W4A8 Quark synthetic scenario")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi W4A8 Quark scenario")


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


class _HotEmptyExpertTopK(torch.nn.Module):
    def __init__(self, *, top_k: int, num_experts: int) -> None:
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=top_k)
        self._top_k = top_k
        self._num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput

        base_ids = torch.tensor(
            [
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
            ],
            device=hidden_states.device,
            dtype=torch.int32,
        )
        topk_ids = base_ids[: hidden_states.shape[0], : self._top_k].contiguous()
        weights = torch.tensor(
            [
                [0.80, 0.20],
                [0.75, 0.25],
                [0.70, 0.30],
                [0.85, 0.15],
                [0.65, 0.35],
            ],
            device=hidden_states.device,
            dtype=torch.float32,
        )
        topk_weights = weights[: hidden_states.shape[0], : self._top_k].contiguous()
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    def empty_topk_output(
        self,
        device: torch.device,
        *,
        hidden_states: torch.Tensor | None = None,
        router_logits: torch.Tensor | None = None,
    ):
        from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput

        del hidden_states
        if router_logits is None:
            router_logits = torch.empty((0, self._num_experts), device=device)
        topk_weights = torch.empty((0, self._top_k), device=device, dtype=torch.float32)
        topk_ids = torch.empty((0, self._top_k), device=device, dtype=torch.int32)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)


def _make_tiny_kimi_w4a8_language_model(device: torch.device):
    from transformers import DeepseekV3Config

    from tokenspeed.runtime.configs.kimi_k25_config import (
        KimiK25Config,
        KimiK25VisionConfig,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization import W4A8QuarkConfig
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
    quant_config = W4A8QuarkConfig.from_config(quark_kimi_w4a8_quantization_config())
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
        quant_config=quant_config,
        is_multimodal_active=False,
    ).to(device)
    return model


def _pattern(shape: tuple[int, ...], *, offset: int, device: torch.device):
    values = torch.arange(
        int(torch.tensor(shape).prod().item()),
        device=device,
        dtype=torch.int32,
    ).reshape(shape)
    return ((values + offset) % 251).to(torch.uint8)


def _expert_checkpoint_tensors(experts, *, expert_id: int, device: torch.device):
    local_gate_rows = experts.w13_weight.shape[1] // 2
    gate_rows = local_gate_rows * experts.tp_size
    hidden_packed = experts.w13_weight.shape[2]
    down_rows = experts.w2_weight.shape[1]
    down_cols = experts.w2_weight.shape[2] * experts.tp_size
    base = 17 + expert_id * 29
    return {
        "gate_proj.weight": _pattern(
            (gate_rows, hidden_packed),
            offset=base,
            device=device,
        ),
        "up_proj.weight": _pattern(
            (gate_rows, hidden_packed),
            offset=base + 5,
            device=device,
        ),
        "down_proj.weight": _pattern(
            (down_rows, down_cols),
            offset=base + 11,
            device=device,
        ),
        "gate_proj.weight_scale": torch.linspace(
            0.25 + expert_id,
            1.25 + expert_id,
            steps=gate_rows,
            device=device,
            dtype=torch.float32,
        ),
        "up_proj.weight_scale": torch.linspace(
            1.50 + expert_id,
            2.50 + expert_id,
            steps=gate_rows,
            device=device,
            dtype=torch.float32,
        ),
        "down_proj.weight_scale": torch.linspace(
            2.75 + expert_id,
            3.75 + expert_id,
            steps=down_rows,
            device=device,
            dtype=torch.float32,
        ),
        "gate_proj.weight_scale_2": torch.tensor(
            0.125 + expert_id,
            device=device,
            dtype=torch.float32,
        ),
        "up_proj.weight_scale_2": torch.tensor(
            0.250 + expert_id,
            device=device,
            dtype=torch.float32,
        ),
        "down_proj.weight_scale_2": torch.tensor(
            0.500 + expert_id,
            device=device,
            dtype=torch.float32,
        ),
        "gate_proj.input_scale": torch.tensor(
            1.0 + expert_id,
            device=device,
            dtype=torch.float32,
        ),
        "down_proj.input_scale": torch.tensor(
            2.0 + expert_id,
            device=device,
            dtype=torch.float32,
        ),
    }


def _synthetic_quark_w4a8_checkpoint(model, device: torch.device):
    experts = model.language_model.model.layers[0].mlp.experts
    weights = []
    originals = {}
    for expert_id in range(experts.num_experts):
        tensors = _expert_checkpoint_tensors(
            experts,
            expert_id=expert_id,
            device=device,
        )
        originals[expert_id] = tensors
        for suffix, tensor in tensors.items():
            weights.append(
                (
                    f"model.layers.0.mlp.experts.{expert_id}.{suffix}",
                    tensor,
                )
            )
    return weights, originals


def _assert_expert_zero_loaded_tp_slice(experts, originals) -> None:
    local_gate_rows = experts.w13_weight.shape[1] // 2
    down_cols = experts.w2_weight.shape[2]
    tp_rank = experts.tp_rank
    expert = originals[0]

    torch.testing.assert_close(
        experts.w13_weight[0, :local_gate_rows],
        expert["gate_proj.weight"][
            local_gate_rows * tp_rank : local_gate_rows * (tp_rank + 1)
        ],
    )
    torch.testing.assert_close(
        experts.w13_weight[0, local_gate_rows:],
        expert["up_proj.weight"][
            local_gate_rows * tp_rank : local_gate_rows * (tp_rank + 1)
        ],
    )
    torch.testing.assert_close(
        experts.w2_weight[0],
        expert["down_proj.weight"][:, down_cols * tp_rank : down_cols * (tp_rank + 1)],
    )
    torch.testing.assert_close(
        experts.w13_weight_scale[0, :local_gate_rows, 0],
        expert["gate_proj.weight_scale"][
            local_gate_rows * tp_rank : local_gate_rows * (tp_rank + 1)
        ],
    )
    torch.testing.assert_close(
        experts.w13_weight_scale[0, local_gate_rows:, 0],
        expert["up_proj.weight_scale"][
            local_gate_rows * tp_rank : local_gate_rows * (tp_rank + 1)
        ],
    )
    torch.testing.assert_close(
        experts.w2_weight_scale[0, :, 0],
        expert["down_proj.weight_scale"],
    )
    torch.testing.assert_close(
        experts.w13_weight_scale_2[0],
        torch.tensor([0.125, 0.250], device=experts.w13_weight_scale_2.device),
    )
    torch.testing.assert_close(
        experts.w2_weight_scale_2[0],
        torch.tensor(0.500, device=experts.w2_weight_scale_2.device),
    )
    assert experts.w13_input_scale is None
    assert experts.w2_input_scale is None


def _install_no_collective_runtime(model) -> None:
    layer = model.language_model.model.layers[0]
    layer.mlp.gate.weight.data = (
        torch.randn_like(layer.mlp.gate.weight) * 0.05
    ).bfloat16()
    layer.self_attn = _IdentityAttention().to(next(model.parameters()).device)
    layer.comm_manager = _NoCollectiveComm(layer)
    layer.mlp.topk = _HotEmptyExpertTopK(
        top_k=layer.mlp.experts.top_k,
        num_experts=layer.mlp.experts.num_experts,
    )


def _run_kimi_language_case_expect_w4a8_guard(
    model,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    is_prefill: bool,
) -> None:
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

    with pytest.raises(NotImplementedError) as exc_info:
        model(
            ctx,
            input_ids,
            torch.arange(total_tokens, device=device, dtype=torch.long),
            torch.arange(total_tokens, device=device, dtype=torch.long),
            input_lengths,
            input_embeds=input_embeds,
        )

    _assert_w4a8_guard_message(str(exc_info.value))


def _assert_w4a8_guard_message(message: str) -> None:
    assert "Quark W4A8 MoE expert kernel is not registered" in message
    assert "quark-w4a8-int4" in message
    assert "INT4 x dynamic-8-bit" in message
    assert "scaled-fp8" not in message
    assert "mxfp4" not in message


def test_kimi_w4a8_quark_synthetic_checkpoint_rejects_runtime_selection():
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.utils import MoeBackend

    moe_utils.MOE_BACKEND = MoeBackend.AUTO
    torch.manual_seed(5150)
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(
        RuntimeError,
        match="gfx9(5|50)/w4a8_quark.*triton:unsupported",
    ):
        _make_tiny_kimi_w4a8_language_model(device)
