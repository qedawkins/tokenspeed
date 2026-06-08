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

import torch

from tokenspeed.runtime.layers.moe.backends.ep_ownership import (
    build_uniform_expert_owner_maps,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_down_combine import (
    mxfp4_ep_down_gemm_combine,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.ep_gate_up import (
    dispatch_mxfp4_hidden_states_gate_up,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.routing import (
    is_kimi_sigmoid_noaux_topk_config,
    select_kimi_sigmoid_noaux_topk,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import (
    Mxfp4Config,
    Mxfp4TritonKernelBackend,
    current_platform,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer


class Mxfp4TritonKernelEPBackend(Mxfp4TritonKernelBackend):
    """MXFP4 backend for owner-directed TP/EP execution.

    Unlike the local ``triton_kernel`` backend, this path keeps checkpoint-packed
    MXFP4 weights intact after loading because the owner-rank EP helpers consume
    those tensors directly.
    """

    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, Mxfp4Config):
            return False
        if should_ignore_quant_layer(
            prefix=spec.prefix,
            ignored_layers=getattr(quant_config, "ignored_layers", []) or [],
        ):
            return False
        if quant_config.is_w4a8_fp8 or not quant_config.is_checkpoint_mxfp4_serialized:
            return False
        platform = current_platform()
        return (
            platform.is_amd
            and platform.is_cdna4_plus
            and spec.ep_size > 1
            and spec.activation in {"silu", "swiglu"}
            and spec.num_experts % spec.ep_size == 0
        )

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    @property
    def returns_replicated_routed_output(self) -> bool:
        return True

    def process_weights_after_loading(self, layer) -> None:
        self._activation = layer.activation
        self._swiglu_arg = getattr(layer, "swiglu_arg", None)

    def forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        if self.spec.ep_size <= 1:
            raise RuntimeError("MXFP4 TP/EP backend requires ep_size > 1")
        if not _is_bypassed_topk_output(topk_output):
            raise ValueError("MXFP4 TP/EP backend requires bypassed Kimi top-k output")
        topk_config = topk_output.topk_config
        if not is_kimi_sigmoid_noaux_topk_config(topk_config):
            raise ValueError(
                "MXFP4 TP/EP backend currently supports Kimi sigmoid/noaux top-k"
            )
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        num_tokens = hidden_states.shape[0]
        num_global_tokens = int(num_global_tokens or 0)
        if num_tokens == 0 and num_global_tokens == 0:
            return hidden_states.new_zeros((0, self.spec.hidden_size))

        topk_weights, topk_ids = select_kimi_sigmoid_noaux_topk(
            topk_output.router_logits,
            top_k=topk_config.top_k,
            correction_bias=topk_config.correction_bias,
            renormalize=topk_config.renormalize,
            routed_scaling_factor=topk_config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                topk_config.apply_routed_scaling_factor_on_output
            ),
            topk_indices_dtype=torch.int32,
            hidden_states=hidden_states,
        )
        topk_ids = topk_ids.to(device=hidden_states.device, dtype=torch.int32).contiguous()
        topk_weights = topk_weights.to(
            device=hidden_states.device,
            dtype=torch.float32,
        ).contiguous()

        import tokenspeed_kernel

        expert_owner, local_expert_id = build_uniform_expert_owner_maps(
            num_experts=self.spec.num_experts,
            num_local_experts=self.spec.num_local_experts,
            device=hidden_states.device,
        )
        ep_metadata = tokenspeed_kernel.moe_dispatch(
            topk_ids,
            expert_owner,
            local_expert_id,
            self.spec.ep_rank,
            self.spec.ep_size,
            self.spec.num_local_experts,
            dtype=torch.int32,
            traits={"comm_strategy": "ep_metadata"},
            expected_kernel_name="gluon_ep_metadata_gfx950",
        )
        fallback_max_tokens_per_rank = (
            num_global_tokens + self.spec.ep_size - 1
        ) // self.spec.ep_size
        workspace = self.ensure_ep_workspace(
            max_tokens_per_rank=max(
                int(max_num_tokens_per_gpu or 0),
                num_tokens,
                fallback_max_tokens_per_rank,
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            iris_mode="auto",
        )

        if self._swiglu_arg is None:
            swiglu_alpha = 1.0
            swiglu_limit = None
            swiglu_beta = None
        else:
            swiglu_alpha = self._swiglu_arg.alpha
            swiglu_limit = self._swiglu_arg.limit
            swiglu_beta = getattr(layer, "swiglu_beta", None)
        gate_up_result = dispatch_mxfp4_hidden_states_gate_up(
            hidden_states,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
            layer.w13_weight,
            layer.w13_weight_scale,
            bias=getattr(layer, "w13_weight_bias", None),
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            output_dtype=hidden_states.dtype,
        )
        return mxfp4_ep_down_gemm_combine(
            gate_up_result.gate_up,
            topk_ids,
            topk_weights,
            ep_metadata,
            workspace,
            gate_up_result.fused_metadata,
            gate_up_result.dispatch_plan,
            layer.w2_weight,
            layer.w2_weight_scale,
            expert_owner,
            local_expert_id,
            bias=getattr(layer, "w2_weight_bias", None),
            routed_scaling_factor=1.0,
            expected_reduce_kernel_name="gluon_local_sum_reduce_gfx950",
        ).output


def _is_bypassed_topk_output(topk_output: object) -> bool:
    output_format = getattr(topk_output, "format", None)
    is_bypassed = getattr(output_format, "is_bypassed", None)
    return is_bypassed is not None and is_bypassed()


__all__ = ["Mxfp4TritonKernelEPBackend"]
