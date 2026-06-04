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

import torch

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.triton_common import (
    build_triton_gemms,
    triton_forward,
)
from tokenspeed.runtime.layers.moe.backends.triton_weights import (
    attach_dense_weight_pair,
    register_block_scale_inverses,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Fp8Config


class Fp8TritonBackend(MoEBackend):
    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not (
            isinstance(quant_config, Fp8Config)
            and quant_config.weight_block_size is not None
        ):
            return False
        if spec.ep_size <= 1:
            return True

        platform = _current_platform()
        return platform.is_amd and platform.is_cdna4_plus and spec.activation == "silu"

    def create_layer_weights(self, layer, *, with_bias: bool = False) -> None:
        ispp = attach_dense_weight_pair(
            self,
            layer,
            with_bias=with_bias,
            params_dtype=torch.float8_e4m3fn,
        )
        register_block_scale_inverses(
            self,
            layer,
            num_local_experts=self.spec.num_local_experts,
            hidden_size=self.spec.hidden_size,
            intermediate_size_per_partition=ispp,
            block_shape=self.quant_config.weight_block_size,
        )

        self._gate_up_gemm, self._down_gemm, self._get_config_func = build_triton_gemms(
            layer,
            self.spec,
            use_fp8_w8a8=True,
            block_shape=self.quant_config.weight_block_size,
            dtype_tag="fp8_w8a8",
            gate_up_B_scale=layer.w13_weight_scale_inv,
            down_B_scale=layer.w2_weight_scale_inv,
        )

    def forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        if self.spec.ep_size > 1:
            return self._ep_forward(
                layer,
                hidden_states,
                topk_output,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )

        del num_global_tokens
        del max_num_tokens_per_gpu
        return triton_forward(
            self._gate_up_gemm,
            self._down_gemm,
            self._get_config_func,
            layer.activation,
            layer,
            hidden_states,
            topk_output,
        )

    def _ep_forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        platform = _current_platform()
        if not (platform.is_amd and platform.is_cdna4_plus):
            raise RuntimeError("FP8 Triton EP path requires AMD CDNA4+")
        if layer.activation != "silu":
            raise ValueError(f"Unsupported EP FP8 activation: {layer.activation}")
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        from tokenspeed.runtime.layers.activation import silu_and_mul
        from tokenspeed.runtime.layers.moe.backends.ep_combine import (
            owner_directed_combine,
        )
        from tokenspeed.runtime.layers.moe.backends.ep_dispatch import (
            owner_directed_dispatch,
            prepare_owner_directed_dispatch,
        )
        from tokenspeed.runtime.layers.moe.backends.ep_experts import (
            owner_rank_fp8_expert_gemm,
        )
        from tokenspeed.runtime.layers.moe.backends.ep_reduce import ep_weighted_reduce
        import tokenspeed_kernel

        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights
        num_tokens = hidden_states.shape[0]
        num_global_tokens = int(num_global_tokens or 0)
        if num_tokens == 0 and num_global_tokens == 0:
            return hidden_states.new_zeros((0, self.spec.hidden_size))

        expert_owner, local_expert_id = _expert_ownership_tensors(
            num_experts=self.spec.num_experts,
            num_local_experts=self.spec.num_local_experts,
            device=hidden_states.device,
        )
        metadata = tokenspeed_kernel.moe_dispatch(
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
        dispatch_plan = prepare_owner_directed_dispatch(metadata, workspace)
        dispatch_step = owner_directed_dispatch(
            hidden_states,
            topk_ids,
            metadata,
            workspace,
            dispatch_plan=dispatch_plan,
        )

        block_shape = tuple(self.quant_config.weight_block_size)
        gate_up = owner_rank_fp8_expert_gemm(
            dispatch_step.dispatch_buffer,
            layer.w13_weight,
            layer.w13_weight_scale_inv,
            dispatch_plan.local_expert_counts,
            block_shape=block_shape,
            block_size=16,
            local_expert_offsets=dispatch_plan.local_expert_offsets,
            expected_kernel_name="gluon_fp8_local_experts_gfx950",
        )
        intermediate = torch.empty(
            (gate_up.shape[0], gate_up.shape[1] // 2),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        silu_and_mul(gate_up.view(-1, gate_up.shape[-1]), intermediate)

        owner_outputs = owner_rank_fp8_expert_gemm(
            intermediate,
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            dispatch_plan.local_expert_counts,
            block_shape=block_shape,
            block_size=16,
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
        return ep_weighted_reduce(returned_slots, topk_weights)


__all__ = ["Fp8TritonBackend"]


def _expert_ownership_tensors(
    *,
    num_experts: int,
    num_local_experts: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if num_local_experts <= 0:
        raise ValueError(
            f"num_local_experts must be positive, got {num_local_experts}"
        )
    if num_experts % num_local_experts != 0:
        raise ValueError(
            f"num_experts {num_experts} must be divisible by num_local_experts "
            f"{num_local_experts}"
        )
    experts = torch.arange(num_experts, dtype=torch.int32, device=device)
    return experts // num_local_experts, experts % num_local_experts


def _current_platform():
    from tokenspeed_kernel.platform import current_platform

    return current_platform()
