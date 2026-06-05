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

from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.triton_weights import (
    register_input_scale_placeholders,
)
from tokenspeed.runtime.layers.moe.backends.w4a8_quark.weights import (
    QUARK_W4A8_INT4_PER_CHANNEL_FORMAT,
    create_w4a8_quark_weights,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import W4A8QuarkConfig
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer
from tokenspeed.runtime.utils import round_up


class W4A8QuarkTritonBackend(MoEBackend):
    supported_arches = frozenset({"gfx950"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, W4A8QuarkConfig):
            return False
        if should_ignore_quant_layer(
            prefix=spec.prefix,
            ignored_layers=getattr(quant_config, "ignored_layers", []) or [],
        ):
            return False
        return spec.ep_size <= 1 and spec.activation in {"silu", "swiglu"}

    @property
    def expert_weight_format_signature(self):
        return QUARK_W4A8_INT4_PER_CHANNEL_FORMAT

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        hidden_padded = round_up(self.spec.hidden_size, 2)
        ispp = self.spec.intermediate_size // self.spec.tp_size
        ispp_padded = round_up(ispp, 2)
        create_w4a8_quark_weights(
            self,
            layer,
            self.spec.num_local_experts,
            hidden_padded,
            ispp_padded,
            with_bias=with_bias,
        )
        register_input_scale_placeholders(layer)

    def forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        del layer, hidden_states, topk_output, num_global_tokens, max_num_tokens_per_gpu
        raise NotImplementedError(
            "Quark W4A8 MoE expert kernel is not registered for dynamic 8-bit "
            "activations plus packed INT4 per-channel weights; a gfx950 "
            "INT4 x dynamic-8-bit expert kernel must be wired first"
        )


__all__ = ["W4A8QuarkTritonBackend"]
