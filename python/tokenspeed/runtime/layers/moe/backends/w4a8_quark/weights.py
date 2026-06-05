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
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    ExpertLocalLayout,
    PackedExpertWeightFormat,
    PackedScaleGranularity,
)
from tokenspeed.runtime.utils import set_weight_attrs

QUARK_W4A8_INT4_PER_CHANNEL_FORMAT = PackedExpertWeightFormat(
    name="quark_w4a8_int4_per_channel",
    storage_dtype=torch.uint8,
    logical_dtype="int4",
    pack_factor=2,
    scale_dtype=torch.float32,
    scale_granularity=PackedScaleGranularity.CHANNEL,
    block_shape=None,
    expert_local_layout=ExpertLocalLayout.EXPERT_OUT_IN,
    transposed_from_logical=False,
)


def create_w4a8_quark_weights(
    backend: MoEBackend,
    layer: nn.Module,
    num_local_experts: int,
    hidden_size_padded: int,
    intermediate_size_per_partition_padded: int,
    with_bias: bool = False,
) -> None:
    w13_weight = nn.Parameter(
        torch.zeros(
            num_local_experts,
            2 * intermediate_size_per_partition_padded,
            hidden_size_padded // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)

    w13_weight_scale = nn.Parameter(
        torch.ones(
            num_local_experts,
            2 * intermediate_size_per_partition_padded,
            1,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)

    w2_weight = nn.Parameter(
        torch.zeros(
            num_local_experts,
            hidden_size_padded,
            intermediate_size_per_partition_padded // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight", w2_weight)

    w2_weight_scale = nn.Parameter(
        torch.ones(
            num_local_experts,
            hidden_size_padded,
            1,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    w13_weight_scale_2 = nn.Parameter(
        torch.ones(num_local_experts, 2, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

    w2_weight_scale_2 = nn.Parameter(
        torch.ones(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

    if with_bias:
        w13_weight_bias = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size_per_partition_padded,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        w2_weight_bias = nn.Parameter(
            torch.zeros(num_local_experts, hidden_size_padded, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)

    weight_loader = backend._make_weight_loader()
    scale_loader = backend._make_per_channel_scale_loader()
    per_tensor_loader = backend._per_tensor_scale_loader()
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w13_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w13_weight_scale_2, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w2_weight_scale_2, {"weight_loader": per_tensor_loader})
    if with_bias:
        set_weight_attrs(w13_weight_bias, {"weight_loader": weight_loader})
        set_weight_attrs(w2_weight_bias, {"weight_loader": weight_loader})


__all__ = [
    "QUARK_W4A8_INT4_PER_CHANNEL_FORMAT",
    "create_w4a8_quark_weights",
]
