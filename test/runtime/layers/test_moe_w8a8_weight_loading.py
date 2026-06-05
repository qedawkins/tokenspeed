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

from functools import partial

import pytest
import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.weight_loaders import (
    load_model_weight,
    load_per_channel_weight_scale,
)


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU visibility is required for W8A8 loader/runtime imports")


def _make_w8a8_backend(*, tp_rank: int, tp_size: int):
    from tokenspeed.runtime.layers.moe.backends.w8a8_fp8.triton import (
        W8A8PerTokenPerChannelFp8TritonBackend,
    )
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import W8A8Fp8Config

    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        intermediate_size=24,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=0,
        ep_size=1,
    )
    return W8A8PerTokenPerChannelFp8TritonBackend(
        BackendKey("amd", "w8a8_fp8", "triton"),
        spec,
        W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
    )


def _per_channel_quantize_fp8(
    dense: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.platform import current_platform

    fp8 = current_platform().fp8e4m3fn
    scale = torch.clamp(
        dense.float().abs().amax(dim=1, keepdim=True) / fp8.max,
        min=1e-6,
    )
    quantized = torch.clamp(
        dense.float() / scale,
        min=fp8.min,
        max=fp8.max,
    ).to(fp8.dtype)
    return quantized, scale


def _pattern(
    rows: int,
    cols: int,
    *,
    device: torch.device,
    offset: float,
) -> torch.Tensor:
    values = torch.arange(rows * cols, device=device, dtype=torch.float32).reshape(
        rows,
        cols,
    )
    row_scales = torch.linspace(
        0.03,
        1.40,
        steps=rows,
        device=device,
        dtype=torch.float32,
    ).view(rows, 1)
    return (values * 0.002 + offset) * row_scales


def _make_loader_test_layer() -> nn.Module:
    fp8_dtype = torch.float8_e4m3fn
    layer = nn.Module()
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(torch.zeros(2, 12, 8, dtype=fp8_dtype), requires_grad=False),
    )
    layer.register_parameter(
        "w13_weight_scale",
        nn.Parameter(torch.ones(2, 12, 1, dtype=torch.float32), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight",
        nn.Parameter(torch.zeros(2, 8, 6, dtype=fp8_dtype), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_scale",
        nn.Parameter(torch.ones(2, 8, 1, dtype=torch.float32), requires_grad=False),
    )
    layer.register_parameter("w13_input_scale", None)
    layer.register_parameter("w2_input_scale", None)

    weight_loader = partial(
        load_model_weight,
        tp_rank=1,
        is_bias=False,
        use_presharded_weights=False,
        do_transpose=False,
    )
    scale_loader = partial(
        load_per_channel_weight_scale,
        tp_rank=1,
        do_transpose=False,
    )
    layer.w13_weight.weight_loader = weight_loader
    layer.w2_weight.weight_loader = weight_loader
    layer.w13_weight_scale.weight_loader = scale_loader
    layer.w2_weight_scale.weight_loader = scale_loader
    return layer


def _params_dict(layer: nn.Module) -> dict[str, nn.Parameter]:
    prefix = "model.layers.0.mlp.experts."
    return {f"{prefix}{name}": param for name, param in layer.named_parameters()}


def _fp8_pattern(rows: int, cols: int, offset: float) -> torch.Tensor:
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (values * 0.003 + offset).to(torch.float8_e4m3fn)


def test_quark_w8a8_moe_checkpoint_loader_handles_tp_ep_and_input_scale_skip():
    _require_gpu()
    from tokenspeed.runtime.layers.moe.checkpoint import (
        ExpertCheckpointSchema,
        build_moe_checkpoint_loader,
    )

    layer = _make_loader_test_layer()
    loader = build_moe_checkpoint_loader(
        params_dict=_params_dict(layer),
        expert_schema=ExpertCheckpointSchema(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
        ),
        num_experts=4,
        ep_rank=1,
        ep_size=2,
    )

    assert not loader.matches("model.layers.0.mlp.experts.1.gate_proj.weight")
    assert loader.matches("model.layers.0.mlp.experts.2.gate_proj.weight")
    assert loader.matches("model.layers.0.mlp.experts.3.down_proj.weight_scale")

    w1 = _fp8_pattern(12, 8, 0.10)
    w3 = _fp8_pattern(12, 8, 0.30)
    w2 = _fp8_pattern(8, 12, 0.20)
    w1_scale = torch.arange(12, dtype=torch.float32) + 0.5
    w3_scale = (torch.arange(12, dtype=torch.float32) + 10.5).reshape(12, 1)
    w2_scale = torch.arange(8, dtype=torch.float32) + 20.5

    assert (
        loader.load("model.layers.0.mlp.experts.2.gate_proj.weight", w1)
        == "model.layers.0.mlp.experts.w13_weight"
    )
    loader.load("model.layers.0.mlp.experts.2.up_proj.weight", w3)
    loader.load("model.layers.0.mlp.experts.2.down_proj.weight", w2)
    loader.load("model.layers.0.mlp.experts.2.gate_proj.weight_scale", w1_scale)
    loader.load("model.layers.0.mlp.experts.2.up_proj.weight_scale", w3_scale)
    loader.load("model.layers.0.mlp.experts.2.down_proj.weight_scale", w2_scale)
    assert (
        loader.load(
            "model.layers.0.mlp.experts.2.gate_proj.input_scale",
            torch.tensor(1.25),
        )
        == "model.layers.0.mlp.experts.w13_input_scale"
    )
    assert (
        loader.load(
            "model.layers.0.mlp.experts.2.down_proj.input_scale",
            torch.tensor(2.25),
        )
        == "model.layers.0.mlp.experts.w2_input_scale"
    )

    torch.testing.assert_close(
        layer.w13_weight[0, :6].float(),
        w1[6:12].float(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        layer.w13_weight[0, 6:12].float(),
        w3[6:12].float(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        layer.w2_weight[0].float(),
        w2[:, 6:12].float(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(layer.w13_weight_scale[0, :6, 0], w1_scale[6:12])
    torch.testing.assert_close(layer.w13_weight_scale[0, 6:12, 0], w3_scale[6:12, 0])
    torch.testing.assert_close(layer.w2_weight_scale[0, :, 0], w2_scale)
    assert torch.count_nonzero(layer.w13_weight[1].float()) == 0
    assert torch.count_nonzero(layer.w2_weight[1].float()) == 0
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None


def test_per_channel_scale_loader_accepts_1d_checkpoint_scales() -> None:
    w13_scale = torch.nn.Parameter(torch.zeros(2, 24, 1, dtype=torch.float32))
    w2_scale = torch.nn.Parameter(torch.zeros(2, 16, 1, dtype=torch.float32))
    w1 = torch.arange(24, dtype=torch.float32)
    w3 = torch.arange(100, 124, dtype=torch.float32)
    w2 = torch.arange(200, 216, dtype=torch.float32)

    load_per_channel_weight_scale(
        w13_scale,
        w1,
        local_expert_id=1,
        shard_id="w1",
        tp_rank=1,
        do_transpose=False,
    )
    load_per_channel_weight_scale(
        w13_scale,
        w3,
        local_expert_id=1,
        shard_id="w3",
        tp_rank=1,
        do_transpose=False,
    )
    load_per_channel_weight_scale(
        w2_scale,
        w2,
        local_expert_id=1,
        shard_id="w2",
        tp_rank=1,
        do_transpose=False,
    )

    torch.testing.assert_close(w13_scale[1, :12, 0], w1[12:24])
    torch.testing.assert_close(w13_scale[1, 12:, 0], w3[12:24])
    torch.testing.assert_close(w2_scale[1, :, 0], w2)


def test_w8a8_backend_loads_fp8_weights_and_channel_scales_on_gpu() -> None:
    _require_gpu()
    torch.manual_seed(8406)
    device = torch.device("cuda", torch.cuda.current_device())
    backend = _make_w8a8_backend(tp_rank=1, tp_size=2)
    layer = nn.Module()
    backend.create_layer_weights(layer)
    layer.to(device)

    assert layer.w13_weight.shape == (2, 24, 16)
    assert layer.w13_weight_scale.shape == (2, 24, 1)
    assert layer.w2_weight.shape == (2, 16, 12)
    assert layer.w2_weight_scale.shape == (2, 16, 1)
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None

    w1, w1_scale = _per_channel_quantize_fp8(
        _pattern(24, 16, device=device, offset=0.10)
    )
    w3, w3_scale = _per_channel_quantize_fp8(
        _pattern(24, 16, device=device, offset=0.30)
    )
    w2, w2_scale = _per_channel_quantize_fp8(
        _pattern(16, 24, device=device, offset=0.20)
    )

    layer.w13_weight.weight_loader(layer.w13_weight, w1, "w1", local_expert_id=0)
    layer.w13_weight.weight_loader(layer.w13_weight, w3, "w3", local_expert_id=0)
    layer.w2_weight.weight_loader(layer.w2_weight, w2, "w2", local_expert_id=0)
    layer.w13_weight_scale.weight_loader(
        layer.w13_weight_scale,
        w1_scale.squeeze(-1),
        "w1",
        local_expert_id=0,
    )
    layer.w13_weight_scale.weight_loader(
        layer.w13_weight_scale,
        w3_scale,
        "w3",
        local_expert_id=0,
    )
    layer.w2_weight_scale.weight_loader(
        layer.w2_weight_scale,
        w2_scale.squeeze(-1),
        "w2",
        local_expert_id=0,
    )

    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        tensor = getattr(layer, name)
        assert tensor.device == device, name
    assert layer.w13_weight.dtype == w1.dtype
    assert layer.w2_weight.dtype == w2.dtype
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w2_weight_scale.dtype == torch.float32

    expected_w1 = w1[12:24].float() * w1_scale[12:24].float()
    expected_w3 = w3[12:24].float() * w3_scale[12:24].float()
    expected_w2 = w2[:, 12:24].float() * w2_scale.float()
    actual_w1 = layer.w13_weight[0, :12].float() * layer.w13_weight_scale[0, :12]
    actual_w3 = layer.w13_weight[0, 12:].float() * layer.w13_weight_scale[0, 12:]
    actual_w2 = layer.w2_weight[0].float() * layer.w2_weight_scale[0]

    torch.testing.assert_close(actual_w1, expected_w1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_w3, expected_w3, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_w2, expected_w2, atol=0.0, rtol=0.0)
