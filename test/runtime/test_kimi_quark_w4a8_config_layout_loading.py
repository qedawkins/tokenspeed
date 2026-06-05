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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_w4a8_quantization_config,
    quark_kimi_w8a8_quantization_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TokenSpeed runtime quantization imports require GPU platform detection",
)


def _make_spec(*, tp_rank: int = 1, tp_size: int = 2, ep_rank: int = 1, ep_size: int = 2):
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    return MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4 // ep_size,
        hidden_size=8,
        intermediate_size=12,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )


def _make_backend(spec=None):
    from tokenspeed.runtime.layers.moe.backends.w4a8_quark.triton import (
        W4A8QuarkTritonBackend,
    )
    from tokenspeed.runtime.layers.moe.core.types import BackendKey
    from tokenspeed.runtime.layers.quantization import W4A8QuarkConfig

    spec = spec or _make_spec()
    return W4A8QuarkTritonBackend(
        BackendKey("gfx950", "w4a8_quark", "triton"),
        spec,
        W4A8QuarkConfig.from_config(quark_kimi_w4a8_quantization_config()),
    )


def _pattern(rows: int, cols: int, offset: int) -> torch.Tensor:
    return (
        torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols)
        + offset
    )


def _params_dict(layer: nn.Module) -> dict[str, nn.Parameter]:
    prefix = "model.layers.0.mlp.experts."
    return {f"{prefix}{name}": param for name, param in layer.named_parameters()}


def test_quark_w4a8_config_detection_rejects_other_quark_schemas() -> None:
    from tokenspeed.runtime.layers.quantization import (
        W4A8QuarkConfig,
        get_quantization_config,
    )
    from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer

    w4a8 = quark_kimi_w4a8_quantization_config()
    assert W4A8QuarkConfig.override_quantization_method(w4a8, None) == "w4a8_quark"
    assert (
        W4A8QuarkConfig.override_quantization_method(w4a8, "w4a8_quark")
        == "w4a8_quark"
    )
    assert get_quantization_config("w4a8_quark") is W4A8QuarkConfig
    quant_config = W4A8QuarkConfig.from_config(w4a8)
    assert quant_config.get_name() == "w4a8_quark"
    assert should_ignore_quant_layer(
        "model.layers.0.self_attn.q_proj",
        quant_config.ignored_layers,
    )
    assert should_ignore_quant_layer(
        "model.layers.0.mlp.gate_proj",
        quant_config.ignored_layers,
    )
    assert not should_ignore_quant_layer(
        "model.layers.0.mlp.experts",
        quant_config.ignored_layers,
    )

    assert (
        W4A8QuarkConfig.override_quantization_method(
            quark_kimi_w8a8_quantization_config(),
            None,
        )
        is None
    )

    mxfp4_like = deepcopy(w4a8)
    mxfp4_like["global_quant_config"]["weight"] = {
        "dtype": "mxfp4",
        "group_size": 32,
        "is_dynamic": False,
    }
    assert W4A8QuarkConfig.override_quantization_method(mxfp4_like, None) is None

    broken = deepcopy(w4a8)
    broken["global_quant_config"]["weight"][1]["qscheme"] = "per_tensor"
    with pytest.raises(ValueError, match="staged fp8_e4m3 \\+ int4"):
        W4A8QuarkConfig.from_config(broken)


def test_quark_w4a8_selector_uses_distinct_backend_and_rejects_fused_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.core import selector as selector_module
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import W4A8QuarkConfig

    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)
    monkeypatch.setattr(
        selector_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setattr(selector_module, "_detect_arch", lambda: "gfx950")

    quant_config = W4A8QuarkConfig.from_config(quark_kimi_w4a8_quantization_config())
    with pytest.raises(RuntimeError, match="gfx950/w4a8_quark.*triton:unsupported"):
        select_backend(_make_spec(ep_rank=0, ep_size=1), quant_config)

    with pytest.raises(RuntimeError, match="gfx950/w4a8_quark.*triton:unsupported"):
        select_backend(
            _make_spec(ep_rank=0, ep_size=1),
            quant_config,
            routing_config={"moe_fused_features": {"self_routing"}},
        )


def test_quark_w4a8_allocates_packed_int4_weights_and_channel_scales() -> None:
    backend = _make_backend()
    layer = nn.Module()
    backend.create_layer_weights(layer)

    assert layer.w13_weight.shape == (2, 12, 4)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.shape == (2, 12, 1)
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w2_weight.shape == (2, 8, 3)
    assert layer.w2_weight.dtype == torch.uint8
    assert layer.w2_weight_scale.shape == (2, 8, 1)
    assert layer.w2_weight_scale.dtype == torch.float32
    assert layer.w13_weight_scale_2.shape == (2, 2)
    assert layer.w13_weight_scale_2.dtype == torch.float32
    assert layer.w2_weight_scale_2.shape == (2,)
    assert layer.w2_weight_scale_2.dtype == torch.float32
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None

    signature = backend.expert_weight_format_signature
    assert signature.expected_weight_shape((2, 12, 8)) == (2, 12, 4)
    assert signature.expected_scale_shape((2, 12, 8)) == (2, 12, 1)

    with pytest.raises(NotImplementedError, match="INT4 x dynamic-8-bit"):
        backend.forward(layer, None, None, 0, 0)


def test_quark_w4a8_loader_handles_tp_slices_ep_ownership_and_input_scale_skip() -> None:
    from tokenspeed.runtime.layers.moe.checkpoint import (
        ExpertCheckpointSchema,
        build_moe_checkpoint_loader,
    )

    backend = _make_backend()
    layer = nn.Module()
    backend.create_layer_weights(layer)
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

    w1 = _pattern(12, 4, 10)
    w3 = _pattern(12, 4, 90)
    w2 = _pattern(8, 6, 30)
    w1_scale = torch.arange(12, dtype=torch.float32) + 0.5
    w3_scale = (torch.arange(12, dtype=torch.float32) + 10.5).reshape(12, 1)
    w2_scale = torch.arange(8, dtype=torch.float32) + 20.5
    w1_stage_scale = torch.tensor(0.125, dtype=torch.float32)
    w3_stage_scale = torch.tensor(0.250, dtype=torch.float32)
    w2_stage_scale = torch.tensor(0.500, dtype=torch.float32)

    assert (
        loader.load("model.layers.0.mlp.experts.2.gate_proj.weight", w1)
        == "model.layers.0.mlp.experts.w13_weight"
    )
    loader.load("model.layers.0.mlp.experts.2.up_proj.weight", w3)
    loader.load("model.layers.0.mlp.experts.2.down_proj.weight", w2)
    loader.load("model.layers.0.mlp.experts.2.gate_proj.weight_scale", w1_scale)
    loader.load("model.layers.0.mlp.experts.2.up_proj.weight_scale", w3_scale)
    loader.load("model.layers.0.mlp.experts.2.down_proj.weight_scale", w2_scale)
    loader.load(
        "model.layers.0.mlp.experts.2.gate_proj.weight_scale_2",
        w1_stage_scale,
    )
    loader.load(
        "model.layers.0.mlp.experts.2.up_proj.weight_scale_2",
        w3_stage_scale,
    )
    loader.load(
        "model.layers.0.mlp.experts.2.down_proj.weight_scale_2",
        w2_stage_scale,
    )
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

    torch.testing.assert_close(layer.w13_weight[0, :6], w1[6:12])
    torch.testing.assert_close(layer.w13_weight[0, 6:12], w3[6:12])
    torch.testing.assert_close(layer.w2_weight[0], w2[:, 3:6])
    torch.testing.assert_close(layer.w13_weight_scale[0, :6, 0], w1_scale[6:12])
    torch.testing.assert_close(layer.w13_weight_scale[0, 6:12, 0], w3_scale[6:12, 0])
    torch.testing.assert_close(layer.w2_weight_scale[0, :, 0], w2_scale)
    torch.testing.assert_close(layer.w13_weight_scale_2[0], torch.tensor([0.125, 0.250]))
    torch.testing.assert_close(layer.w2_weight_scale_2[0], w2_stage_scale)

    assert torch.count_nonzero(layer.w13_weight[1]) == 0
    assert torch.count_nonzero(layer.w2_weight[1]) == 0
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None
