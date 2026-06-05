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

import pytest
import torch

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_w4a8_quantization_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TokenSpeed runtime imports require GPU platform detection",
)


def _set_auto_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.utils import MoeBackend

    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)


def _moe_spec(*, tp_rank: int = 0, tp_size: int = 1, ep_rank: int = 0, ep_size: int = 1):
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    return MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4 // ep_size,
        hidden_size=16,
        intermediate_size=24,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )


def _w4a8_quant_config():
    from tokenspeed.runtime.layers.quantization import W4A8QuarkConfig

    return W4A8QuarkConfig.from_config(quark_kimi_w4a8_quantization_config())


def test_w4a8_backend_selects_only_w4a8_quant_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_auto_backend(monkeypatch)
    from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import (
        Mxfp4TritonKernelBackend,
    )
    from tokenspeed.runtime.layers.moe.backends.w8a8_fp8.triton import (
        W8A8PerTokenPerChannelFp8TritonBackend,
    )
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.quantization import Mxfp4Config, W8A8Fp8Config

    spec = _moe_spec()
    with pytest.raises(RuntimeError, match="gfx9(5|50)/w4a8_quark.*triton:unsupported"):
        select_backend(spec, _w4a8_quant_config())
    w8a8_backend = select_backend(
        spec,
        W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
    )
    mxfp4_backend = select_backend(
        spec,
        Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )

    assert isinstance(w8a8_backend, W8A8PerTokenPerChannelFp8TritonBackend)
    assert w8a8_backend.key.quant == "w8a8_fp8"
    assert isinstance(mxfp4_backend, Mxfp4TritonKernelBackend)
    assert mxfp4_backend.key.quant == "mxfp4"


def test_w4a8_backend_rejects_ep_and_fused_feature_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_auto_backend(monkeypatch)
    from tokenspeed.runtime.layers.moe.core.selector import select_backend

    with pytest.raises(RuntimeError, match="gfx9(5|50)/w4a8_quark.*triton:unsupported"):
        select_backend(_moe_spec(ep_rank=0, ep_size=2), _w4a8_quant_config())

    with pytest.raises(RuntimeError, match="gfx9(5|50)/w4a8_quark.*triton:unsupported"):
        select_backend(
            _moe_spec(),
            _w4a8_quant_config(),
            routing_config={"moe_fused_features": {"self_routing"}},
        )


def test_w4a8_backend_allocates_runtime_layout_for_tp() -> None:
    from torch import nn
    from tokenspeed.runtime.layers.moe.backends.w4a8_quark.triton import (
        W4A8QuarkTritonBackend,
    )
    from tokenspeed.runtime.layers.moe.core.types import BackendKey

    backend = W4A8QuarkTritonBackend(
        BackendKey("gfx950", "w4a8_quark", "triton"),
        _moe_spec(tp_rank=1, tp_size=2),
        _w4a8_quant_config(),
    )
    layer = nn.Module()
    backend.create_layer_weights(layer)

    assert layer.w13_weight.shape == (4, 24, 8)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.shape == (4, 24, 1)
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w13_weight_scale_2.shape == (4, 2)
    assert layer.w2_weight.shape == (4, 16, 6)
    assert layer.w2_weight.dtype == torch.uint8
    assert layer.w2_weight_scale.shape == (4, 16, 1)
    assert layer.w2_weight_scale_2.shape == (4,)
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None


def test_w4a8_moe_layer_selection_fails_until_runtime_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_auto_backend(monkeypatch)
    from tokenspeed.runtime.layers.moe.layer import MoELayer

    with pytest.raises(
        RuntimeError,
        match="gfx9(5|50)/w4a8_quark.*triton:unsupported",
    ):
        MoELayer(
            top_k=2,
            num_experts=4,
            hidden_size=16,
            intermediate_size=24,
            quant_config=_w4a8_quant_config(),
            layer_index=0,
            prefix="model.layers.0.mlp",
            activation="silu",
        )
