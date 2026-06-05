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

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for FP8 EP backend wiring validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for FP8 EP backend wiring validation")


def test_fp8_ep_selects_existing_triton_backend_key() -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=4,
    )
    backend = select_backend(
        spec,
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
    )

    assert isinstance(backend, Fp8TritonBackend)
    assert backend.key.quant == "fp8"
    assert backend.key.impl == "triton"


def test_fp8_ep_supports_requires_cdna4_silu(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=4,
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[16, 16],
    )

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=False, is_cdna4_plus=False),
    )
    assert not fp8_triton.Fp8TritonBackend.supports(spec, quant_config)

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )
    assert fp8_triton.Fp8TritonBackend.supports(spec, quant_config)
    unsupported_activation = MoELayerSpec(
        top_k=spec.top_k,
        num_experts=spec.num_experts,
        num_local_experts=spec.num_local_experts,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        activation="gelu",
        tp_rank=spec.tp_rank,
        tp_size=spec.tp_size,
        ep_rank=spec.ep_rank,
        ep_size=spec.ep_size,
    )
    assert not fp8_triton.Fp8TritonBackend.supports(
        unsupported_activation,
        quant_config,
    )


def test_fp8_ep_pre_routed_fused_support_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    def make_spec(
        *,
        activation: str = "silu",
        ep_size: int = 4,
        tp_size: int = 1,
    ) -> MoELayerSpec:
        return MoELayerSpec(
            top_k=2,
            num_experts=8,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=16,
            activation=activation,
            tp_rank=0,
            tp_size=tp_size,
            ep_rank=0,
            ep_size=ep_size,
        )

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[16, 16],
    )
    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )

    assert fp8_triton.Fp8TritonBackend._supports_pre_routed_fused_ep(
        make_spec(),
        quant_config,
    )
    assert fp8_triton.Fp8TritonBackend.supports(make_spec(), quant_config)
    assert not fp8_triton.Fp8TritonBackend._supports_pre_routed_fused_ep(
        make_spec(ep_size=1),
        quant_config,
    )
    assert fp8_triton.Fp8TritonBackend.supports(make_spec(ep_size=1), quant_config)
    assert not fp8_triton.Fp8TritonBackend._supports_pre_routed_fused_ep(
        make_spec(activation="gelu"),
        quant_config,
    )
    assert not fp8_triton.Fp8TritonBackend._supports_pre_routed_fused_ep(
        make_spec(tp_size=2),
        quant_config,
    )
    assert not fp8_triton.Fp8TritonBackend._supports_pre_routed_fused_ep(
        make_spec(),
        Fp8Config(is_checkpoint_fp8_serialized=True),
    )

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=False, is_cdna4_plus=False),
    )
    assert not fp8_triton.Fp8TritonBackend._supports_pre_routed_fused_ep(
        make_spec(),
        quant_config,
    )


def test_fp8_ep_self_routing_fused_support_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    def make_spec(
        *,
        activation: str = "silu",
        ep_size: int = 4,
        tp_size: int = 1,
    ) -> MoELayerSpec:
        return MoELayerSpec(
            top_k=2,
            num_experts=8,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=16,
            activation=activation,
            tp_rank=0,
            tp_size=tp_size,
            ep_rank=0,
            ep_size=ep_size,
        )

    def make_backend(spec: MoELayerSpec, routing_config=None):
        return fp8_triton.Fp8TritonBackend(
            key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
            spec=spec,
            quant_config=quant_config,
            routing_config=routing_config,
        )

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[16, 16],
    )
    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )

    assert make_backend(make_spec()).topk_output_format.is_standard()
    assert make_backend(
        make_spec(),
        routing_config={"moe_fused_features": {"self_routing"}},
    ).topk_output_format.is_bypassed()
    assert make_backend(
        make_spec(),
        routing_config={"features": "self_routing"},
    ).topk_output_format.is_bypassed()
    assert fp8_triton.Fp8TritonBackend._supports_self_routing_fused_ep(
        make_spec(),
        quant_config,
    )
    assert not fp8_triton.Fp8TritonBackend._supports_self_routing_fused_ep(
        make_spec(ep_size=1),
        quant_config,
    )
    assert make_backend(
        make_spec(ep_size=1),
        routing_config={"moe_fused_features": {"self_routing"}},
    ).topk_output_format.is_standard()
    assert not fp8_triton.Fp8TritonBackend._supports_self_routing_fused_ep(
        make_spec(activation="gelu"),
        quant_config,
    )
    assert not fp8_triton.Fp8TritonBackend._supports_self_routing_fused_ep(
        make_spec(tp_size=2),
        quant_config,
    )

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=False, is_cdna4_plus=False),
    )
    assert not fp8_triton.Fp8TritonBackend._supports_self_routing_fused_ep(
        make_spec(),
        quant_config,
    )


def test_fp8_triton_backend_self_routing_rejects_runtime_dtype_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for fused wiring validation")

    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig
    from tokenspeed.runtime.layers.quantization import Fp8Config

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )
    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=2,
    )
    backend = fp8_triton.Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
        routing_config={"moe_fused_features": {"self_routing"}},
    )
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=torch.empty((2, 32, 32), dtype=fp8_dtype),
        w13_weight_scale_inv=torch.empty((2, 2, 2), dtype=torch.float32),
        w2_weight=torch.empty((2, 32, 16), dtype=fp8_dtype),
        w2_weight_scale_inv=torch.empty((2, 2, 1), dtype=torch.float32),
    )
    hidden_states = torch.empty((2, 32), dtype=torch.float32)
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=torch.empty((2, 4), dtype=torch.float32),
        topk_config=TopKConfig(top_k=2),
    )

    with pytest.raises(ValueError, match="self-routing fused EP requires"):
        backend.forward(
            layer,
            hidden_states,
            topk_output,
            num_global_tokens=4,
            max_num_tokens_per_gpu=2,
        )


@pytest.mark.parametrize("forced_backend", [False, True])
def test_fp8_ep_selection_uses_existing_triton_backend_for_pre_routed_fused(
    monkeypatch: pytest.MonkeyPatch,
    forced_backend: bool,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core import selector as selector_module
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Fp8Config

    monkeypatch.setattr(
        selector_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setattr(selector_module, "_detect_arch", lambda: "gfx950")
    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )
    monkeypatch.setattr(
        moe_utils,
        "MOE_BACKEND",
        MoeBackend.TRITON if forced_backend else MoeBackend.AUTO,
    )
    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=4,
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[16, 16],
    )

    backend = selector_module.select_backend(spec, quant_config)

    assert isinstance(backend, fp8_triton.Fp8TritonBackend)
    assert backend.key.impl == "triton"
    assert backend._supports_pre_routed_fused_ep(spec, quant_config)


@pytest.mark.parametrize("forced_backend", [False, True])
def test_fp8_ep_selection_uses_existing_triton_backend_for_self_routing_fused(
    monkeypatch: pytest.MonkeyPatch,
    forced_backend: bool,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core import selector as selector_module
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Fp8Config

    monkeypatch.setattr(
        selector_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setattr(selector_module, "_detect_arch", lambda: "gfx950")
    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )
    monkeypatch.setattr(
        moe_utils,
        "MOE_BACKEND",
        MoeBackend.TRITON if forced_backend else MoeBackend.AUTO,
    )
    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=4,
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[16, 16],
    )

    backend = selector_module.select_backend(
        spec,
        quant_config,
        routing_config={"moe_fused_features": {"self_routing"}},
    )

    assert isinstance(backend, fp8_triton.Fp8TritonBackend)
    assert backend.key.impl == "triton"
    assert backend._supports_self_routing_fused_ep(spec, quant_config)
    assert backend.topk_output_format.is_bypassed()


def test_fp8_ep_selection_rejects_unsupported_pre_routed_fused_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core import selector as selector_module
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Fp8Config

    monkeypatch.setattr(
        selector_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setattr(selector_module, "_detect_arch", lambda: "gfx950")
    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)
    spec = MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="gelu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=4,
    )

    with pytest.raises(RuntimeError, match="triton:unsupported"):
        selector_module.select_backend(
            spec,
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[16, 16],
            ),
        )


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_fp8_triton_backend_ep_forward_matches_dense_reference(ep_rank: int) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(8707 + ep_rank)
    from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    device = "cuda"
    num_tokens = 6
    top_k = 2
    ep_size = 2
    num_local_experts = 2
    num_experts = ep_size * num_local_experts
    hidden_size = 32
    intermediate_size = 16
    block_shape = (16, 16)
    local_start = ep_rank * num_local_experts
    local_experts = torch.arange(
        local_start,
        local_start + num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    topk_ids = local_experts[(torch.arange(num_tokens * top_k, device=device) + ep_rank) % 2]
    topk_ids = topk_ids.reshape(num_tokens, top_k).contiguous()
    topk_weights = torch.linspace(
        0.2,
        0.9,
        steps=num_tokens * top_k,
        dtype=torch.float32,
        device=device,
    ).reshape(num_tokens, top_k)
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32) * 0.12
    ).to(torch.bfloat16)
    dense_w13 = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device)
        * 0.12
    )
    dense_w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device) * 0.12
    )
    fp8_w13, fp8_w13_scale = _make_fp8_weight(dense_w13, block_shape)
    fp8_w2, fp8_w2_scale = _make_fp8_weight(dense_w2, block_shape)
    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    backend = Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=list(block_shape),
        ),
    )
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=fp8_w13[local_start : local_start + num_local_experts],
        w13_weight_scale_inv=fp8_w13_scale[local_start : local_start + num_local_experts],
        w2_weight=fp8_w2[local_start : local_start + num_local_experts],
        w2_weight_scale_inv=fp8_w2_scale[local_start : local_start + num_local_experts],
    )
    topk_output = SimpleNamespace(topk_ids=topk_ids, topk_weights=topk_weights)

    out = backend.forward(
        layer,
        hidden_states,
        topk_output,
        num_global_tokens=num_tokens * ep_size,
        max_num_tokens_per_gpu=num_tokens,
    )
    torch.cuda.synchronize()

    expected = _dense_moe_reference(
        hidden_states,
        _dequantize_fp8_weight(fp8_w13, fp8_w13_scale, block_shape),
        _dequantize_fp8_weight(fp8_w2, fp8_w2_scale, block_shape),
        topk_ids,
        topk_weights,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


def test_fp8_triton_backend_self_routing_ep_forward_matches_dense_reference() -> None:
    _require_cdna4_gpu()
    torch.manual_seed(9404)
    from tokenspeed.runtime.layers.moe.backends.ep_self_routing import (
        self_routing_topk_from_bypassed,
    )
    from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig
    from tokenspeed.runtime.layers.quantization import Fp8Config

    device = "cuda"
    num_tokens = 6
    top_k = 2
    ep_rank = 0
    ep_size = 2
    num_local_experts = 2
    num_experts = ep_size * num_local_experts
    hidden_size = 32
    intermediate_size = 16
    block_shape = (16, 16)
    local_start = ep_rank * num_local_experts
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32) * 0.12
    ).to(torch.bfloat16)
    router_logits = torch.full(
        (num_tokens, num_experts),
        -8.0,
        dtype=torch.float32,
        device=device,
    )
    router_logits[:, 0] = torch.linspace(2.0, 3.0, steps=num_tokens, device=device)
    router_logits[:, 1] = torch.linspace(1.0, 1.5, steps=num_tokens, device=device)
    dense_w13 = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device)
        * 0.12
    )
    dense_w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device) * 0.12
    )
    fp8_w13, fp8_w13_scale = _make_fp8_weight(dense_w13, block_shape)
    fp8_w2, fp8_w2_scale = _make_fp8_weight(dense_w2, block_shape)
    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    backend = Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=list(block_shape),
        ),
        routing_config={"moe_fused_features": {"self_routing"}},
    )
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=fp8_w13[local_start : local_start + num_local_experts],
        w13_weight_scale_inv=fp8_w13_scale[local_start : local_start + num_local_experts],
        w2_weight=fp8_w2[local_start : local_start + num_local_experts],
        w2_weight_scale_inv=fp8_w2_scale[local_start : local_start + num_local_experts],
    )
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=top_k,
            routed_scaling_factor=1.0,
            apply_routed_scaling_factor_on_output=(
                backend.apply_routed_scaling_factor_on_output
            ),
        ),
    )
    routed_topk = self_routing_topk_from_bypassed(topk_output)

    out = backend.forward(
        layer,
        hidden_states,
        topk_output,
        num_global_tokens=num_tokens * ep_size,
        max_num_tokens_per_gpu=num_tokens,
    )
    torch.cuda.synchronize()

    expected = _dense_moe_reference(
        hidden_states,
        _dequantize_fp8_weight(fp8_w13, fp8_w13_scale, block_shape),
        _dequantize_fp8_weight(fp8_w2, fp8_w2_scale, block_shape),
        routed_topk.topk_ids,
        routed_topk.topk_weights,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


def test_fp8_triton_backend_ep_forward_handles_empty_rank() -> None:
    _require_cdna4_gpu()
    from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=16,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=2,
    )
    backend = Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
    )
    layer = SimpleNamespace(activation="silu")
    topk_output = SimpleNamespace(
        topk_ids=torch.empty((0, 2), dtype=torch.int32, device="cuda"),
        topk_weights=torch.empty((0, 2), dtype=torch.float32, device="cuda"),
    )
    hidden_states = torch.empty((0, 32), dtype=torch.bfloat16, device="cuda")

    out = backend.forward(
        layer,
        hidden_states,
        topk_output,
        num_global_tokens=0,
        max_num_tokens_per_gpu=0,
    )

    assert out.shape == (0, 32)
    assert out.dtype == torch.bfloat16


def test_fp8_triton_backend_zero_local_rank_participates_when_global_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    import tokenspeed_kernel

    from tokenspeed.runtime.layers import activation
    from tokenspeed.runtime.layers.moe.backends import (
        ep_combine,
        ep_dispatch,
        ep_experts,
        ep_reduce,
    )
    from tokenspeed.runtime.layers.moe.backends.fp8.triton import Fp8TritonBackend
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    hidden_size = 32
    intermediate_size = 16
    top_k = 2
    ep_size = 2
    num_local_experts = 2
    device = "cuda"
    calls: list[str] = []

    def fake_moe_dispatch(*args, **kwargs):
        del args, kwargs
        calls.append("metadata")
        return SimpleNamespace(
            owner_counts=torch.zeros((ep_size,), dtype=torch.int32, device=device),
            owner_expert_counts=torch.zeros(
                (ep_size, num_local_experts),
                dtype=torch.int32,
                device=device,
            ),
            owner_expert_offsets=torch.zeros(
                (ep_size, num_local_experts + 1),
                dtype=torch.int32,
                device=device,
            ),
            combine_offsets=torch.empty((ep_size, 0), dtype=torch.int32, device=device),
            dispatch_offsets=torch.empty((0, top_k), dtype=torch.int32, device=device),
        )

    def fake_prepare(metadata, workspace):
        del metadata
        calls.append("prepare")
        zero_counts = torch.zeros(
            (ep_size, num_local_experts),
            dtype=torch.int32,
            device=workspace.device,
        )
        zero_offsets = torch.zeros(
            (ep_size, num_local_experts + 1),
            dtype=torch.int32,
            device=workspace.device,
        )
        return ep_dispatch.EPOwnerDispatchPlan(
            owner_base_offsets=torch.zeros(
                (ep_size,),
                dtype=torch.int32,
                device=workspace.device,
            ),
            owner_expert_base_offsets=zero_counts,
            aggregate_owner_expert_counts=zero_counts,
            aggregate_owner_expert_offsets=zero_offsets,
            local_expert_counts=zero_counts[0],
            local_expert_offsets=zero_offsets[0],
        )

    def fake_dispatch(hidden_states, topk_ids, metadata, workspace, **kwargs):
        del hidden_states, topk_ids, metadata, kwargs
        calls.append("dispatch")
        return workspace.view(0)

    def fake_gemm(A, B, B_scale, counts, **kwargs):
        del A, B_scale, counts, kwargs
        calls.append("gemm")
        return torch.empty((0, B.shape[1]), dtype=torch.bfloat16, device=device)

    def fake_combine(
        owner_outputs,
        topk_ids,
        metadata,
        dispatch_plan,
        workspace,
        expert_owner,
        local_expert_id,
        **kwargs,
    ):
        del metadata, dispatch_plan, workspace, expert_owner, local_expert_id, kwargs
        calls.append("combine")
        return torch.empty(
            (0, topk_ids.shape[1], owner_outputs.shape[1]),
            dtype=owner_outputs.dtype,
            device=owner_outputs.device,
        )

    def fake_reduce(returned_slots, topk_weights):
        del topk_weights
        calls.append("reduce")
        return torch.empty(
            (0, hidden_size),
            dtype=returned_slots.dtype,
            device=returned_slots.device,
        )

    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", fake_moe_dispatch)
    monkeypatch.setattr(ep_dispatch, "prepare_owner_directed_dispatch", fake_prepare)
    monkeypatch.setattr(ep_dispatch, "owner_directed_dispatch", fake_dispatch)
    monkeypatch.setattr(ep_experts, "owner_rank_fp8_expert_gemm", fake_gemm)
    monkeypatch.setattr(activation, "silu_and_mul", lambda gate_up, out: None)
    monkeypatch.setattr(ep_combine, "owner_directed_combine", fake_combine)
    monkeypatch.setattr(ep_reduce, "ep_weighted_reduce", fake_reduce)

    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=ep_size * num_local_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=ep_size,
    )
    backend = Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
    )
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=torch.empty(
            (num_local_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        w13_weight_scale_inv=torch.empty(
            (num_local_experts, 2, 2),
            dtype=torch.float32,
            device=device,
        ),
        w2_weight=torch.empty(
            (num_local_experts, hidden_size, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        w2_weight_scale_inv=torch.empty(
            (num_local_experts, 2, 1),
            dtype=torch.float32,
            device=device,
        ),
    )
    hidden_states = torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device)
    topk_output = SimpleNamespace(
        topk_ids=torch.empty((0, top_k), dtype=torch.int32, device=device),
        topk_weights=torch.empty((0, top_k), dtype=torch.float32, device=device),
    )

    out = backend.forward(
        layer,
        hidden_states,
        topk_output,
        num_global_tokens=3,
        max_num_tokens_per_gpu=0,
    )

    assert out.shape == (0, hidden_size)
    assert backend._ep_workspace.max_tokens_per_rank == 2
    assert calls == ["metadata", "prepare", "dispatch", "gemm", "gemm", "combine", "reduce"]


def test_fp8_triton_backend_ep_forward_uses_pre_routed_fused_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for fused wiring validation")

    import tokenspeed_kernel

    from tokenspeed.runtime.layers import activation
    from tokenspeed.runtime.layers.moe.backends import (
        ep_dispatch,
        ep_experts,
        ep_fused_down_combine,
        ep_fused_gate_up,
        ep_fused_metadata,
    )
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Fp8Config

    device = "cpu"
    hidden_size = 32
    intermediate_size = 16
    top_k = 2
    ep_size = 2
    num_local_experts = 2
    calls: list[str] = []
    dispatch_plan = ep_dispatch.EPOwnerDispatchPlan(
        owner_base_offsets=torch.zeros((ep_size,), dtype=torch.int32, device=device),
        owner_expert_base_offsets=torch.zeros(
            (ep_size, num_local_experts),
            dtype=torch.int32,
            device=device,
        ),
        aggregate_owner_expert_counts=torch.zeros(
            (ep_size, num_local_experts),
            dtype=torch.int32,
            device=device,
        ),
        aggregate_owner_expert_offsets=torch.zeros(
            (ep_size, num_local_experts + 1),
            dtype=torch.int32,
            device=device,
        ),
        local_expert_counts=torch.tensor([2, 2], dtype=torch.int32, device=device),
        local_expert_offsets=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
    )
    fused_metadata = SimpleNamespace(tag="fused")
    expected = torch.full((2, hidden_size), 3.0, dtype=torch.bfloat16, device=device)

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )

    def fake_moe_dispatch(*args, **kwargs):
        del args, kwargs
        calls.append("metadata")
        return SimpleNamespace(
            owner_counts=torch.tensor([4, 0], dtype=torch.int32, device=device),
            owner_expert_counts=torch.tensor(
                [[2, 2], [0, 0]],
                dtype=torch.int32,
                device=device,
            ),
            owner_expert_offsets=torch.tensor(
                [[0, 2, 4], [0, 0, 0]],
                dtype=torch.int32,
                device=device,
            ),
            dispatch_offsets=torch.tensor(
                [[0, 2], [1, 3]],
                dtype=torch.int32,
                device=device,
            ),
            combine_offsets=torch.tensor(
                [[0, 2, 1, 3], [-1, -1, -1, -1]],
                dtype=torch.int32,
                device=device,
            ),
        )

    def fake_prepare(metadata, workspace):
        del metadata
        calls.append("prepare")
        assert workspace.backend == "torch"
        return dispatch_plan

    def fake_build(hidden_states, topk_ids, topk_weights, metadata, workspace, **kwargs):
        del hidden_states, topk_ids, topk_weights, metadata, workspace
        calls.append("fused_metadata")
        assert kwargs["dispatch_plan"] is dispatch_plan
        return fused_metadata

    def fake_gate_up(*args, **kwargs):
        del args, kwargs
        calls.append("fused_gate_up")
        return SimpleNamespace(
            gate_up=torch.ones(
                (4, 2 * intermediate_size),
                dtype=torch.bfloat16,
                device=device,
            ),
            dispatch_plan=dispatch_plan,
        )

    def fake_silu_and_mul(gate_up, out):
        del gate_up
        calls.append("activation")
        out.fill_(1)

    def fake_down(owner_intermediate, *args, **kwargs):
        del args, kwargs
        calls.append("fused_down")
        assert owner_intermediate.shape == (4, intermediate_size)
        return SimpleNamespace(output=expected, dispatch_plan=dispatch_plan)

    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", fake_moe_dispatch)
    monkeypatch.setattr(ep_dispatch, "prepare_owner_directed_dispatch", fake_prepare)
    monkeypatch.setattr(
        ep_dispatch,
        "owner_directed_dispatch",
        lambda *args, **kwargs: pytest.fail("unfused dispatch was called"),
    )
    monkeypatch.setattr(
        ep_experts,
        "owner_rank_fp8_expert_gemm",
        lambda *args, **kwargs: pytest.fail("unfused expert GEMM was called"),
    )
    monkeypatch.setattr(
        ep_fused_metadata,
        "build_pre_routed_fused_ep_metadata",
        fake_build,
    )
    monkeypatch.setattr(
        ep_fused_gate_up,
        "pre_routed_fused_dispatch_gate_up",
        fake_gate_up,
    )
    monkeypatch.setattr(activation, "silu_and_mul", fake_silu_and_mul)
    monkeypatch.setattr(
        ep_fused_down_combine,
        "pre_routed_fused_down_combine",
        fake_down,
    )

    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=ep_size * num_local_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=ep_size,
    )
    backend = fp8_triton.Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
    )
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=torch.empty(
            (num_local_experts, 2 * intermediate_size, hidden_size),
            dtype=fp8_dtype,
            device=device,
        ),
        w13_weight_scale_inv=torch.empty(
            (num_local_experts, 2, 2),
            dtype=torch.float32,
            device=device,
        ),
        w2_weight=torch.empty(
            (num_local_experts, hidden_size, intermediate_size),
            dtype=fp8_dtype,
            device=device,
        ),
        w2_weight_scale_inv=torch.empty(
            (num_local_experts, 2, 1),
            dtype=torch.float32,
            device=device,
        ),
    )
    hidden_states = torch.empty((2, hidden_size), dtype=torch.bfloat16, device=device)
    topk_output = SimpleNamespace(
        topk_ids=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device),
        topk_weights=torch.ones((2, top_k), dtype=torch.float32, device=device),
    )

    out = backend.forward(
        layer,
        hidden_states,
        topk_output,
        num_global_tokens=4,
        max_num_tokens_per_gpu=2,
    )

    assert out is expected
    assert calls == [
        "metadata",
        "prepare",
        "fused_metadata",
        "fused_gate_up",
        "activation",
        "fused_down",
    ]


def test_fp8_triton_backend_ep_forward_uses_self_routing_fused_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch FP8 dtype is required for fused wiring validation")

    import tokenspeed_kernel

    from tokenspeed.runtime.layers import activation
    from tokenspeed.runtime.layers.moe.backends import (
        ep_dispatch,
        ep_experts,
        ep_fused_down_combine,
        ep_fused_gate_up,
        ep_fused_metadata,
    )
    from tokenspeed.runtime.layers.moe.backends.fp8 import triton as fp8_triton
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig
    from tokenspeed.runtime.layers.quantization import Fp8Config

    device = "cpu"
    hidden_size = 32
    intermediate_size = 16
    top_k = 2
    ep_size = 2
    num_local_experts = 2
    calls: list[str] = []
    expected = torch.full((2, hidden_size), 5.0, dtype=torch.bfloat16, device=device)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        fp8_triton,
        "_current_platform",
        lambda: SimpleNamespace(is_amd=True, is_cdna4_plus=True),
    )

    def fail_pre_routed(*args, **kwargs):
        del args, kwargs
        pytest.fail("pre-routed fused helper was called")

    def fail_unfused(*args, **kwargs):
        del args, kwargs
        pytest.fail("unfused EP helper was called")

    def fake_self_gate_up(
        hidden_states_arg,
        topk_output_arg,
        expert_owner,
        local_expert_id,
        workspace,
        local_gate_up_weight,
        local_gate_up_weight_scale,
        **kwargs,
    ):
        del local_gate_up_weight, local_gate_up_weight_scale
        calls.append("self_fused_gate_up")
        assert hidden_states_arg is hidden_states
        assert topk_output_arg.format.is_bypassed()
        assert topk_output_arg.router_logits is router_logits
        assert topk_output_arg.topk_config.top_k == top_k
        assert workspace.backend == "torch"
        assert expert_owner.tolist() == [0, 0, 1, 1]
        assert local_expert_id.tolist() == [0, 1, 0, 1]
        assert kwargs["expected_metadata_kernel_name"] == "gluon_ep_metadata_gfx950"
        assert kwargs["expected_gemm_kernel_name"] == "gluon_fp8_local_experts_gfx950"
        result = SimpleNamespace(
            topk_output=SimpleNamespace(
                topk_ids=torch.tensor(
                    [[0, 1], [2, 3]],
                    dtype=torch.int32,
                    device=device,
                ),
                topk_weights=torch.ones((2, top_k), dtype=torch.float32, device=device),
            ),
            ep_metadata=SimpleNamespace(tag="metadata"),
            fused_metadata=SimpleNamespace(tag="fused"),
            gate_up=torch.ones(
                (4, 2 * intermediate_size),
                dtype=torch.bfloat16,
                device=device,
            ),
            dispatch_plan=SimpleNamespace(tag="dispatch_plan"),
        )
        captured["gate_up_result"] = result
        return result

    def fake_silu_and_mul(gate_up, out):
        del gate_up
        calls.append("activation")
        out.fill_(2)

    def fake_self_down(
        owner_intermediate,
        gate_up_result,
        workspace,
        local_down_weight,
        local_down_weight_scale,
        expert_owner,
        local_expert_id,
        **kwargs,
    ):
        del local_down_weight, local_down_weight_scale, expert_owner, local_expert_id
        calls.append("self_fused_down")
        assert gate_up_result is captured["gate_up_result"]
        assert owner_intermediate.shape == (4, intermediate_size)
        assert workspace.backend == "torch"
        assert kwargs["expected_gemm_kernel_name"] == "gluon_fp8_local_experts_gfx950"
        assert kwargs["expected_reduce_kernel_name"] == "gluon_local_sum_reduce_gfx950"
        return SimpleNamespace(output=expected)

    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", fail_pre_routed)
    monkeypatch.setattr(ep_dispatch, "prepare_owner_directed_dispatch", fail_pre_routed)
    monkeypatch.setattr(ep_dispatch, "owner_directed_dispatch", fail_unfused)
    monkeypatch.setattr(ep_experts, "owner_rank_fp8_expert_gemm", fail_unfused)
    monkeypatch.setattr(
        ep_fused_metadata,
        "build_pre_routed_fused_ep_metadata",
        fail_pre_routed,
    )
    monkeypatch.setattr(
        ep_fused_gate_up,
        "pre_routed_fused_dispatch_gate_up",
        fail_pre_routed,
    )
    monkeypatch.setattr(
        ep_fused_gate_up,
        "self_routing_fused_dispatch_gate_up",
        fake_self_gate_up,
    )
    monkeypatch.setattr(activation, "silu_and_mul", fake_silu_and_mul)
    monkeypatch.setattr(
        ep_fused_down_combine,
        "pre_routed_fused_down_combine",
        fail_pre_routed,
    )
    monkeypatch.setattr(
        ep_fused_down_combine,
        "self_routing_fused_down_combine",
        fake_self_down,
    )

    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=ep_size * num_local_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=ep_size,
    )
    backend = fp8_triton.Fp8TritonBackend(
        key=BackendKey(arch="gfx950", quant="fp8", impl="triton"),
        spec=spec,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[16, 16],
        ),
        routing_config={"moe_fused_features": {"self_routing"}},
    )
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=torch.empty(
            (num_local_experts, 2 * intermediate_size, hidden_size),
            dtype=fp8_dtype,
            device=device,
        ),
        w13_weight_scale_inv=torch.empty(
            (num_local_experts, 2, 2),
            dtype=torch.float32,
            device=device,
        ),
        w2_weight=torch.empty(
            (num_local_experts, hidden_size, intermediate_size),
            dtype=fp8_dtype,
            device=device,
        ),
        w2_weight_scale_inv=torch.empty(
            (num_local_experts, 2, 1),
            dtype=torch.float32,
            device=device,
        ),
    )
    hidden_states = torch.empty((2, hidden_size), dtype=torch.bfloat16, device=device)
    router_logits = torch.empty(
        (2, ep_size * num_local_experts),
        dtype=torch.float32,
        device=device,
    )
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=top_k,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=2,
            routed_scaling_factor=0.5,
            apply_routed_scaling_factor_on_output=True,
        ),
    )

    out = backend.forward(
        layer,
        hidden_states,
        topk_output,
        num_global_tokens=4,
        max_num_tokens_per_gpu=2,
    )

    assert out is expected
    assert backend.topk_output_format.is_bypassed()
    assert calls == ["self_fused_gate_up", "activation", "self_fused_down"]


def _dense_moe_reference(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros(
        (hidden_states.shape[0], hidden_states.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for token in range(topk_ids.shape[0]):
        for topk_idx in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, topk_idx].item())
            gate_up = (hidden_states[token].float() @ w13[expert].float().T).to(
                torch.bfloat16
            )
            intermediate = (
                F.silu(gate_up[: gate_up.numel() // 2]) * gate_up[gate_up.numel() // 2 :]
            ).to(torch.bfloat16)
            slot = intermediate.float() @ w2[expert].float().T
            out[token] += slot * topk_weights[token, topk_idx].float()
    return out


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
