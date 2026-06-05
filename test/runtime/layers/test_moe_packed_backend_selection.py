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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _require_cdna4_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for packed backend wiring validation")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for packed backend wiring validation")


def _moe_spec(*, ep_size: int = 1, intermediate_size: int = 64):
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    return MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=4 // ep_size,
        hidden_size=128,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=ep_size,
    )


def _set_auto_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.utils import MoeBackend

    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)


def test_mxfp4_selects_amd_packed_backend_without_fused_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cdna4_gpu()
    _set_auto_backend(monkeypatch)
    from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import (
        Mxfp4TritonKernelBackend,
    )
    from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
        MXFP4_E2M1_BLOCK32_FORMAT,
    )
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    backend = select_backend(
        _moe_spec(),
        Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )

    assert isinstance(backend, Mxfp4TritonKernelBackend)
    assert backend.key.quant == "mxfp4"
    assert backend.key.impl == "triton_kernel"
    assert backend.expert_weight_format_signature is MXFP4_E2M1_BLOCK32_FORMAT


@pytest.mark.parametrize("feature", ["self_routing", "pre_routed"])
def test_mxfp4_fused_request_rejects_non_fused_amd_backend(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
) -> None:
    _require_cdna4_gpu()
    _set_auto_backend(monkeypatch)
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    with pytest.raises(
        RuntimeError,
        match=rf"triton_kernel:unsupported-packed-fused\({feature}\)",
    ):
        select_backend(
            _moe_spec(),
            Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
            routing_config={"moe_fused_features": {feature}},
        )


def _patch_fake_blackwell_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenspeed.runtime.layers.moe.core import selector as selector_module

    monkeypatch.setattr(
        selector_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=False),
    )
    monkeypatch.setattr(selector_module, "_detect_arch", lambda: "sm100")


def _patch_nvfp4_backends_as_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from tokenspeed.runtime.layers.moe.backends.nvfp4 import (
            flashinfer_cutedsl,
            flashinfer_cutlass,
            flashinfer_trtllm,
        )
    except ImportError as exc:
        pytest.skip(f"NVFP4 FlashInfer backend modules are unavailable: {exc}")

    fake_platform = SimpleNamespace(is_nvidia=True)
    monkeypatch.setattr(
        flashinfer_trtllm,
        "current_platform",
        lambda: fake_platform,
    )
    monkeypatch.setattr(
        flashinfer_cutedsl,
        "current_platform",
        lambda: fake_platform,
    )
    monkeypatch.setattr(
        flashinfer_cutlass,
        "current_platform",
        lambda: fake_platform,
    )


def test_nvfp4_pre_routed_fused_request_skips_self_routing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_auto_backend(monkeypatch)
    _patch_fake_blackwell_selector(monkeypatch)
    _patch_nvfp4_backends_as_available(monkeypatch)
    from tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_cutedsl import (
        Nvfp4FlashinferCuteDslBackend,
    )
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.quantization import Nvfp4Config

    backend = select_backend(
        _moe_spec(),
        Nvfp4Config(group_size=16),
        routing_config={"moe_fused_features": {"pre_routed"}},
    )

    assert isinstance(backend, Nvfp4FlashinferCuteDslBackend)
    assert backend.key.impl == "flashinfer_cutedsl"


def test_nvfp4_self_routing_fused_request_skips_pre_routed_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_auto_backend(monkeypatch)
    _patch_fake_blackwell_selector(monkeypatch)
    _patch_nvfp4_backends_as_available(monkeypatch)
    from tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_trtllm import (
        Nvfp4FlashinferTrtllmBackend,
    )
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.quantization import Nvfp4Config

    backend = select_backend(
        _moe_spec(),
        Nvfp4Config(group_size=16),
        routing_config={"features": "self_routing"},
    )

    assert isinstance(backend, Nvfp4FlashinferTrtllmBackend)
    assert backend.key.impl == "flashinfer_trtllm"
