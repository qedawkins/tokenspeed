"""CPU-only Kimi MXFP4 runtime wiring checks."""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec


class _FakeMxfp4Config:
    def __init__(
        self,
        *,
        ignored_layers: list[str] | None = None,
        is_checkpoint_mxfp4_serialized: bool = True,
        is_w4a8_fp8: bool = False,
    ) -> None:
        self.ignored_layers = ignored_layers or []
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.is_w4a8_fp8 = is_w4a8_fp8


class _FakeFp8Config:
    weight_block_size = None


class _FakeCompressedTensorsConfig:
    target_scheme_map = {"Linear": {}}

    def _is_wNa16_group_channel(self, *_args: object) -> bool:
        return False


class _FakeQuantizationConfig:
    pass


class _FakeNvfp4Config:
    pass


class _FakeW4A8QuarkConfig:
    pass


class _FakeW8A8Fp8Config:
    pass


class _StubObject:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


class _StubEnvValue:
    def __init__(self, value: object = False) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class _FakeLayout:
    @staticmethod
    def make_default_matmul_mxfp4_w_layout(**_kwargs: object) -> object:
        return object()

    @staticmethod
    def make_default_matmul_mxfp4_w_scale_layout(**_kwargs: object) -> object:
        return object()


class _FakeOptFlags:
    @staticmethod
    def update_opt_flags_constraints(_constraints: dict[str, object]) -> None:
        return None


def _install_cpu_safe_kernel_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name.startswith("tokenspeed_kernel"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (
        "tokenspeed.runtime.layers.quantization",
        "tokenspeed.runtime.layers.quantization.utils",
        "tokenspeed.runtime.layers.moe.core.selector",
        "tokenspeed.runtime.layers.moe.layer",
        "tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel",
        "tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel_ep",
        "tokenspeed.runtime.layers.moe.topk",
        "tokenspeed.runtime.moe.distribution_recorder",
        "tokenspeed.runtime.utils.env",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    fake_platform = SimpleNamespace(
        arch_version=SimpleNamespace(major=9, minor=50),
        is_amd=True,
        is_nvidia=False,
        is_blackwell=False,
        is_hopper=False,
        is_cdna4=True,
        is_cdna4_plus=True,
        fp8e4m3fn=SimpleNamespace(dtype=torch.uint8),
    )

    kernel = ModuleType("tokenspeed_kernel")
    kernel.moe_route = _not_called
    kernel.moe_experts = _not_called
    kernel.quantize_fp8 = _not_called
    kernel.quantize_mxfp4 = _not_called
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel", kernel)

    triton_redirect = ModuleType("tokenspeed_kernel._triton")
    triton_redirect.redirect_triton_to_tokenspeed_triton = nullcontext
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel._triton", triton_redirect)

    platform = ModuleType("tokenspeed_kernel.platform")
    platform.current_platform = lambda: fake_platform
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.platform", platform)

    triton_kernels = ModuleType("tokenspeed_kernel.ops.moe.triton_kernels")
    triton_kernels.FP4 = object()
    triton_kernels.FlexCtx = _StubObject
    triton_kernels.FnSpecs = _StubObject
    triton_kernels.FusedActivation = _StubObject
    triton_kernels.InFlexData = _StubObject
    triton_kernels.PrecisionConfig = _StubObject
    triton_kernels.convert_layout = lambda tensor, _layout: tensor
    triton_kernels.layout = _FakeLayout()
    triton_kernels.make_ragged_tensor_metadata = lambda counts, rows: (counts, rows)
    triton_kernels.opt_flags = _FakeOptFlags()
    triton_kernels.swiglu_fn = object()
    triton_kernels.wrap_torch_tensor = lambda tensor, dtype=None: tensor

    ops = ModuleType("tokenspeed_kernel.ops")
    moe_ops = ModuleType("tokenspeed_kernel.ops.moe")
    moe_ops.ExpertLocationDispatchInfo = _StubObject
    moe_ops.topk_ids_logical_to_physical = _not_called
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.ops", ops)
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.ops.moe", moe_ops)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.moe.triton_kernels",
        triton_kernels,
    )
    communication_ops = ModuleType("tokenspeed_kernel.ops.communication")
    trtllm_comm = ModuleType("tokenspeed_kernel.ops.communication.trtllm")
    trtllm_comm.allgather_dual_rmsnorm = _not_called
    trtllm_comm.allreduce_residual_rmsnorm = _not_called
    trtllm_comm.reducescatter_residual_rmsnorm = _not_called
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.communication",
        communication_ops,
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.communication.trtllm",
        trtllm_comm,
    )

    numerics = ModuleType("tokenspeed_kernel.numerics")
    reference = ModuleType("tokenspeed_kernel.numerics.reference")
    reference_moe = ModuleType("tokenspeed_kernel.numerics.reference.moe")
    reference_moe._mask_topk_ids_padded_region = _not_called
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.numerics", numerics)
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.numerics.reference", reference)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.numerics.reference.moe",
        reference_moe,
    )

    quantization = ModuleType("tokenspeed.runtime.layers.quantization")
    quantization.CompressedTensorsConfig = _FakeCompressedTensorsConfig
    quantization.Fp8Config = _FakeFp8Config
    quantization.Mxfp4Config = _FakeMxfp4Config
    quantization.Nvfp4Config = _FakeNvfp4Config
    quantization.QuantizationConfig = _FakeQuantizationConfig
    quantization.W4A8QuarkConfig = _FakeW4A8QuarkConfig
    quantization.W8A8Fp8Config = _FakeW8A8Fp8Config
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.layers.quantization",
        quantization,
    )

    quantization_utils = ModuleType("tokenspeed.runtime.layers.quantization.utils")
    quantization_utils.should_ignore_quant_layer = _should_ignore_quant_layer
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.layers.quantization.utils",
        quantization_utils,
    )

    distribution_recorder = ModuleType("tokenspeed.runtime.moe.distribution_recorder")
    distribution_recorder.get_global_expert_distribution_recorder = lambda: None
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.moe.distribution_recorder",
        distribution_recorder,
    )

    env = ModuleType("tokenspeed.runtime.utils.env")
    env.global_server_args_dict = {"ep_num_redundant_experts": 0}
    env.envs = SimpleNamespace(
        TOKENSPEED_MOE_PADDING=_StubEnvValue(False),
        TOKENSPEED_NVTX=_StubEnvValue(False),
    )
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils.env", env)


def _not_called(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("stubbed kernel function should not be called")


def _should_ignore_quant_layer(prefix: str, ignored_layers: list[str]) -> bool:
    for pattern in ignored_layers:
        if pattern.startswith("re:") and re.match(pattern[3:], prefix):
            return True
        if pattern == prefix:
            return True
    return False


def _tp4_spec(*, prefix: str = "language_model.model.layers.1.mlp.experts"):
    return MoELayerSpec(
        top_k=2,
        num_experts=8,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        activation="silu",
        tp_rank=2,
        tp_size=4,
        ep_rank=0,
        ep_size=1,
        prefix=prefix,
    )


def _tp4_ep4_spec(*, prefix: str = "language_model.model.layers.1.mlp.experts"):
    return MoELayerSpec(
        top_k=2,
        num_experts=16,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=128,
        activation="silu",
        tp_rank=2,
        tp_size=4,
        ep_rank=2,
        ep_size=4,
        prefix=prefix,
    )


def test_kimi_mxfp4_local_tp_selects_packed_moe_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_safe_kernel_surface(monkeypatch)
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends import _REGISTERED
    from tokenspeed.runtime.layers.moe.core import registry
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.moe.utils import MoeBackend

    _REGISTERED.clear()
    registry._REGISTRY.clear()
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)

    quant_config = _FakeMxfp4Config(is_checkpoint_mxfp4_serialized=True)
    backend = select_backend(_tp4_spec(), quant_config)

    assert type(backend).__name__ == "Mxfp4TritonKernelBackend"
    assert backend.key == BackendKey(
        arch="gfx950",
        quant="mxfp4",
        impl="triton_kernel",
    )
    assert backend.topk_output_format.is_bypassed()
    assert backend.expert_weight_format_signature.name == "mxfp4_e2m1_block32"
    assert type(backend).supports(_tp4_spec(), quant_config) is True
    assert type(backend).supports(
        MoELayerSpec(
            **{
                **_tp4_spec().__dict__,
                "num_local_experts": 4,
                "ep_size": 2,
            }
        ),
        quant_config,
    ) is False

    layer = nn.Module()
    layer.activation = "silu"
    backend.create_layer_weights(layer, with_bias=True)

    assert tuple(layer.w13_weight.shape) == (8, 64, 64)
    assert tuple(layer.w13_weight_scale.shape) == (8, 64, 4)
    assert tuple(layer.w2_weight.shape) == (8, 128, 16)
    assert tuple(layer.w2_weight_scale.shape) == (8, 128, 1)
    assert tuple(layer.w13_weight_bias.shape) == (8, 64)
    assert tuple(layer.w2_weight_bias.shape) == (8, 128)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w2_weight.dtype == torch.uint8
    assert hasattr(layer.w13_weight, "weight_loader")
    assert hasattr(layer.w2_weight_scale, "weight_loader")


def test_kimi_mxfp4_local_tp_process_weights_allows_missing_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_safe_kernel_surface(monkeypatch)
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends import _REGISTERED
    from tokenspeed.runtime.layers.moe.core import registry
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.moe.utils import MoeBackend

    _REGISTERED.clear()
    registry._REGISTRY.clear()
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)

    backend = select_backend(
        _tp4_spec(),
        _FakeMxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )
    layer = nn.Module()
    layer.activation = "silu"
    backend.create_layer_weights(layer, with_bias=False)

    assert not hasattr(layer, "w13_weight_bias")
    assert not hasattr(layer, "w2_weight_bias")

    backend.process_weights_after_loading(layer)

    assert not hasattr(layer, "w13_weight_bias")
    assert not hasattr(layer, "w2_weight_bias")
    assert hasattr(layer, "w13_weight_triton_tensor")
    assert hasattr(layer, "w2_weight_triton_tensor")


def test_kimi_mxfp4_tp_ep_selects_ep_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_safe_kernel_surface(monkeypatch)
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends import _REGISTERED
    from tokenspeed.runtime.layers.moe.core import registry
    from tokenspeed.runtime.layers.moe.core.selector import select_backend
    from tokenspeed.runtime.layers.moe.utils import MoeBackend

    _REGISTERED.clear()
    registry._REGISTRY.clear()
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)

    quant_config = _FakeMxfp4Config(is_checkpoint_mxfp4_serialized=True)
    backend = select_backend(_tp4_ep4_spec(), quant_config)

    assert type(backend).__name__ == "Mxfp4TritonKernelEPBackend"
    assert backend.key == BackendKey(
        arch="gfx950",
        quant="mxfp4",
        impl="triton_kernel_ep",
    )
    assert backend.topk_output_format.is_bypassed()
    assert backend.expert_weight_format_signature.name == "mxfp4_e2m1_block32"

    layer = nn.Module()
    layer.activation = "silu"
    backend.create_layer_weights(layer, with_bias=True)
    backend.process_weights_after_loading(layer)

    assert tuple(layer.w13_weight.shape) == (4, 64, 64)
    assert tuple(layer.w13_weight_scale.shape) == (4, 64, 4)
    assert tuple(layer.w2_weight.shape) == (4, 128, 16)
    assert tuple(layer.w2_weight_scale.shape) == (4, 128, 1)
    assert not hasattr(layer, "w13_weight_triton_tensor")
    assert not hasattr(layer, "w2_weight_triton_tensor")


def test_kimi_mxfp4_moelayer_allows_tp_ep_only_for_checkpoint_mxfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_safe_kernel_surface(monkeypatch)
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.backends import _REGISTERED
    from tokenspeed.runtime.layers.moe.core import registry
    from tokenspeed.runtime.layers.moe.layer import MoELayer
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.utils.env import global_server_args_dict

    _REGISTERED.clear()
    registry._REGISTRY.clear()
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", MoeBackend.AUTO)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)

    layer = MoELayer(
        top_k=2,
        num_experts=16,
        hidden_size=128,
        intermediate_size=128,
        quant_config=_FakeMxfp4Config(is_checkpoint_mxfp4_serialized=True),
        layer_index=1,
        prefix="language_model.model.layers.1.mlp.experts",
        tp_rank=2,
        tp_size=4,
        ep_rank=2,
        ep_size=4,
        activation="silu",
        with_bias=True,
    )

    assert type(layer.backend).__name__ == "Mxfp4TritonKernelEPBackend"
    assert layer.backend.key.impl == "triton_kernel_ep"
    assert layer.topk_output_format.is_bypassed()

    with pytest.raises(ValueError, match="Mixed TP and EP"):
        MoELayer(
            top_k=2,
            num_experts=16,
            hidden_size=128,
            intermediate_size=128,
            quant_config=None,
            layer_index=1,
            prefix="language_model.model.layers.1.mlp.experts",
            tp_rank=0,
            tp_size=4,
            ep_rank=0,
            ep_size=4,
        )


def test_kimi_mxfp4_ignored_prefixes_remain_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_safe_kernel_surface(monkeypatch)
    from tokenspeed.runtime.layers.moe.core import selector

    quant_config = _FakeMxfp4Config(
        ignored_layers=[
            "re:.*self_attn.*",
            "re:.*mlp.gate",
            "re:.*lm_head",
        ],
    )

    assert (
        selector._normalize_quant_kind(
            quant_config,
            prefix="language_model.model.layers.1.self_attn.q_proj",
        )
        == "unquantized"
    )
    assert (
        selector._normalize_quant_kind(
            quant_config,
            prefix="language_model.model.layers.0.mlp.gate_up_proj",
        )
        == "unquantized"
    )
    assert (
        selector._normalize_quant_kind(
            quant_config,
            prefix="language_model.lm_head",
        )
        == "unquantized"
    )
    assert (
        selector._normalize_quant_kind(
            quant_config,
            prefix="language_model.model.layers.1.mlp.shared_experts.gate_up_proj",
        )
        == "mxfp4"
    )
    assert (
        selector._normalize_quant_kind(
            quant_config,
            prefix="language_model.model.layers.1.mlp.experts",
        )
        == "mxfp4"
    )
