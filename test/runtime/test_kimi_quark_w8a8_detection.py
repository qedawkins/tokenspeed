"""Quark W8A8 quantization detection tests."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_w4a8_quantization_config,
    quark_kimi_w8a8_quantization_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TokenSpeed runtime config imports require a CUDA/ROCm platform",
)


def _verify_model_quantization(quant_config: dict, user_quant: str | None = None) -> str:
    from tokenspeed.runtime.configs.model_config import ModelConfig

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(quantization_config=quant_config)
    model_config.hf_generation_config = None
    model_config.model_path = "local-no-network"
    model_config.revision = None
    model_config.quantization = user_quant

    model_config._verify_quantization()

    return model_config.quantization


def test_quark_w8a8_from_config_marks_checkpoint_serialized() -> None:
    from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config

    config = W8A8Fp8Config.from_config(quark_kimi_w8a8_quantization_config())

    assert config.is_checkpoint_fp8_serialized is True


def test_model_config_accepts_quark_w8a8_without_user_override() -> None:
    quantization = _verify_model_quantization(quark_kimi_w8a8_quantization_config())

    assert quantization == "w8a8_fp8"


def test_model_config_accepts_explicit_w8a8_for_quark_w8a8() -> None:
    quantization = _verify_model_quantization(
        quark_kimi_w8a8_quantization_config(),
        user_quant="w8a8_fp8",
    )

    assert quantization == "w8a8_fp8"


def test_existing_w8a8_and_compressed_tensor_detection_is_unchanged() -> None:
    from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config

    explicit = W8A8Fp8Config.from_config({"quant_method": "w8a8_fp8"})
    compressed = W8A8Fp8Config.from_config({"quant_method": "compressed-tensors"})

    assert explicit.is_checkpoint_fp8_serialized is True
    assert compressed.is_checkpoint_fp8_serialized is True


def test_unsupported_quark_schema_does_not_select_w8a8() -> None:
    from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config

    config = deepcopy(quark_kimi_w8a8_quantization_config())
    config["global_quant_config"]["weight"]["qscheme"] = "per_tensor"

    quant_config = W8A8Fp8Config.from_config(config)
    assert quant_config.is_checkpoint_fp8_serialized is False

    with pytest.raises(ValueError, match="Unknown quantization method: quark"):
        _verify_model_quantization(config)


def test_quark_w4a8_schema_is_not_silently_treated_as_w8a8() -> None:
    from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config

    config = quark_kimi_w4a8_quantization_config()

    quant_config = W8A8Fp8Config.from_config(config)
    assert quant_config.is_checkpoint_fp8_serialized is False

    with pytest.raises(
        ValueError,
        match="Quantization method specified in the model config",
    ):
        _verify_model_quantization(config, user_quant="w8a8_fp8")
