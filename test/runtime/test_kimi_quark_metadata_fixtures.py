"""No-weight Kimi Quark metadata fixture tests."""

from __future__ import annotations

import json

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_w4a8_model_config,
    quark_kimi_w4a8_quantization_config,
    quark_kimi_w8a8_model_config,
    quark_kimi_w8a8_quantization_config,
)


def _roundtrip_json(config: dict) -> dict:
    return json.loads(json.dumps(config))


def test_w8a8_quark_fixture_pins_fp8_per_channel_weight_metadata() -> None:
    quant_config = quark_kimi_w8a8_quantization_config()

    assert quant_config["quant_method"] == "quark"
    global_config = quant_config["global_quant_config"]
    weight = global_config["weight"]
    input_tensors = global_config["input_tensors"]

    assert weight["dtype"] == "fp8_e4m3"
    assert weight["is_dynamic"] is False
    assert weight["qscheme"] == "per_channel"
    assert weight["ch_axis"] == 0
    assert input_tensors["dtype"] == "fp8_e4m3"
    assert input_tensors["is_dynamic"] is True
    assert input_tensors["qscheme"] == "per_tensor"
    assert input_tensors["ch_axis"] is None


def test_w4a8_quark_fixture_pins_fp8_int4_staged_weight_metadata() -> None:
    quant_config = quark_kimi_w4a8_quantization_config()

    assert quant_config["quant_method"] == "quark"
    global_config = quant_config["global_quant_config"]
    weights = global_config["weight"]
    input_tensors = global_config["input_tensors"]
    export = quant_config["export"]

    assert isinstance(weights, list)
    assert len(weights) == 2
    assert weights[0]["dtype"] == "fp8_e4m3"
    assert weights[0]["is_dynamic"] is False
    assert weights[1]["dtype"] == "int4"
    assert weights[1]["is_dynamic"] is False
    assert weights[1]["qscheme"] == "per_channel"
    assert weights[1]["ch_axis"] == 0
    assert input_tensors["dtype"] == "fp8_e4m3"
    assert input_tensors["is_dynamic"] is True
    assert input_tensors["qscheme"] == "per_tensor"
    assert export["pack_method"] == "reorder"
    assert export["weight_format"] == "real_quantized"


def test_quark_kimi_model_fixtures_are_json_only_detection_inputs() -> None:
    for model_config in (
        quark_kimi_w8a8_model_config(),
        quark_kimi_w4a8_model_config(),
    ):
        roundtripped = _roundtrip_json(model_config)

        assert roundtripped["model_type"] == "kimi_k25"
        assert "quantization_config" in roundtripped
        assert roundtripped["quantization_config"]["quant_method"] == "quark"
        assert "global_quant_config" in roundtripped["quantization_config"]
        assert "input_tensors" in roundtripped["quantization_config"][
            "global_quant_config"
        ]
        assert "weight" in roundtripped["quantization_config"]["global_quant_config"]


def test_quark_fixture_helpers_return_independent_mutable_dicts() -> None:
    first = quark_kimi_w8a8_quantization_config()
    second = quark_kimi_w8a8_quantization_config()

    first["global_quant_config"]["weight"]["dtype"] = "corrupted"

    assert second["global_quant_config"]["weight"]["dtype"] == "fp8_e4m3"
