"""No-weight Kimi Quark CLI and local metadata preflight tests."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_mxfp4_model_config,
    quark_kimi_mxfp4_safetensors_index,
    quark_kimi_w4a8_model_config,
    quark_kimi_w8a8_model_config,
)
from tokenspeed.runtime.kimi_quantization_preflight import (
    preflight_kimi_quantization,
    summarize_kimi_mxfp4_artifact,
    summarize_kimi_mxfp4_metadata,
)


_LOCAL_KIMI_MXFP4_MODEL = Path(
    "/data/models/hf/hub/models--amd--Kimi-K2.5-MXFP4/"
    "snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a"
)


def _parse_preflight_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization")
    return parser.parse_args(argv)


def _server_args_quantization_choices() -> set[str]:
    server_args_path = (
        Path(__file__).parents[2]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "utils"
        / "server_args.py"
    )
    tree = ast.parse(server_args_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--quantization"
        ):
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, ast.List):
                    return {
                        choice.value
                        for choice in keyword.value.elts
                        if isinstance(choice, ast.Constant)
                    }
    raise AssertionError("--quantization CLI choices not found in ServerArgs")


def _write_config(tmp_path: Path, model_config: dict, dirname: str) -> Path:
    model_dir = tmp_path / dirname
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(model_config))
    return model_dir


def test_cli_quantization_choices_include_quark_routes() -> None:
    choices = _server_args_quantization_choices()

    assert "nvfp4" in choices
    assert "mxfp4" in choices
    assert "w8a8_fp8" in choices
    assert "w4a8_quark" in choices


def test_w8a8_hf_model_id_cli_preflights_without_local_json() -> None:
    args = _parse_preflight_args(
        [
            "--model",
            "amd/Kimi-K2.5-Quark-W8A8",
            "--quantization",
            "w8a8_fp8",
        ]
    )

    result = preflight_kimi_quantization(
        args.model,
        args.quantization,
    )

    assert result.quantization == "w8a8_fp8"
    assert result.source == "explicit-cli"
    assert result.requires_local_artifact is True
    assert result.is_quark is True


def test_w4a8_hf_model_id_cli_preflights_without_local_json() -> None:
    args = _parse_preflight_args(
        [
            "--model",
            "amd/Kimi-K2.5-Quark-W4A8",
            "--quantization",
            "w4a8_quark",
        ]
    )

    result = preflight_kimi_quantization(
        args.model,
        args.quantization,
    )

    assert result.quantization == "w4a8_quark"
    assert result.source == "explicit-cli"
    assert result.requires_local_artifact is True
    assert result.is_quark is True


def test_local_w8a8_config_resolves_from_json(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_w8a8_model_config(),
        "kimi-w8a8",
    )

    result = preflight_kimi_quantization(str(model_dir))

    assert result.quantization == "w8a8_fp8"
    assert result.source == "local-json"
    assert result.config_path == str(model_dir / "config.json")
    assert result.requires_local_artifact is False
    assert result.is_quark is True


def test_local_w4a8_config_resolves_from_json(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_w4a8_model_config(),
        "kimi-w4a8",
    )

    result = preflight_kimi_quantization(str(model_dir))

    assert result.quantization == "w4a8_quark"
    assert result.source == "local-json"
    assert result.config_path == str(model_dir / "config.json")
    assert result.requires_local_artifact is False
    assert result.is_quark is True


def test_local_mxfp4_config_resolves_from_json(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_mxfp4_model_config(),
        "kimi-mxfp4",
    )

    result = preflight_kimi_quantization(str(model_dir))

    assert result.quantization == "mxfp4"
    assert result.source == "local-json"
    assert result.config_path == str(model_dir / "config.json")
    assert result.requires_local_artifact is False
    assert result.is_quark is True


def test_mxfp4_fixture_records_dynamic_fp4_artifact_metadata() -> None:
    summary = summarize_kimi_mxfp4_metadata(
        quark_kimi_mxfp4_model_config(),
        quark_kimi_mxfp4_safetensors_index(),
    )

    assert summary.is_quark_mxfp4_dynamic_fp4 is True
    assert summary.num_quantization_excludes == 5
    assert summary.excludes_attention is True
    assert summary.excludes_lm_head is True
    assert summary.excludes_mlp_gate is True
    assert summary.indexed_total_size == 558995180568
    assert summary.num_input_scale_tensors == 0
    assert summary.num_weight_scale_tensors == 9
    assert summary.num_routed_expert_weight_tensors == 4
    assert summary.num_routed_expert_scale_tensors == 4
    assert summary.num_shared_expert_weight_tensors == 2
    assert summary.num_shared_expert_scale_tensors == 2
    assert summary.num_layer0_mlp_weight_tensors == 3
    assert summary.num_layer0_mlp_scale_tensors == 3
    assert summary.num_layers == 3
    assert summary.num_single_shard_layers == 3
    assert summary.first_layer_shard == "model-00001-of-000064.safetensors"
    assert summary.first_moe_layer_shard == "model-00002-of-000064.safetensors"
    assert summary.last_moe_layer_shard == "model-00061-of-000064.safetensors"
    assert summary.hidden_size == 7168
    assert summary.intermediate_size == 18432
    assert summary.moe_intermediate_size == 2048
    assert summary.num_routed_experts == 384
    assert summary.num_experts_per_tok == 8
    assert summary.num_shared_experts == 1
    assert summary.num_hidden_layers == 61
    assert summary.first_k_dense_replace == 1


def test_local_kimi_mxfp4_artifact_summary_if_available() -> None:
    if not _LOCAL_KIMI_MXFP4_MODEL.is_dir():
        pytest.skip("local Kimi-K2.5 MXFP4 artifact is not available")

    summary = summarize_kimi_mxfp4_artifact(_LOCAL_KIMI_MXFP4_MODEL)

    assert summary.is_quark_mxfp4_dynamic_fp4 is True
    assert summary.num_quantization_excludes == 530
    assert summary.excludes_attention is True
    assert summary.excludes_lm_head is True
    assert summary.excludes_mlp_gate is True
    assert summary.indexed_total_size == 558995180568
    assert summary.num_tensors == 139613
    assert summary.num_safetensor_shards == 64
    assert summary.num_input_scale_tensors == 0
    assert summary.num_weight_scale_tensors == 69303
    assert summary.num_routed_expert_weight_tensors == 69120
    assert summary.num_routed_expert_scale_tensors == 69120
    assert summary.num_shared_expert_weight_tensors == 180
    assert summary.num_shared_expert_scale_tensors == 180
    assert summary.num_layer0_mlp_weight_tensors == 3
    assert summary.num_layer0_mlp_scale_tensors == 3
    assert summary.num_layers == 61
    assert summary.num_single_shard_layers == 61
    assert summary.first_layer_shard == "model-00001-of-000064.safetensors"
    assert summary.first_moe_layer_shard == "model-00002-of-000064.safetensors"
    assert summary.last_moe_layer_shard == "model-00061-of-000064.safetensors"
    assert summary.hidden_size == 7168
    assert summary.intermediate_size == 18432
    assert summary.moe_intermediate_size == 2048
    assert summary.num_routed_experts == 384
    assert summary.num_experts_per_tok == 8
    assert summary.num_shared_experts == 1
    assert summary.num_hidden_layers == 61
    assert summary.first_k_dense_replace == 1


def test_local_w4a8_config_accepts_matching_explicit_cli(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_w4a8_model_config(),
        "kimi-w4a8",
    )
    args = _parse_preflight_args(
        [
            "--model",
            str(model_dir),
            "--quantization",
            "w4a8_quark",
        ]
    )

    result = preflight_kimi_quantization(
        args.model,
        args.quantization,
    )

    assert result.quantization == "w4a8_quark"
    assert result.source == "local-json"
    assert result.is_quark is True


def test_local_mxfp4_config_accepts_matching_explicit_cli(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_mxfp4_model_config(),
        "kimi-mxfp4",
    )
    args = _parse_preflight_args(
        [
            "--model",
            str(model_dir),
            "--quantization",
            "mxfp4",
        ]
    )

    result = preflight_kimi_quantization(
        args.model,
        args.quantization,
    )

    assert result.quantization == "mxfp4"
    assert result.source == "local-json"
    assert result.is_quark is True


def test_local_quark_config_rejects_mismatched_explicit_cli(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_w4a8_model_config(),
        "kimi-w4a8",
    )

    with pytest.raises(ValueError, match="does not match"):
        preflight_kimi_quantization(str(model_dir), "w8a8_fp8")


def test_local_mxfp4_config_rejects_mismatched_explicit_cli(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_mxfp4_model_config(),
        "kimi-mxfp4",
    )

    with pytest.raises(ValueError, match="does not match"):
        preflight_kimi_quantization(str(model_dir), "w4a8_quark")


def test_nested_text_config_quantization_is_detected(tmp_path: Path) -> None:
    model_config = quark_kimi_w8a8_model_config()
    model_config["text_config"]["quantization_config"] = model_config.pop(
        "quantization_config"
    )
    model_dir = _write_config(tmp_path, model_config, "kimi-nested-w8a8")

    result = preflight_kimi_quantization(str(model_dir))

    assert result.quantization == "w8a8_fp8"
    assert result.source == "local-json"
    assert result.is_quark is True


def test_nvfp4_cli_path_remains_explicit() -> None:
    args = _parse_preflight_args(
        [
            "--model",
            "nvidia/Kimi-K2.5-NVFP4",
            "--quantization",
            "nvfp4",
        ]
    )

    result = preflight_kimi_quantization(
        args.model,
        args.quantization,
    )

    assert result.quantization == "nvfp4"
    assert result.source == "explicit-cli"
    assert result.requires_local_artifact is True
    assert result.is_quark is False


def _load_mxfp4_config_module_without_quantization_package_import():
    module_path = (
        Path(__file__).parents[2]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "layers"
        / "quantization"
        / "mxfp4.py"
    )

    class QuantizationConfig:
        @staticmethod
        def get_from_keys(config: dict, keys: list[str]):
            for key in keys:
                if key in config:
                    return config[key]
            raise ValueError(keys)

        @staticmethod
        def get_from_keys_or(config: dict, keys: list[str], default):
            try:
                return QuantizationConfig.get_from_keys(config, keys)
            except ValueError:
                return default

    stubs = {
        "tokenspeed.runtime.layers": types.ModuleType("tokenspeed.runtime.layers"),
        "tokenspeed.runtime.layers.quantization": types.ModuleType(
            "tokenspeed.runtime.layers.quantization"
        ),
        "tokenspeed.runtime.layers.quantization.base_config": types.ModuleType(
            "tokenspeed.runtime.layers.quantization.base_config"
        ),
    }
    stubs[
        "tokenspeed.runtime.layers.quantization.base_config"
    ].QuantizationConfig = QuantizationConfig
    for module in stubs.values():
        module.__path__ = []

    original_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_tokenspeed_mxfp4_config",
            module_path,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_runtime_mxfp4_override_detects_dynamic_fp4_quark_without_gpu_imports() -> None:
    module = _load_mxfp4_config_module_without_quantization_package_import()
    mxfp4_config = quark_kimi_mxfp4_model_config()["quantization_config"]
    w8a8_config = quark_kimi_w8a8_model_config()["quantization_config"]
    w4a8_config = quark_kimi_w4a8_model_config()["quantization_config"]
    mxfp4_a_fp8_config = {
        "quant_method": "quark",
        "global_quant_config": {
            "input_tensors": {"dtype": "fp8_e4m3"},
            "weight": {"dtype": "fp4", "group_size": 32},
        },
    }

    assert module.Mxfp4Config.override_quantization_method(mxfp4_config, None) == (
        "mxfp4"
    )
    assert module.Mxfp4Config.override_quantization_method(
        mxfp4_config,
        "mxfp4",
    ) == "mxfp4"
    assert (
        module.Mxfp4Config.override_quantization_method(
            mxfp4_config,
            "w4a8_quark",
        )
        is None
    )
    assert module.Mxfp4Config.override_quantization_method(
        mxfp4_a_fp8_config,
        None,
    ) == "mxfp4"
    assert module.Mxfp4Config.override_quantization_method(w8a8_config, None) is None
    assert module.Mxfp4Config.override_quantization_method(w4a8_config, None) is None

    parsed = module.Mxfp4Config.from_config(mxfp4_config)

    assert parsed.is_checkpoint_mxfp4_serialized is True
    assert parsed.is_w4a8_fp8 is False
