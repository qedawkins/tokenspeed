"""No-weight Kimi Quark CLI and local metadata preflight tests."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

from test.runtime.fixtures.kimi_quark_metadata import (
    quark_kimi_w4a8_model_config,
    quark_kimi_w8a8_model_config,
)
from tokenspeed.runtime.kimi_quantization_preflight import (
    preflight_kimi_quantization,
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


def test_local_quark_config_rejects_mismatched_explicit_cli(tmp_path: Path) -> None:
    model_dir = _write_config(
        tmp_path,
        quark_kimi_w4a8_model_config(),
        "kimi-w4a8",
    )

    with pytest.raises(ValueError, match="does not match"):
        preflight_kimi_quantization(str(model_dir), "w8a8_fp8")


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
