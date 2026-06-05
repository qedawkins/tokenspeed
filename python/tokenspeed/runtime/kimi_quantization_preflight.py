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

"""JSON-only Kimi quantization preflight helpers.

These helpers avoid importing runtime quantization modules or
calling Hugging Face APIs. They are used by tests/docs to validate that local
Kimi Quark metadata and explicit CLI choices route to AMD quantization paths
without touching checkpoint weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_EXPLICIT_QUANTIZATIONS = frozenset(
    {
        "fp8",
        "nvfp4",
        "mxfp4",
        "compressed-tensors",
        "compressed_tensors",
        "w8a8_fp8",
        "w4a8_quark",
    }
)


@dataclass(frozen=True)
class KimiQuantizationPreflight:
    model: str
    quantization: str | None
    source: str
    config_path: str | None = None
    requires_local_artifact: bool = False
    is_quark: bool = False


def normalize_cli_quantization(quantization: str | None) -> str | None:
    if quantization is None:
        return None
    normalized = quantization.lower()
    if not normalized:
        return None
    if normalized == "compressed_tensors":
        return "compressed-tensors"
    return normalized


def _is_quark_w8a8_fp8(config: Mapping[str, Any]) -> bool:
    if str(config.get("quant_method", "")).lower() != "quark":
        return False
    global_quant_config = config.get("global_quant_config") or {}
    if not isinstance(global_quant_config, Mapping):
        return False
    weight = global_quant_config.get("weight") or {}
    input_tensors = global_quant_config.get("input_tensors") or {}
    if not isinstance(weight, Mapping) or not isinstance(input_tensors, Mapping):
        return False
    return (
        str(weight.get("dtype", "")).lower() == "fp8_e4m3"
        and weight.get("is_dynamic") is False
        and str(weight.get("qscheme", "")).lower() == "per_channel"
        and str(input_tensors.get("dtype", "")).lower() == "fp8_e4m3"
        and input_tensors.get("is_dynamic") is True
        and str(input_tensors.get("qscheme", "")).lower() == "per_tensor"
    )


def _is_static_fp8_e4m3_stage(stage: object) -> bool:
    return (
        isinstance(stage, Mapping)
        and str(stage.get("dtype", "")).lower() == "fp8_e4m3"
        and stage.get("is_dynamic") is False
    )


def _is_static_int4_per_channel_stage(stage: object) -> bool:
    return (
        isinstance(stage, Mapping)
        and str(stage.get("dtype", "")).lower() == "int4"
        and stage.get("is_dynamic") is False
        and str(stage.get("qscheme", "")).lower() == "per_channel"
        and stage.get("ch_axis") in {0, "0"}
    )


def _is_quark_w4a8(config: Mapping[str, Any]) -> bool:
    if str(config.get("quant_method", "")).lower() != "quark":
        return False
    global_quant_config = config.get("global_quant_config") or {}
    if not isinstance(global_quant_config, Mapping):
        return False
    input_tensors = global_quant_config.get("input_tensors") or {}
    weight = global_quant_config.get("weight") or []
    export = config.get("export") or {}
    if (
        not isinstance(input_tensors, Mapping)
        or not isinstance(weight, list)
        or not isinstance(export, Mapping)
        or len(weight) != 2
    ):
        return False
    return (
        str(input_tensors.get("dtype", "")).lower() == "fp8_e4m3"
        and input_tensors.get("is_dynamic") is True
        and str(input_tensors.get("qscheme", "")).lower() == "per_tensor"
        and _is_static_fp8_e4m3_stage(weight[0])
        and _is_static_int4_per_channel_stage(weight[1])
        and str(export.get("pack_method", "")).lower() == "reorder"
        and str(export.get("weight_format", "")).lower() == "real_quantized"
    )


def detect_quark_quantization(
    quantization_config: Mapping[str, Any],
) -> str | None:
    if _is_quark_w8a8_fp8(quantization_config):
        return "w8a8_fp8"
    if _is_quark_w4a8(quantization_config):
        return "w4a8_quark"
    return None


def _generic_quantization_method(config: Mapping[str, Any]) -> str | None:
    raw_quant_method = config.get("quant_method")
    quant_method = (
        normalize_cli_quantization(str(raw_quant_method))
        if raw_quant_method is not None
        else None
    )
    if quant_method:
        return quant_method
    quant_algo = str(config.get("quant_algo", "")).upper()
    if quant_algo == "NVFP4":
        return "nvfp4"
    return None


def _extract_quantization_config(
    model_config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for key in ("quantization_config", "compression_config"):
        value = model_config.get(key)
        if isinstance(value, Mapping):
            return value
    text_config = model_config.get("text_config")
    if isinstance(text_config, Mapping):
        for key in ("quantization_config", "compression_config"):
            value = text_config.get(key)
            if isinstance(value, Mapping):
                return value
    return None


def resolve_kimi_quantization_from_config(
    model_config: Mapping[str, Any],
    user_quantization: str | None = None,
) -> str | None:
    explicit = normalize_cli_quantization(user_quantization)
    if explicit is not None and explicit not in _EXPLICIT_QUANTIZATIONS:
        raise ValueError(f"Unknown quantization method: {explicit}")

    quantization_config = _extract_quantization_config(model_config)
    detected: str | None = None
    if quantization_config is not None:
        detected = detect_quark_quantization(quantization_config)
        if detected is None:
            detected = _generic_quantization_method(quantization_config)

    if detected is None:
        return explicit
    if explicit is not None and explicit != detected:
        raise ValueError(
            "Quantization method specified by local JSON metadata "
            f"({detected}) does not match the explicit quantization "
            f"argument ({explicit})."
        )
    return detected


def _local_config_path(model: str) -> Path | None:
    path = Path(model).expanduser()
    if path.is_dir():
        candidate = path / "config.json"
        return candidate if candidate.is_file() else None
    if path.is_file() and path.name == "config.json":
        return path
    return None


def preflight_kimi_quantization(
    model: str,
    user_quantization: str | None = None,
    *,
    model_config: Mapping[str, Any] | None = None,
) -> KimiQuantizationPreflight:
    config_path = None
    source = "explicit-cli"
    if model_config is None:
        local_config = _local_config_path(model)
        if local_config is not None:
            config_path = str(local_config)
            with local_config.open() as f:
                model_config = json.load(f)
            source = "local-json"
    else:
        source = "json-fixture"

    if model_config is None:
        quantization = normalize_cli_quantization(user_quantization)
        return KimiQuantizationPreflight(
            model=model,
            quantization=quantization,
            source=source if quantization else "missing-local-json",
            config_path=None,
            requires_local_artifact=not os.path.isdir(os.path.expanduser(model)),
            is_quark=quantization in {"w8a8_fp8", "w4a8_quark"},
        )

    quantization = resolve_kimi_quantization_from_config(
        model_config,
        user_quantization,
    )
    return KimiQuantizationPreflight(
        model=model,
        quantization=quantization,
        source=source,
        config_path=config_path,
        requires_local_artifact=False,
        is_quark=quantization in {"w8a8_fp8", "w4a8_quark"},
    )
