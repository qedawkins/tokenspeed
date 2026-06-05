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

from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig


def _normalize_ignored_layer_patterns(patterns: list[str] | None) -> list[str]:
    if not patterns:
        return []
    normalized: list[str] = []
    for raw in patterns:
        if not isinstance(raw, str) or not raw:
            continue
        if raw.startswith("re:") or "*" not in raw:
            normalized.append(raw)
            continue
        import re

        regex = re.escape(raw).replace(r"\*", ".*")
        normalized.append(f"re:{regex}")
    return normalized


def _is_static_fp8_e4m3_stage(stage: object) -> bool:
    if not isinstance(stage, dict):
        return False
    return (
        str(stage.get("dtype", "")).lower() == "fp8_e4m3"
        and stage.get("is_dynamic") is False
    )


def _is_static_int4_per_channel_stage(stage: object) -> bool:
    if not isinstance(stage, dict):
        return False
    return (
        str(stage.get("dtype", "")).lower() == "int4"
        and stage.get("is_dynamic") is False
        and str(stage.get("qscheme", "")).lower() == "per_channel"
        and stage.get("ch_axis") in {0, "0"}
    )


def _is_quark_w4a8(config: dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        return False
    if str(config.get("quant_method", "")).lower() != "quark":
        return False

    global_quant_config = config.get("global_quant_config") or {}
    input_tensors = global_quant_config.get("input_tensors") or {}
    weight = global_quant_config.get("weight") or []
    export = config.get("export") or {}

    if not isinstance(input_tensors, dict) or not isinstance(weight, list):
        return False
    if len(weight) != 2:
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


class W4A8QuarkConfig(QuantizationConfig):
    """AMD Quark Kimi W4A8 contract: INT4 per-channel experts plus dynamic 8-bit activations."""

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 95

    @classmethod
    def get_name(self) -> str:
        return "w4a8_quark"

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        if not _is_quark_w4a8(config):
            raise ValueError("Quark W4A8 config must use staged fp8_e4m3 + int4 metadata")
        raw_ignored = cls.get_from_keys_or(config, ["ignored_layers", "exclude"], None)
        ignored_layers = _normalize_ignored_layer_patterns(raw_ignored)
        return cls(ignored_layers=ignored_layers)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if user_quant in {"w4a8_quark", None} and _is_quark_w4a8(hf_quant_cfg):
            return "w4a8_quark"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


__all__ = ["W4A8QuarkConfig"]
