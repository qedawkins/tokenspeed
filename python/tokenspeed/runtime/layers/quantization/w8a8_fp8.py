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

from typing import Any

import torch

from tokenspeed.runtime.layers.quantization import QuantizationConfig


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


def _is_quark_w8a8_fp8(config: dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        return False
    if str(config.get("quant_method", "")).lower() != "quark":
        return False

    global_quant_config = config.get("global_quant_config") or {}
    weight = global_quant_config.get("weight") or {}
    input_tensors = global_quant_config.get("input_tensors") or {}

    if not isinstance(weight, dict) or not isinstance(input_tensors, dict):
        return False

    return (
        str(weight.get("dtype", "")).lower() == "fp8_e4m3"
        and weight.get("is_dynamic") is False
        and str(weight.get("qscheme", "")).lower() == "per_channel"
        and str(input_tensors.get("dtype", "")).lower() == "fp8_e4m3"
        and input_tensors.get("is_dynamic") is True
        and str(input_tensors.get("qscheme", "")).lower() == "per_tensor"
    )


class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    Weight Quantization:
    - Method: Static quantization
    - Granularity: Per-channel
    - Type: Symmetric

    Activation Quantization:
    - Method: Dynamic quantization
    - Granularity: Per-token
    - Type: Symmetric

    Note:
    - For models without offline quantization, weights will be quantized during model loading:
        - If CUTLASS is supported: Per-channel weight quantization is used
        - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = (
            "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method
            or _is_quark_w8a8_fp8(config)
        )
        raw_ignored = cls.get_from_keys_or(config, ["ignored_layers", "exclude"], None)
        ignored_layers = _normalize_ignored_layer_patterns(raw_ignored)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            ignored_layers=ignored_layers,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if user_quant in {"w8a8_fp8", None} and _is_quark_w8a8_fp8(hf_quant_cfg):
            return "w8a8_fp8"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []
