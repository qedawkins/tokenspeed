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
import re
from collections import defaultdict
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


@dataclass(frozen=True)
class KimiMxfp4ArtifactSummary:
    model: str
    config_path: str | None
    index_path: str | None
    is_quark_mxfp4_dynamic_fp4: bool
    num_quantization_excludes: int
    excludes_attention: bool
    excludes_lm_head: bool
    excludes_mlp_gate: bool
    indexed_total_size: int | None
    num_tensors: int
    num_safetensor_shards: int
    num_input_scale_tensors: int
    num_weight_scale_tensors: int
    num_routed_expert_weight_tensors: int
    num_routed_expert_scale_tensors: int
    num_shared_expert_weight_tensors: int
    num_shared_expert_scale_tensors: int
    num_layer0_mlp_weight_tensors: int
    num_layer0_mlp_scale_tensors: int
    num_layers: int
    num_single_shard_layers: int
    first_layer_shard: str | None
    first_moe_layer_shard: str | None
    last_moe_layer_shard: str | None
    hidden_size: int | None
    intermediate_size: int | None
    moe_intermediate_size: int | None
    num_routed_experts: int | None
    num_experts_per_tok: int | None
    num_shared_experts: int | None
    num_hidden_layers: int | None
    first_k_dense_replace: int | None


def normalize_cli_quantization(quantization: str | None) -> str | None:
    if quantization is None:
        return None
    normalized = quantization.lower()
    if not normalized:
        return None
    if normalized == "compressed_tensors":
        return "compressed-tensors"
    return normalized


def _is_fp4_e8m0_per_group(stage: object, *, is_dynamic: bool) -> bool:
    return (
        isinstance(stage, Mapping)
        and str(stage.get("dtype", "")).lower() in {"fp4", "mxfp4"}
        and stage.get("is_dynamic") is is_dynamic
        and str(stage.get("qscheme", "")).lower() == "per_group"
        and stage.get("group_size") in {32, "32"}
        and str(stage.get("scale_format", "")).lower() == "e8m0"
    )


def is_quark_mxfp4_dynamic_fp4_config(config: Mapping[str, Any]) -> bool:
    if str(config.get("quant_method", "")).lower() != "quark":
        return False
    global_quant_config = config.get("global_quant_config") or {}
    export = config.get("export") or {}
    if not isinstance(global_quant_config, Mapping) or not isinstance(export, Mapping):
        return False
    input_tensors = global_quant_config.get("input_tensors") or {}
    weight = global_quant_config.get("weight") or {}
    return (
        _is_fp4_e8m0_per_group(input_tensors, is_dynamic=True)
        and _is_fp4_e8m0_per_group(weight, is_dynamic=False)
        and str(export.get("pack_method", "")).lower() == "reorder"
        and str(export.get("weight_format", "")).lower() == "real_quantized"
    )


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
    if is_quark_mxfp4_dynamic_fp4_config(quantization_config):
        return "mxfp4"
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


_LAYER_NAME_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


def summarize_kimi_mxfp4_metadata(
    model_config: Mapping[str, Any],
    safetensors_index: Mapping[str, Any],
    *,
    model: str = "json-fixture",
    config_path: str | None = None,
    index_path: str | None = None,
) -> KimiMxfp4ArtifactSummary:
    quantization_config = _extract_quantization_config(model_config) or {}
    text_config = model_config.get("text_config") or {}
    if not isinstance(text_config, Mapping):
        text_config = {}
    excludes = (
        quantization_config.get("exclude")
        if isinstance(quantization_config, Mapping)
        else None
    )
    exclude_patterns = [str(item) for item in excludes] if isinstance(excludes, list) else []

    weight_map = safetensors_index.get("weight_map") or {}
    if not isinstance(weight_map, Mapping):
        raise ValueError("safetensors index must contain a weight_map object")
    metadata = safetensors_index.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    layer_shards: dict[int, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        match = _LAYER_NAME_RE.match(str(name))
        if match is not None:
            layer_shards[int(match.group(1))].add(str(shard))

    def count_names(suffix: str, contains: str) -> int:
        return sum(
            contains in str(name) and str(name).endswith(suffix)
            for name in weight_map
        )

    def layer_shard(layer_id: int) -> str | None:
        shards = sorted(layer_shards.get(layer_id, ()))
        return shards[0] if len(shards) == 1 else None

    return KimiMxfp4ArtifactSummary(
        model=model,
        config_path=config_path,
        index_path=index_path,
        is_quark_mxfp4_dynamic_fp4=(
            isinstance(quantization_config, Mapping)
            and is_quark_mxfp4_dynamic_fp4_config(quantization_config)
        ),
        num_quantization_excludes=len(exclude_patterns),
        excludes_attention=any("self_attn" in item for item in exclude_patterns),
        excludes_lm_head=any("lm_head" in item for item in exclude_patterns),
        excludes_mlp_gate=any(
            "mlp.gate" in item or "mlp\\.gate" in item
            for item in exclude_patterns
        ),
        indexed_total_size=_as_optional_int(metadata.get("total_size")),
        num_tensors=len(weight_map),
        num_safetensor_shards=len(set(weight_map.values())),
        num_input_scale_tensors=sum("input_scale" in str(name) for name in weight_map),
        num_weight_scale_tensors=sum(
            str(name).endswith("weight_scale") for name in weight_map
        ),
        num_routed_expert_weight_tensors=count_names(".weight", ".mlp.experts."),
        num_routed_expert_scale_tensors=count_names(
            ".weight_scale", ".mlp.experts."
        ),
        num_shared_expert_weight_tensors=count_names(
            ".weight", ".mlp.shared_experts."
        ),
        num_shared_expert_scale_tensors=count_names(
            ".weight_scale", ".mlp.shared_experts."
        ),
        num_layer0_mlp_weight_tensors=count_names(
            ".weight", "language_model.model.layers.0.mlp."
        ),
        num_layer0_mlp_scale_tensors=count_names(
            ".weight_scale", "language_model.model.layers.0.mlp."
        ),
        num_layers=len(layer_shards),
        num_single_shard_layers=sum(len(shards) == 1 for shards in layer_shards.values()),
        first_layer_shard=layer_shard(0),
        first_moe_layer_shard=layer_shard(1),
        last_moe_layer_shard=layer_shard(
            _as_optional_int(text_config.get("num_hidden_layers"), default=0) - 1
        ),
        hidden_size=_as_optional_int(text_config.get("hidden_size")),
        intermediate_size=_as_optional_int(text_config.get("intermediate_size")),
        moe_intermediate_size=_as_optional_int(text_config.get("moe_intermediate_size")),
        num_routed_experts=_as_optional_int(text_config.get("n_routed_experts")),
        num_experts_per_tok=_as_optional_int(text_config.get("num_experts_per_tok")),
        num_shared_experts=_as_optional_int(text_config.get("n_shared_experts")),
        num_hidden_layers=_as_optional_int(text_config.get("num_hidden_layers")),
        first_k_dense_replace=_as_optional_int(text_config.get("first_k_dense_replace")),
    )


def summarize_kimi_mxfp4_artifact(model: str | os.PathLike[str]) -> KimiMxfp4ArtifactSummary:
    model_path = Path(model).expanduser()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    with config_path.open() as f:
        model_config = json.load(f)
    with index_path.open() as f:
        safetensors_index = json.load(f)
    return summarize_kimi_mxfp4_metadata(
        model_config,
        safetensors_index,
        model=str(model_path),
        config_path=str(config_path),
        index_path=str(index_path),
    )


def _as_optional_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
            is_quark=quantization in {"w8a8_fp8", "w4a8_quark", "mxfp4"},
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
        is_quark=quantization in {"w8a8_fp8", "w4a8_quark", "mxfp4"},
    )
