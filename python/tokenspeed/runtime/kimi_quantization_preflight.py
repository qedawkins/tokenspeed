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

_MXFP4_PACK_FACTOR = 2
_MXFP4_SCALE_BLOCK = 32
_KIMI_MXFP4_BF16_PASSTHROUGH_PATTERNS = (
    "self_attn",
    "mlp.gate",
    "lm_head",
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


@dataclass(frozen=True)
class KimiMxfp4ProjectionShard:
    name: str
    logical_shape: tuple[int, int]
    checkpoint_weight_shape: tuple[int, int]
    checkpoint_scale_shape: tuple[int, int]
    rank_logical_shape: tuple[int, int]
    rank_weight_shape: tuple[int, int]
    rank_scale_shape: tuple[int, int]
    checkpoint_weight_slice: tuple[tuple[int, int], tuple[int, int]]
    checkpoint_scale_slice: tuple[tuple[int, int], tuple[int, int]]
    destination_weight_slice: tuple[tuple[int, int], tuple[int, int]]
    destination_scale_slice: tuple[tuple[int, int], tuple[int, int]]
    logical_shard_axis: int
    checkpoint_weight_shard_axis: int
    checkpoint_scale_shard_axis: int


@dataclass(frozen=True)
class KimiMxfp4ExpertOwnership:
    num_global_experts: int
    ep_size: int
    ep_rank: int
    experts_per_ep_rank: int
    global_expert_start: int
    global_expert_end: int

    def owner_rank(self, global_expert_id: int) -> int:
        self._validate_global_expert(global_expert_id)
        return global_expert_id // self.experts_per_ep_rank

    def local_expert_id(self, global_expert_id: int) -> int:
        self._validate_global_expert(global_expert_id)
        return global_expert_id % self.experts_per_ep_rank

    def owns_expert(self, global_expert_id: int) -> bool:
        self._validate_global_expert(global_expert_id)
        return self.global_expert_start <= global_expert_id < self.global_expert_end

    def owned_expert_ids(self) -> range:
        return range(self.global_expert_start, self.global_expert_end)

    def _validate_global_expert(self, global_expert_id: int) -> None:
        if not 0 <= global_expert_id < self.num_global_experts:
            raise ValueError(
                f"global expert id {global_expert_id} is outside "
                f"[0, {self.num_global_experts})"
            )


@dataclass(frozen=True)
class KimiMxfp4MlpShardContract:
    name: str
    checkpoint_prefix: str
    uses_ep_ownership: bool
    num_rank_local_experts: int | None
    rank_local_w13_weight_shape: tuple[int, ...]
    rank_local_w13_scale_shape: tuple[int, ...]
    rank_local_w2_weight_shape: tuple[int, ...]
    rank_local_w2_scale_shape: tuple[int, ...]
    gate_proj: KimiMxfp4ProjectionShard
    up_proj: KimiMxfp4ProjectionShard
    down_proj: KimiMxfp4ProjectionShard


@dataclass(frozen=True)
class KimiMxfp4ShardingContract:
    tp_size: int
    ep_size: int
    tp_rank: int
    ep_rank: int
    hidden_size: int
    dense_intermediate_size: int
    moe_intermediate_size: int
    num_shared_experts: int
    first_k_dense_replace: int
    pack_factor: int
    scale_block: int
    bf16_passthrough_patterns: tuple[str, ...]
    routed_ownership: KimiMxfp4ExpertOwnership
    routed: KimiMxfp4MlpShardContract
    shared: KimiMxfp4MlpShardContract
    dense: KimiMxfp4MlpShardContract


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


def build_kimi_mxfp4_sharding_contract(
    model_config: Mapping[str, Any],
    *,
    tp_size: int = 4,
    ep_size: int = 4,
    tp_rank: int = 0,
    ep_rank: int = 0,
) -> KimiMxfp4ShardingContract:
    quantization_config = _extract_quantization_config(model_config)
    if quantization_config is None or not is_quark_mxfp4_dynamic_fp4_config(
        quantization_config
    ):
        raise ValueError("Kimi MXFP4 sharding contract requires Quark dynamic-FP4 MXFP4")

    text_config = model_config.get("text_config") or {}
    if not isinstance(text_config, Mapping):
        raise ValueError("Kimi MXFP4 sharding contract requires text_config metadata")

    _validate_rank("TP", tp_size, tp_rank)
    _validate_rank("EP", ep_size, ep_rank)

    hidden_size = _required_text_int(text_config, "hidden_size")
    dense_intermediate_size = _required_text_int(text_config, "intermediate_size")
    moe_intermediate_size = _required_text_int(text_config, "moe_intermediate_size")
    num_routed_experts = _required_text_int(text_config, "n_routed_experts")
    num_shared_experts = _required_text_int(text_config, "n_shared_experts")
    first_k_dense_replace = _required_text_int(text_config, "first_k_dense_replace")

    if num_routed_experts % ep_size != 0:
        raise ValueError(
            "Kimi MXFP4 routed experts must be divisible by EP size: "
            f"{num_routed_experts} vs {ep_size}"
        )
    if moe_intermediate_size % tp_size != 0:
        raise ValueError(
            "Kimi MXFP4 MoE intermediate size must be divisible by TP size: "
            f"{moe_intermediate_size} vs {tp_size}"
        )
    if dense_intermediate_size % tp_size != 0:
        raise ValueError(
            "Kimi MXFP4 dense intermediate size must be divisible by TP size: "
            f"{dense_intermediate_size} vs {tp_size}"
        )
    _validate_mxfp4_dims(moe_intermediate_size, hidden_size)
    _validate_mxfp4_dims(dense_intermediate_size, hidden_size)
    _validate_mxfp4_dims(hidden_size, moe_intermediate_size)
    _validate_mxfp4_dims(hidden_size, dense_intermediate_size)

    experts_per_ep_rank = num_routed_experts // ep_size
    expert_start = ep_rank * experts_per_ep_rank
    expert_end = expert_start + experts_per_ep_rank
    routed_ownership = KimiMxfp4ExpertOwnership(
        num_global_experts=num_routed_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        experts_per_ep_rank=experts_per_ep_rank,
        global_expert_start=expert_start,
        global_expert_end=expert_end,
    )

    routed = _make_mlp_shard_contract(
        name="routed",
        checkpoint_prefix="mlp.experts.<global_expert_id>",
        hidden_size=hidden_size,
        intermediate_size=moe_intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_rank_local_experts=experts_per_ep_rank,
        uses_ep_ownership=True,
    )
    shared = _make_mlp_shard_contract(
        name="shared",
        checkpoint_prefix="mlp.shared_experts",
        hidden_size=hidden_size,
        intermediate_size=moe_intermediate_size * num_shared_experts,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_rank_local_experts=None,
        uses_ep_ownership=False,
    )
    dense = _make_mlp_shard_contract(
        name="dense",
        checkpoint_prefix="mlp",
        hidden_size=hidden_size,
        intermediate_size=dense_intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_rank_local_experts=None,
        uses_ep_ownership=False,
    )

    return KimiMxfp4ShardingContract(
        tp_size=tp_size,
        ep_size=ep_size,
        tp_rank=tp_rank,
        ep_rank=ep_rank,
        hidden_size=hidden_size,
        dense_intermediate_size=dense_intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_shared_experts=num_shared_experts,
        first_k_dense_replace=first_k_dense_replace,
        pack_factor=_MXFP4_PACK_FACTOR,
        scale_block=_MXFP4_SCALE_BLOCK,
        bf16_passthrough_patterns=_KIMI_MXFP4_BF16_PASSTHROUGH_PATTERNS,
        routed_ownership=routed_ownership,
        routed=routed,
        shared=shared,
        dense=dense,
    )


def _validate_rank(name: str, size: int, rank: int) -> None:
    if size <= 0:
        raise ValueError(f"{name} size must be positive, got {size}")
    if not 0 <= rank < size:
        raise ValueError(f"{name} rank {rank} is outside [0, {size})")


def _required_text_int(text_config: Mapping[str, Any], key: str) -> int:
    value = _as_optional_int(text_config.get(key))
    if value is None:
        raise ValueError(f"Kimi MXFP4 text_config is missing integer {key!r}")
    if value <= 0:
        raise ValueError(f"Kimi MXFP4 text_config {key!r} must be positive")
    return value


def _validate_mxfp4_dims(out_features: int, in_features: int) -> None:
    if in_features % _MXFP4_PACK_FACTOR != 0:
        raise ValueError(
            f"MXFP4 input dim {in_features} must be divisible by "
            f"pack factor {_MXFP4_PACK_FACTOR}"
        )
    if in_features % _MXFP4_SCALE_BLOCK != 0:
        raise ValueError(
            f"MXFP4 input dim {in_features} must be divisible by "
            f"scale block {_MXFP4_SCALE_BLOCK}"
        )
    if out_features <= 0:
        raise ValueError(f"MXFP4 output dim must be positive, got {out_features}")


def _make_mlp_shard_contract(
    *,
    name: str,
    checkpoint_prefix: str,
    hidden_size: int,
    intermediate_size: int,
    tp_size: int,
    tp_rank: int,
    num_rank_local_experts: int | None,
    uses_ep_ownership: bool,
) -> KimiMxfp4MlpShardContract:
    local_intermediate = intermediate_size // tp_size
    gate_proj = _make_column_sharded_projection(
        name=f"{name}.gate_proj",
        out_features=intermediate_size,
        in_features=hidden_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        destination_row_offset=0,
    )
    up_proj = _make_column_sharded_projection(
        name=f"{name}.up_proj",
        out_features=intermediate_size,
        in_features=hidden_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        destination_row_offset=local_intermediate,
    )
    down_proj = _make_row_sharded_projection(
        name=f"{name}.down_proj",
        out_features=hidden_size,
        in_features=intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )

    w13_shape = (2 * local_intermediate, hidden_size // _MXFP4_PACK_FACTOR)
    w13_scale_shape = (2 * local_intermediate, hidden_size // _MXFP4_SCALE_BLOCK)
    w2_shape = (hidden_size, local_intermediate // _MXFP4_PACK_FACTOR)
    w2_scale_shape = (hidden_size, local_intermediate // _MXFP4_SCALE_BLOCK)
    if num_rank_local_experts is not None:
        rank_local_w13_weight_shape = (num_rank_local_experts, *w13_shape)
        rank_local_w13_scale_shape = (num_rank_local_experts, *w13_scale_shape)
        rank_local_w2_weight_shape = (num_rank_local_experts, *w2_shape)
        rank_local_w2_scale_shape = (num_rank_local_experts, *w2_scale_shape)
    else:
        rank_local_w13_weight_shape = w13_shape
        rank_local_w13_scale_shape = w13_scale_shape
        rank_local_w2_weight_shape = w2_shape
        rank_local_w2_scale_shape = w2_scale_shape

    return KimiMxfp4MlpShardContract(
        name=name,
        checkpoint_prefix=checkpoint_prefix,
        uses_ep_ownership=uses_ep_ownership,
        num_rank_local_experts=num_rank_local_experts,
        rank_local_w13_weight_shape=rank_local_w13_weight_shape,
        rank_local_w13_scale_shape=rank_local_w13_scale_shape,
        rank_local_w2_weight_shape=rank_local_w2_weight_shape,
        rank_local_w2_scale_shape=rank_local_w2_scale_shape,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
    )


def _make_column_sharded_projection(
    *,
    name: str,
    out_features: int,
    in_features: int,
    tp_size: int,
    tp_rank: int,
    destination_row_offset: int,
) -> KimiMxfp4ProjectionShard:
    local_out = out_features // tp_size
    row_start = tp_rank * local_out
    row_end = row_start + local_out
    weight_cols = in_features // _MXFP4_PACK_FACTOR
    scale_cols = in_features // _MXFP4_SCALE_BLOCK
    destination_start = destination_row_offset
    destination_end = destination_start + local_out
    return KimiMxfp4ProjectionShard(
        name=name,
        logical_shape=(out_features, in_features),
        checkpoint_weight_shape=(out_features, weight_cols),
        checkpoint_scale_shape=(out_features, scale_cols),
        rank_logical_shape=(local_out, in_features),
        rank_weight_shape=(local_out, weight_cols),
        rank_scale_shape=(local_out, scale_cols),
        checkpoint_weight_slice=((row_start, row_end), (0, weight_cols)),
        checkpoint_scale_slice=((row_start, row_end), (0, scale_cols)),
        destination_weight_slice=(
            (destination_start, destination_end),
            (0, weight_cols),
        ),
        destination_scale_slice=(
            (destination_start, destination_end),
            (0, scale_cols),
        ),
        logical_shard_axis=0,
        checkpoint_weight_shard_axis=0,
        checkpoint_scale_shard_axis=0,
    )


def _make_row_sharded_projection(
    *,
    name: str,
    out_features: int,
    in_features: int,
    tp_size: int,
    tp_rank: int,
) -> KimiMxfp4ProjectionShard:
    local_in = in_features // tp_size
    logical_start = tp_rank * local_in
    logical_end = logical_start + local_in
    weight_cols = in_features // _MXFP4_PACK_FACTOR
    scale_cols = in_features // _MXFP4_SCALE_BLOCK
    local_weight_cols = local_in // _MXFP4_PACK_FACTOR
    local_scale_cols = local_in // _MXFP4_SCALE_BLOCK
    weight_col_start = logical_start // _MXFP4_PACK_FACTOR
    weight_col_end = logical_end // _MXFP4_PACK_FACTOR
    scale_col_start = logical_start // _MXFP4_SCALE_BLOCK
    scale_col_end = logical_end // _MXFP4_SCALE_BLOCK
    return KimiMxfp4ProjectionShard(
        name=name,
        logical_shape=(out_features, in_features),
        checkpoint_weight_shape=(out_features, weight_cols),
        checkpoint_scale_shape=(out_features, scale_cols),
        rank_logical_shape=(out_features, local_in),
        rank_weight_shape=(out_features, local_weight_cols),
        rank_scale_shape=(out_features, local_scale_cols),
        checkpoint_weight_slice=(
            (0, out_features),
            (weight_col_start, weight_col_end),
        ),
        checkpoint_scale_slice=(
            (0, out_features),
            (scale_col_start, scale_col_end),
        ),
        destination_weight_slice=((0, out_features), (0, local_weight_cols)),
        destination_scale_slice=((0, out_features), (0, local_scale_cols)),
        logical_shard_axis=1,
        checkpoint_weight_shard_axis=1,
        checkpoint_scale_shard_axis=1,
    )


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
