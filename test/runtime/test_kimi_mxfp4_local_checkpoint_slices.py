"""Read-only local Kimi MXFP4 checkpoint slice preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.kimi_quantization_preflight import (
    KimiMxfp4ProjectionShard,
    build_kimi_mxfp4_sharding_contract,
    summarize_kimi_mxfp4_artifact,
)


_LOCAL_KIMI_MXFP4_MODEL = Path(
    "/data/models/hf/hub/models--amd--Kimi-K2.5-MXFP4/"
    "snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a"
)


def _require_local_kimi_mxfp4_model() -> Path:
    if not _LOCAL_KIMI_MXFP4_MODEL.is_dir():
        pytest.skip("local Kimi-K2.5 MXFP4 artifact is not available")
    return _LOCAL_KIMI_MXFP4_MODEL


def _safe_open():
    return pytest.importorskip("safetensors").safe_open


def _model_config(model: Path) -> dict:
    with (model / "config.json").open() as f:
        return json.load(f)


def _weight_map(model: Path) -> dict[str, str]:
    with (model / "model.safetensors.index.json").open() as f:
        index = json.load(f)
    return index["weight_map"]


def _tensor_slice(model: Path, weight_map: dict[str, str], name: str):
    safe_open = _safe_open()
    with safe_open(model / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_slice(name)


def _assert_tensor(
    model: Path,
    weight_map: dict[str, str],
    name: str,
    *,
    shape: tuple[int, ...],
    safetensors_dtype: str,
) -> None:
    tensor_slice = _tensor_slice(model, weight_map, name)
    assert tuple(tensor_slice.get_shape()) == shape
    assert tensor_slice.get_dtype() == safetensors_dtype


def _assert_readable_2d_slice(
    model: Path,
    weight_map: dict[str, str],
    name: str,
    checkpoint_slice: tuple[tuple[int, int], tuple[int, int]],
    *,
    dtype: torch.dtype,
) -> None:
    tensor_slice = _tensor_slice(model, weight_map, name)
    (row_start, row_end), (col_start, col_end) = checkpoint_slice
    sample = tensor_slice[
        row_start : min(row_start + 1, row_end),
        col_start : min(col_start + 8, col_end),
    ]
    assert sample.ndim == 2
    assert sample.shape[0] == 1
    assert 1 <= sample.shape[1] <= 8
    assert sample.dtype == dtype
    assert sample.isfinite().all() if sample.is_floating_point() else sample.numel() > 0


def _assert_projection(
    model: Path,
    weight_map: dict[str, str],
    prefix: str,
    projection_name: str,
    contract: KimiMxfp4ProjectionShard,
) -> None:
    weight_name = f"{prefix}.{projection_name}.weight"
    scale_name = f"{prefix}.{projection_name}.weight_scale"
    _assert_tensor(
        model,
        weight_map,
        weight_name,
        shape=contract.checkpoint_weight_shape,
        safetensors_dtype="U8",
    )
    _assert_tensor(
        model,
        weight_map,
        scale_name,
        shape=contract.checkpoint_scale_shape,
        safetensors_dtype="U8",
    )
    _assert_readable_2d_slice(
        model,
        weight_map,
        weight_name,
        contract.checkpoint_weight_slice,
        dtype=torch.uint8,
    )
    _assert_readable_2d_slice(
        model,
        weight_map,
        scale_name,
        contract.checkpoint_scale_slice,
        dtype=torch.uint8,
    )


def test_local_kimi_mxfp4_selected_routed_slices_match_tp_ep_contract() -> None:
    model = _require_local_kimi_mxfp4_model()
    model_config = _model_config(model)
    weight_map = _weight_map(model)

    for ep_rank in range(4):
        for tp_rank in range(4):
            contract = build_kimi_mxfp4_sharding_contract(
                model_config,
                tp_size=4,
                ep_size=4,
                tp_rank=tp_rank,
                ep_rank=ep_rank,
            )
            ownership = contract.routed_ownership
            assert contract.routed.rank_local_w13_weight_shape == (96, 1024, 3584)
            assert contract.routed.rank_local_w13_scale_shape == (96, 1024, 224)
            assert contract.routed.rank_local_w2_weight_shape == (96, 7168, 256)
            assert contract.routed.rank_local_w2_scale_shape == (96, 7168, 16)

            for expert_id in (
                ownership.global_expert_start,
                ownership.global_expert_end - 1,
            ):
                assert ownership.owner_rank(expert_id) == ep_rank
                assert 0 <= ownership.local_expert_id(expert_id) < 96
                prefix = f"language_model.model.layers.1.mlp.experts.{expert_id}"
                _assert_projection(
                    model,
                    weight_map,
                    prefix,
                    "gate_proj",
                    contract.routed.gate_proj,
                )
                _assert_projection(
                    model,
                    weight_map,
                    prefix,
                    "up_proj",
                    contract.routed.up_proj,
                )
                _assert_projection(
                    model,
                    weight_map,
                    prefix,
                    "down_proj",
                    contract.routed.down_proj,
                )


def test_local_kimi_mxfp4_shared_and_dense_slices_match_tp_contract() -> None:
    model = _require_local_kimi_mxfp4_model()
    model_config = _model_config(model)
    weight_map = _weight_map(model)

    for tp_rank in range(4):
        contract = build_kimi_mxfp4_sharding_contract(
            model_config,
            tp_size=4,
            ep_size=4,
            tp_rank=tp_rank,
            ep_rank=0,
        )
        shared_prefix = "language_model.model.layers.1.mlp.shared_experts"
        dense_prefix = "language_model.model.layers.0.mlp"

        assert contract.shared.rank_local_w13_weight_shape == (1024, 3584)
        assert contract.shared.rank_local_w13_scale_shape == (1024, 224)
        assert contract.shared.rank_local_w2_weight_shape == (7168, 256)
        assert contract.shared.rank_local_w2_scale_shape == (7168, 16)
        assert contract.dense.rank_local_w13_weight_shape == (9216, 3584)
        assert contract.dense.rank_local_w13_scale_shape == (9216, 224)
        assert contract.dense.rank_local_w2_weight_shape == (7168, 2304)
        assert contract.dense.rank_local_w2_scale_shape == (7168, 144)

        for prefix, mlp_contract in (
            (shared_prefix, contract.shared),
            (dense_prefix, contract.dense),
        ):
            _assert_projection(
                model,
                weight_map,
                prefix,
                "gate_proj",
                mlp_contract.gate_proj,
            )
            _assert_projection(
                model,
                weight_map,
                prefix,
                "up_proj",
                mlp_contract.up_proj,
            )
            _assert_projection(
                model,
                weight_map,
                prefix,
                "down_proj",
                mlp_contract.down_proj,
            )


def test_local_kimi_mxfp4_last_moe_layer_has_readable_routed_slice() -> None:
    model = _require_local_kimi_mxfp4_model()
    model_config = _model_config(model)
    weight_map = _weight_map(model)
    contract = build_kimi_mxfp4_sharding_contract(
        model_config,
        tp_size=4,
        ep_size=4,
        tp_rank=3,
        ep_rank=3,
    )

    assert contract.routed_ownership.owns_expert(383)
    prefix = "language_model.model.layers.60.mlp.experts.383"
    _assert_projection(model, weight_map, prefix, "gate_proj", contract.routed.gate_proj)
    _assert_projection(model, weight_map, prefix, "up_proj", contract.routed.up_proj)
    _assert_projection(model, weight_map, prefix, "down_proj", contract.routed.down_proj)


def test_local_kimi_mxfp4_bf16_excluded_tensors_are_not_mxfp4() -> None:
    model = _require_local_kimi_mxfp4_model()
    weight_map = _weight_map(model)
    summary = summarize_kimi_mxfp4_artifact(model)

    assert summary.is_quark_mxfp4_dynamic_fp4 is True
    assert summary.excludes_attention is True
    assert summary.excludes_lm_head is True
    assert summary.excludes_mlp_gate is True

    bf16_tensors = {
        "language_model.model.layers.1.self_attn.q_a_proj.weight": (1536, 7168),
        "language_model.model.layers.1.mlp.gate.weight": (384, 7168),
        "language_model.lm_head.weight": (163840, 7168),
    }
    for name, shape in bf16_tensors.items():
        _assert_tensor(
            model,
            weight_map,
            name,
            shape=shape,
            safetensors_dtype="BF16",
        )
        _assert_readable_2d_slice(
            model,
            weight_map,
            name,
            ((0, 1), (0, 8)),
            dtype=torch.bfloat16,
        )
