"""CPU-only Kimi MXFP4 TP/EP sharding contract tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tokenspeed.runtime.kimi_quantization_preflight import (
    build_kimi_mxfp4_sharding_contract,
)


def _contract(*, tp_rank: int = 0, ep_rank: int = 0):
    return build_kimi_mxfp4_sharding_contract(
        quark_kimi_mxfp4_model_config(),
        tp_size=4,
        ep_size=4,
        tp_rank=tp_rank,
        ep_rank=ep_rank,
    )


def _fp4_mx_per_group(*, is_dynamic: bool) -> dict:
    return {
        "dtype": "fp4",
        "is_dynamic": is_dynamic,
        "qscheme": "per_group",
        "ch_axis": -1,
        "group_size": 32,
        "scale_format": "e8m0",
    }


def quark_kimi_mxfp4_model_config() -> dict:
    return {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "dtype": "bfloat16",
        "text_config": {
            "architectures": ["DeepseekV3ForCausalLM"],
            "first_k_dense_replace": 1,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "model_type": "kimi_k2",
            "moe_intermediate_size": 2048,
            "n_group": 1,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "norm_topk_prob": True,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 61,
            "routed_scaling_factor": 2.827,
            "scoring_func": "sigmoid",
            "topk_group": 1,
            "topk_method": "noaux_tc",
        },
        "quantization_config": {
            "global_quant_config": {
                "input_tensors": _fp4_mx_per_group(is_dynamic=True),
                "output_tensors": None,
                "weight": _fp4_mx_per_group(is_dynamic=False),
            },
            "quant_method": "quark",
            "export": {"pack_method": "reorder", "weight_format": "real_quantized"},
            "exclude": [
                "re:lm_head",
                "re:.*self_attn.*",
                "re:.*mlp.gate",
            ],
        },
    }


def non_mxfp4_model_config() -> dict:
    config = quark_kimi_mxfp4_model_config()
    config["quantization_config"] = {
        "quant_method": "quark",
        "global_quant_config": {
            "input_tensors": {"dtype": "fp8_e4m3"},
            "weight": {"dtype": "fp8_e4m3"},
        },
    }
    return config


def test_kimi_mxfp4_routed_contract_matches_4gpu_tp_ep_shapes() -> None:
    contract = _contract()

    assert contract.hidden_size == 7168
    assert contract.moe_intermediate_size == 2048
    assert contract.dense_intermediate_size == 18432
    assert contract.pack_factor == 2
    assert contract.scale_block == 32
    assert contract.bf16_passthrough_patterns == (
        "self_attn",
        "mlp.gate",
        "lm_head",
    )
    assert contract.routed_ownership.experts_per_ep_rank == 96
    assert contract.routed_ownership.global_expert_start == 0
    assert contract.routed_ownership.global_expert_end == 96
    assert contract.routed.uses_ep_ownership is True
    assert contract.routed.num_rank_local_experts == 96

    assert contract.routed.rank_local_w13_weight_shape == (96, 1024, 3584)
    assert contract.routed.rank_local_w13_scale_shape == (96, 1024, 224)
    assert contract.routed.rank_local_w2_weight_shape == (96, 7168, 256)
    assert contract.routed.rank_local_w2_scale_shape == (96, 7168, 16)

    assert contract.routed.gate_proj.logical_shape == (2048, 7168)
    assert contract.routed.gate_proj.checkpoint_weight_shape == (2048, 3584)
    assert contract.routed.gate_proj.checkpoint_scale_shape == (2048, 224)
    assert contract.routed.down_proj.logical_shape == (7168, 2048)
    assert contract.routed.down_proj.checkpoint_weight_shape == (7168, 1024)
    assert contract.routed.down_proj.checkpoint_scale_shape == (7168, 64)


@pytest.mark.parametrize(
    ("tp_rank", "gate_rows", "down_weight_cols", "down_scale_cols"),
    [
        (0, (0, 512), (0, 256), (0, 16)),
        (1, (512, 1024), (256, 512), (16, 32)),
        (2, (1024, 1536), (512, 768), (32, 48)),
        (3, (1536, 2048), (768, 1024), (48, 64)),
    ],
)
def test_kimi_mxfp4_routed_tp_slices_are_exact(
    tp_rank: int,
    gate_rows: tuple[int, int],
    down_weight_cols: tuple[int, int],
    down_scale_cols: tuple[int, int],
) -> None:
    contract = _contract(tp_rank=tp_rank)

    assert contract.routed.gate_proj.checkpoint_weight_slice == (
        gate_rows,
        (0, 3584),
    )
    assert contract.routed.gate_proj.checkpoint_scale_slice == (
        gate_rows,
        (0, 224),
    )
    assert contract.routed.gate_proj.destination_weight_slice == (
        (0, 512),
        (0, 3584),
    )
    assert contract.routed.up_proj.checkpoint_weight_slice == (
        gate_rows,
        (0, 3584),
    )
    assert contract.routed.up_proj.destination_weight_slice == (
        (512, 1024),
        (0, 3584),
    )
    assert contract.routed.down_proj.checkpoint_weight_slice == (
        (0, 7168),
        down_weight_cols,
    )
    assert contract.routed.down_proj.checkpoint_scale_slice == (
        (0, 7168),
        down_scale_cols,
    )


@pytest.mark.parametrize(
    ("ep_rank", "start", "end"),
    [
        (0, 0, 96),
        (1, 96, 192),
        (2, 192, 288),
        (3, 288, 384),
    ],
)
def test_kimi_mxfp4_expert_ownership_is_uniform(
    ep_rank: int,
    start: int,
    end: int,
) -> None:
    contract = _contract(ep_rank=ep_rank)
    ownership = contract.routed_ownership

    assert ownership.global_expert_start == start
    assert ownership.global_expert_end == end
    assert ownership.owner_rank(start) == ep_rank
    assert ownership.local_expert_id(start) == 0
    assert ownership.owner_rank(end - 1) == ep_rank
    assert ownership.local_expert_id(end - 1) == 95
    assert list(ownership.owned_expert_ids()) == list(range(start, end))


def test_kimi_mxfp4_expert_ownership_rejects_invalid_experts() -> None:
    ownership = _contract().routed_ownership

    with pytest.raises(ValueError, match="outside"):
        ownership.owner_rank(-1)
    with pytest.raises(ValueError, match="outside"):
        ownership.local_expert_id(384)


def test_kimi_mxfp4_shared_and_dense_contracts_are_tp_only() -> None:
    contract = _contract(tp_rank=2, ep_rank=3)

    assert contract.shared.uses_ep_ownership is False
    assert contract.shared.num_rank_local_experts is None
    assert contract.shared.rank_local_w13_weight_shape == (1024, 3584)
    assert contract.shared.rank_local_w13_scale_shape == (1024, 224)
    assert contract.shared.rank_local_w2_weight_shape == (7168, 256)
    assert contract.shared.rank_local_w2_scale_shape == (7168, 16)
    assert contract.shared.gate_proj.checkpoint_weight_shape == (2048, 3584)
    assert contract.shared.down_proj.checkpoint_scale_shape == (7168, 64)

    assert contract.dense.uses_ep_ownership is False
    assert contract.dense.num_rank_local_experts is None
    assert contract.dense.rank_local_w13_weight_shape == (9216, 3584)
    assert contract.dense.rank_local_w13_scale_shape == (9216, 224)
    assert contract.dense.rank_local_w2_weight_shape == (7168, 2304)
    assert contract.dense.rank_local_w2_scale_shape == (7168, 144)
    assert contract.dense.gate_proj.checkpoint_weight_shape == (18432, 3584)
    assert contract.dense.gate_proj.checkpoint_scale_shape == (18432, 224)
    assert contract.dense.down_proj.checkpoint_weight_shape == (7168, 9216)
    assert contract.dense.down_proj.checkpoint_scale_shape == (7168, 576)
    assert contract.dense.gate_proj.checkpoint_weight_slice == (
        (9216, 13824),
        (0, 3584),
    )
    assert contract.dense.up_proj.destination_weight_slice == (
        (4608, 9216),
        (0, 3584),
    )
    assert contract.dense.down_proj.checkpoint_weight_slice == (
        (0, 7168),
        (4608, 6912),
    )
    assert contract.dense.down_proj.checkpoint_scale_slice == (
        (0, 7168),
        (288, 432),
    )


def test_kimi_mxfp4_routed_contract_supports_ep_only_moe_tp() -> None:
    contract = build_kimi_mxfp4_sharding_contract(
        quark_kimi_mxfp4_model_config(),
        tp_size=4,
        ep_size=4,
        tp_rank=3,
        ep_rank=3,
        moe_tp_size=1,
        moe_tp_rank=0,
    )

    assert contract.tp_size == 4
    assert contract.tp_rank == 3
    assert contract.routed_tp_size == 1
    assert contract.routed_tp_rank == 0
    assert contract.routed_ownership.global_expert_start == 288
    assert contract.routed_ownership.global_expert_end == 384

    assert contract.routed.rank_local_w13_weight_shape == (96, 4096, 3584)
    assert contract.routed.rank_local_w13_scale_shape == (96, 4096, 224)
    assert contract.routed.rank_local_w2_weight_shape == (96, 7168, 1024)
    assert contract.routed.rank_local_w2_scale_shape == (96, 7168, 64)
    assert contract.routed.gate_proj.checkpoint_weight_slice == (
        (0, 2048),
        (0, 3584),
    )
    assert contract.routed.up_proj.destination_weight_slice == (
        (2048, 4096),
        (0, 3584),
    )
    assert contract.routed.down_proj.checkpoint_weight_slice == (
        (0, 7168),
        (0, 1024),
    )
    assert contract.routed.down_proj.checkpoint_scale_slice == (
        (0, 7168),
        (0, 64),
    )

    assert contract.shared.rank_local_w13_weight_shape == (1024, 3584)
    assert contract.dense.rank_local_w13_weight_shape == (9216, 3584)


def test_kimi_mxfp4_contract_rejects_non_mxfp4_metadata() -> None:
    with pytest.raises(ValueError, match="Quark dynamic-FP4 MXFP4"):
        build_kimi_mxfp4_sharding_contract(non_mxfp4_model_config())


def test_kimi_mxfp4_contract_rejects_invalid_parallel_shapes() -> None:
    bad_experts = deepcopy(quark_kimi_mxfp4_model_config())
    bad_experts["text_config"]["n_routed_experts"] = 383
    with pytest.raises(ValueError, match="routed experts"):
        build_kimi_mxfp4_sharding_contract(bad_experts)

    bad_moe = deepcopy(quark_kimi_mxfp4_model_config())
    bad_moe["text_config"]["moe_intermediate_size"] = 2050
    with pytest.raises(ValueError, match="MoE intermediate"):
        build_kimi_mxfp4_sharding_contract(bad_moe)

    bad_hidden = deepcopy(quark_kimi_mxfp4_model_config())
    bad_hidden["text_config"]["hidden_size"] = 7170
    with pytest.raises(ValueError, match="scale block"):
        build_kimi_mxfp4_sharding_contract(bad_hidden)


def test_kimi_mxfp4_contract_rejects_invalid_ranks() -> None:
    with pytest.raises(ValueError, match="TP rank"):
        build_kimi_mxfp4_sharding_contract(
            quark_kimi_mxfp4_model_config(),
            tp_rank=4,
        )
    with pytest.raises(ValueError, match="EP rank"):
        build_kimi_mxfp4_sharding_contract(
            quark_kimi_mxfp4_model_config(),
            ep_rank=4,
        )
    with pytest.raises(ValueError, match="MoE TP rank"):
        build_kimi_mxfp4_sharding_contract(
            quark_kimi_mxfp4_model_config(),
            moe_tp_size=1,
            moe_tp_rank=1,
        )
