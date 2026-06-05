"""Small Kimi-K2.5 Quark metadata snippets for no-weight tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _fp8_e4m3_per_tensor_dynamic() -> dict[str, Any]:
    return {
        "dtype": "fp8_e4m3",
        "is_dynamic": True,
        "qscheme": "per_tensor",
        "ch_axis": None,
        "group_size": None,
        "symmetric": True,
        "round_method": "half_even",
        "scale_type": "float",
        "scale_format": None,
        "scale_calculation_mode": None,
        "mx_element_dtype": None,
        "observer_cls": "PerTensorMinMaxObserver",
        "is_scale_quant": False,
    }


def _fp8_e4m3_per_channel_static() -> dict[str, Any]:
    return {
        "dtype": "fp8_e4m3",
        "is_dynamic": False,
        "qscheme": "per_channel",
        "ch_axis": 0,
        "group_size": None,
        "symmetric": True,
        "round_method": "half_even",
        "scale_type": "float",
        "scale_format": None,
        "scale_calculation_mode": None,
        "mx_element_dtype": None,
        "observer_cls": "PerChannelMinMaxObserver",
        "is_scale_quant": False,
    }


def _fp8_e4m3_per_tensor_static() -> dict[str, Any]:
    return {
        "dtype": "fp8_e4m3",
        "is_dynamic": False,
        "qscheme": "per_tensor",
        "ch_axis": None,
        "group_size": None,
        "symmetric": True,
        "round_method": "half_even",
        "scale_type": "float",
        "scale_format": None,
        "scale_calculation_mode": None,
        "mx_element_dtype": None,
        "observer_cls": "PerTensorMinMaxObserver",
        "is_scale_quant": False,
    }


def _int4_per_channel_static() -> dict[str, Any]:
    return {
        "dtype": "int4",
        "is_dynamic": False,
        "qscheme": "per_channel",
        "ch_axis": 0,
        "group_size": None,
        "symmetric": True,
        "round_method": "half_even",
        "scale_type": "float",
        "scale_format": None,
        "scale_calculation_mode": None,
        "mx_element_dtype": None,
        "observer_cls": "PerChannelMinMaxObserver",
        "is_scale_quant": False,
    }


_KIMI_QUARK_EXCLUDE = [
    "re:lm_head",
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
    "re:.*mm_projector.*",
    "re:.*vision_tower.*",
]


_KIMI_QUARK_W8A8_QUANTIZATION_CONFIG: dict[str, Any] = {
    "global_quant_config": {
        "input_tensors": _fp8_e4m3_per_tensor_dynamic(),
        "output_tensors": None,
        "weight": _fp8_e4m3_per_channel_static(),
        "bias": None,
        "target_device": None,
    },
    "algo_config": None,
    "softmax_quant_spec": None,
    "quant_method": "quark",
    "layer_type_quant_config": {},
    "layer_quant_config": {},
    "kv_cache_quant_config": {},
    "kv_cache_post_rope": False,
    "quant_mode": "eager_mode",
    "version": "0.11",
    "export": {
        "kv_cache_group": [],
        "min_kv_cache": 0.0,
        "pack_method": None,
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
    "exclude": list(_KIMI_QUARK_EXCLUDE),
}


_KIMI_QUARK_W4A8_QUANTIZATION_CONFIG: dict[str, Any] = {
    "global_quant_config": {
        "input_tensors": _fp8_e4m3_per_tensor_dynamic(),
        "output_tensors": None,
        "weight": [
            _fp8_e4m3_per_tensor_static(),
            _int4_per_channel_static(),
        ],
        "bias": None,
        "target_device": None,
    },
    "algo_config": None,
    "softmax_quant_spec": None,
    "quant_method": "quark",
    "layer_type_quant_config": {},
    "layer_quant_config": {},
    "kv_cache_quant_config": {},
    "kv_cache_post_rope": False,
    "quant_mode": "eager_mode",
    "version": "0.11",
    "export": {
        "kv_cache_group": [],
        "min_kv_cache": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
    "exclude": list(_KIMI_QUARK_EXCLUDE),
}


def quark_kimi_w8a8_quantization_config() -> dict[str, Any]:
    return deepcopy(_KIMI_QUARK_W8A8_QUANTIZATION_CONFIG)


def quark_kimi_w4a8_quantization_config() -> dict[str, Any]:
    return deepcopy(_KIMI_QUARK_W4A8_QUANTIZATION_CONFIG)


def quark_kimi_w8a8_model_config() -> dict[str, Any]:
    return _kimi_model_config(quark_kimi_w8a8_quantization_config())


def quark_kimi_w4a8_model_config() -> dict[str, Any]:
    return _kimi_model_config(quark_kimi_w4a8_quantization_config())


def _kimi_model_config(quantization_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "dtype": "bfloat16",
        "text_config": {
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": 7168,
            "model_type": "kimi_k2",
            "moe_intermediate_size": 2048,
            "n_routed_experts": 384,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 61,
        },
        "quantization_config": quantization_config,
    }
