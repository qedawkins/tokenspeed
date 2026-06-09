"""CPU-only Kimi MXFP4 routed expert checkpoint loading tests."""

from __future__ import annotations

from functools import partial
from math import prod

import pytest
import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.weight_loaders import load_model_weight
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)


_NUM_KIMI_EXPERTS = 384
_KIMI_EP_SIZE = 4
_KIMI_EP_RANK = 2
_KIMI_TP_SIZE = 4
_KIMI_TP_RANK = 2
_LOCAL_EXPERTS = _NUM_KIMI_EXPERTS // _KIMI_EP_SIZE
_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_LOCAL_INTERMEDIATE = _INTERMEDIATE_SIZE // _KIMI_TP_SIZE


def _uint8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    return ((values + offset) % 251).to(torch.uint8)


def _maybe_e8m0(raw_bytes: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        return raw_bytes
    return raw_bytes.view(e8m0_dtype)


def _make_mxfp4_loader_layer() -> nn.Module:
    layer = nn.Module()
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(
            torch.zeros(
                _LOCAL_EXPERTS,
                2 * _LOCAL_INTERMEDIATE,
                _HIDDEN_SIZE // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        nn.Parameter(
            torch.zeros(
                _LOCAL_EXPERTS,
                2 * _LOCAL_INTERMEDIATE,
                _HIDDEN_SIZE // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        nn.Parameter(
            torch.zeros(
                _LOCAL_EXPERTS,
                _HIDDEN_SIZE,
                _LOCAL_INTERMEDIATE // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        nn.Parameter(
            torch.zeros(
                _LOCAL_EXPERTS,
                _HIDDEN_SIZE,
                _LOCAL_INTERMEDIATE // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )

    weight_loader = partial(
        load_model_weight,
        tp_rank=_KIMI_TP_RANK,
        is_bias=False,
        use_presharded_weights=False,
        do_transpose=False,
    )
    layer.w13_weight.weight_loader = weight_loader
    layer.w13_weight_scale.weight_loader = weight_loader
    layer.w2_weight.weight_loader = weight_loader
    layer.w2_weight_scale.weight_loader = weight_loader
    return layer


def _params_dict(layer: nn.Module) -> dict[str, nn.Parameter]:
    prefix = "model.layers.1.mlp.experts."
    return {f"{prefix}{name}": param for name, param in layer.named_parameters()}


def _loader(layer: nn.Module):
    return build_moe_checkpoint_loader(
        params_dict=_params_dict(layer),
        expert_schema=ExpertCheckpointSchema(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
        ),
        num_experts=_NUM_KIMI_EXPERTS,
        ep_rank=_KIMI_EP_RANK,
        ep_size=_KIMI_EP_SIZE,
    )


def _load_synthetic_expert(loader, global_expert_id: int, offset: int):
    prefix = f"model.layers.1.mlp.experts.{global_expert_id}"
    tensors = {
        "gate_proj.weight": _uint8_pattern(
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE // 2),
            offset,
        ),
        "up_proj.weight": _uint8_pattern(
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE // 2),
            offset + 17,
        ),
        "down_proj.weight": _uint8_pattern(
            (_HIDDEN_SIZE, _INTERMEDIATE_SIZE // 2),
            offset + 31,
        ),
        "gate_proj.weight_scale": _uint8_pattern(
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE // 32),
            offset + 47,
        ),
        "up_proj.weight_scale": _uint8_pattern(
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE // 32),
            offset + 61,
        ),
        "down_proj.weight_scale": _uint8_pattern(
            (_HIDDEN_SIZE, _INTERMEDIATE_SIZE // 32),
            offset + 79,
        ),
    }
    for suffix, tensor in tensors.items():
        loader.load(
            f"{prefix}.{suffix}",
            _maybe_e8m0(tensor) if suffix.endswith("weight_scale") else tensor,
        )
    return tensors


def _assert_expert_loaded(
    layer: nn.Module,
    originals: dict[str, torch.Tensor],
    *,
    local_expert_id: int,
) -> None:
    row_start = _LOCAL_INTERMEDIATE * _KIMI_TP_RANK
    row_end = row_start + _LOCAL_INTERMEDIATE
    down_weight_col_start = (_LOCAL_INTERMEDIATE // 2) * _KIMI_TP_RANK
    down_weight_col_end = down_weight_col_start + (_LOCAL_INTERMEDIATE // 2)
    down_scale_col_start = (_LOCAL_INTERMEDIATE // 32) * _KIMI_TP_RANK
    down_scale_col_end = down_scale_col_start + (_LOCAL_INTERMEDIATE // 32)

    assert torch.equal(
        layer.w13_weight[local_expert_id, :_LOCAL_INTERMEDIATE],
        originals["gate_proj.weight"][row_start:row_end],
    )
    assert torch.equal(
        layer.w13_weight[local_expert_id, _LOCAL_INTERMEDIATE:],
        originals["up_proj.weight"][row_start:row_end],
    )
    assert torch.equal(
        layer.w2_weight[local_expert_id],
        originals["down_proj.weight"][:, down_weight_col_start:down_weight_col_end],
    )
    assert torch.equal(
        layer.w13_weight_scale[local_expert_id, :_LOCAL_INTERMEDIATE],
        originals["gate_proj.weight_scale"][row_start:row_end],
    )
    assert torch.equal(
        layer.w13_weight_scale[local_expert_id, _LOCAL_INTERMEDIATE:],
        originals["up_proj.weight_scale"][row_start:row_end],
    )
    assert torch.equal(
        layer.w2_weight_scale[local_expert_id],
        originals["down_proj.weight_scale"][
            :,
            down_scale_col_start:down_scale_col_end,
        ],
    )


def test_kimi_mxfp4_loader_loads_96_owned_experts_with_tp_slices() -> None:
    layer = _make_mxfp4_loader_layer()
    loader = _loader(layer)

    assert layer.w13_weight.shape == (96, 64, 32)
    assert layer.w13_weight_scale.shape == (96, 64, 2)
    assert layer.w2_weight.shape == (96, 64, 16)
    assert layer.w2_weight_scale.shape == (96, 64, 1)

    assert not loader.matches("model.layers.1.mlp.experts.191.gate_proj.weight")
    assert loader.matches("model.layers.1.mlp.experts.192.gate_proj.weight")
    assert loader.matches("model.layers.1.mlp.experts.287.down_proj.weight_scale")
    assert not loader.matches("model.layers.1.mlp.experts.288.up_proj.weight")

    first = _load_synthetic_expert(loader, 192, offset=11)
    last = _load_synthetic_expert(loader, 287, offset=103)

    _assert_expert_loaded(layer, first, local_expert_id=0)
    _assert_expert_loaded(layer, last, local_expert_id=95)
    assert torch.count_nonzero(layer.w13_weight[1:95]) == 0
    assert torch.count_nonzero(layer.w13_weight_scale[1:95]) == 0
    assert torch.count_nonzero(layer.w2_weight[1:95]) == 0
    assert torch.count_nonzero(layer.w2_weight_scale[1:95]) == 0


def test_kimi_mxfp4_loader_rejects_invalid_ep_layout() -> None:
    layer = _make_mxfp4_loader_layer()
    with pytest.raises(ValueError, match="num_experts"):
        build_moe_checkpoint_loader(
            params_dict=_params_dict(layer),
            expert_schema=ExpertCheckpointSchema(),
            num_experts=385,
            ep_rank=0,
            ep_size=4,
        )
    with pytest.raises(ValueError, match="ep_rank"):
        build_moe_checkpoint_loader(
            params_dict=_params_dict(layer),
            expert_schema=ExpertCheckpointSchema(),
            num_experts=384,
            ep_rank=4,
            ep_size=4,
        )
