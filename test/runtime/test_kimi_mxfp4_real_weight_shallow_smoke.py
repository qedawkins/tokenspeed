"""4-rank shallow smoke for real local Kimi MXFP4 checkpoint weights."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tokenspeed.runtime.kimi_quantization_preflight import (
    KimiMxfp4MlpShardContract,
    KimiMxfp4ProjectionShard,
    build_kimi_mxfp4_sharding_contract,
)


_LOCAL_KIMI_MXFP4_MODEL = Path(
    "/data/models/hf/hub/models--amd--Kimi-K2.5-MXFP4/"
    "snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a"
)


def _require_local_model() -> Path:
    if not _LOCAL_KIMI_MXFP4_MODEL.is_dir():
        pytest.skip("local Kimi-K2.5 MXFP4 artifact is not available")
    return _LOCAL_KIMI_MXFP4_MODEL


def _require_four_rank_cdna4_gpu() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for the real-weight Kimi MXFP4 smoke")
    if torch.cuda.device_count() < 4:
        pytest.skip(
            "four visible ROCm devices are required for the real-weight Kimi "
            f"MXFP4 smoke, got {torch.cuda.device_count()}"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("run this smoke with torchrun --nproc_per_node=4")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        pytest.skip(f"expected 4 distributed ranks, got {world_size}")

    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Kimi MXFP4 smoke")
    return rank, world_size, torch.device("cuda", local_rank)


def _model_config(model: Path) -> dict:
    with (model / "config.json").open() as f:
        return json.load(f)


def _weight_map(model: Path) -> dict[str, str]:
    with (model / "model.safetensors.index.json").open() as f:
        index = json.load(f)
    return index["weight_map"]


class _SliceReader:
    def __init__(self, model: Path, weight_map: dict[str, str]):
        self._model = model
        self._weight_map = weight_map
        self._stack = ExitStack()
        self._handles = {}
        self._safe_open = pytest.importorskip("safetensors").safe_open

    def __enter__(self) -> "_SliceReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self._stack.close()

    def _handle(self, name: str):
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._stack.enter_context(
                self._safe_open(self._model / shard, framework="pt", device="cpu")
            )
            self._handles[shard] = handle
        return handle

    def read_2d(
        self,
        name: str,
        row_slice: tuple[int, int],
        col_slice: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        tensor_slice = self._handle(name).get_slice(name)
        rows = slice(*row_slice)
        cols = slice(*col_slice)
        return tensor_slice[rows, cols].contiguous().to(device)


def _record_kernel_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    import tokenspeed_kernel

    kernel_calls: dict[str, list[dict]] = {
        "dispatch": [],
        "combine": [],
        "quantize_mxfp4": [],
    }
    original_dispatch = tokenspeed_kernel.moe_dispatch
    original_combine = tokenspeed_kernel.moe_combine
    original_quantize_mxfp4 = tokenspeed_kernel.quantize_mxfp4

    def _record_dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        kernel_calls["dispatch"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_combine(*args, **kwargs):
        result = original_combine(*args, **kwargs)
        kernel_calls["combine"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_quantize_mxfp4(*args, **kwargs):
        result = original_quantize_mxfp4(*args, **kwargs)
        kernel_calls["quantize_mxfp4"].append({"scale_layout": kwargs.get("scale_layout")})
        return result

    def _forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("real-weight MXFP4 smoke used an unsupported MoE fallback")

    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", _record_dispatch)
    monkeypatch.setattr(tokenspeed_kernel, "moe_combine", _record_combine)
    monkeypatch.setattr(tokenspeed_kernel, "quantize_mxfp4", _record_quantize_mxfp4)
    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _forbidden_kernel)
    monkeypatch.setattr(tokenspeed_kernel, "moe_experts", _forbidden_kernel)
    if hasattr(tokenspeed_kernel, "quantize_fp8"):
        monkeypatch.setattr(tokenspeed_kernel, "quantize_fp8", _forbidden_kernel)
    return kernel_calls


def _make_real_moe_layer(rank: int, world_size: int, device: torch.device):
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.layer import MoELayer
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Mxfp4Config
    from tokenspeed.runtime.utils.env import global_server_args_dict

    previous_backend = getattr(moe_utils, "MOE_BACKEND", None)
    moe_utils.MOE_BACKEND = MoeBackend.AUTO
    global_server_args_dict["ep_num_redundant_experts"] = 0
    try:
        layer = MoELayer(
            top_k=8,
            num_experts=384,
            hidden_size=7168,
            intermediate_size=2048,
            quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
            layer_index=1,
            prefix="language_model.model.layers.1.mlp.experts",
            tp_rank=rank,
            tp_size=world_size,
            ep_rank=rank,
            ep_size=world_size,
            activation="silu",
            with_bias=False,
        ).to(device)
    finally:
        moe_utils.MOE_BACKEND = previous_backend
    return layer


def _load_projection_into_fused_routed_weight(
    reader: _SliceReader,
    *,
    prefix: str,
    projection: str,
    contract: KimiMxfp4ProjectionShard,
    local_expert_id: int,
    fused_weight: torch.Tensor,
    fused_scale: torch.Tensor,
    device: torch.device,
) -> None:
    weight_name = f"{prefix}.{projection}.weight"
    scale_name = f"{prefix}.{projection}.weight_scale"
    weight_rows, weight_cols = contract.checkpoint_weight_slice
    scale_rows, scale_cols = contract.checkpoint_scale_slice
    dst_weight_rows, dst_weight_cols = contract.destination_weight_slice
    dst_scale_rows, dst_scale_cols = contract.destination_scale_slice
    fused_weight[
        local_expert_id,
        slice(*dst_weight_rows),
        slice(*dst_weight_cols),
    ].copy_(reader.read_2d(weight_name, weight_rows, weight_cols, device=device))
    fused_scale[
        local_expert_id,
        slice(*dst_scale_rows),
        slice(*dst_scale_cols),
    ].copy_(reader.read_2d(scale_name, scale_rows, scale_cols, device=device))


def _load_real_routed_weights(
    layer,
    reader: _SliceReader,
    contract: KimiMxfp4MlpShardContract,
    *,
    device: torch.device,
) -> None:
    ownership = contract.num_rank_local_experts
    assert ownership == layer.num_local_experts
    expert_start = layer.backend.spec.ep_rank * layer.num_local_experts
    for local_expert_id in range(layer.num_local_experts):
        global_expert_id = expert_start + local_expert_id
        prefix = f"language_model.model.layers.1.mlp.experts.{global_expert_id}"
        _load_projection_into_fused_routed_weight(
            reader,
            prefix=prefix,
            projection="gate_proj",
            contract=contract.gate_proj,
            local_expert_id=local_expert_id,
            fused_weight=layer.w13_weight.data,
            fused_scale=layer.w13_weight_scale.data,
            device=device,
        )
        _load_projection_into_fused_routed_weight(
            reader,
            prefix=prefix,
            projection="up_proj",
            contract=contract.up_proj,
            local_expert_id=local_expert_id,
            fused_weight=layer.w13_weight.data,
            fused_scale=layer.w13_weight_scale.data,
            device=device,
        )
        _load_projection_into_fused_routed_weight(
            reader,
            prefix=prefix,
            projection="down_proj",
            contract=contract.down_proj,
            local_expert_id=local_expert_id,
            fused_weight=layer.w2_weight.data,
            fused_scale=layer.w2_weight_scale.data,
            device=device,
        )


def _load_linear_projection(
    reader: _SliceReader,
    prefix: str,
    projection: str,
    contract: KimiMxfp4ProjectionShard,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_name = f"{prefix}.{projection}.weight"
    scale_name = f"{prefix}.{projection}.weight_scale"
    return (
        reader.read_2d(
            weight_name,
            contract.checkpoint_weight_slice[0],
            contract.checkpoint_weight_slice[1],
            device=device,
        ),
        reader.read_2d(
            scale_name,
            contract.checkpoint_scale_slice[0],
            contract.checkpoint_scale_slice[1],
            device=device,
        ),
    )


def _run_real_mlp_projection(
    reader: _SliceReader,
    *,
    prefix: str,
    contract: KimiMxfp4MlpShardContract,
    hidden_states: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    from tokenspeed.runtime.layers.dense.mxfp4 import dequantize_mxfp4_linear_weight
    from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
        kimi_swiglu_gate_up,
    )

    gate_packed, gate_scale = _load_linear_projection(
        reader,
        prefix,
        "gate_proj",
        contract.gate_proj,
        device=device,
    )
    up_packed, up_scale = _load_linear_projection(
        reader,
        prefix,
        "up_proj",
        contract.up_proj,
        device=device,
    )
    down_packed, down_scale = _load_linear_projection(
        reader,
        prefix,
        "down_proj",
        contract.down_proj,
        device=device,
    )

    gate_weight = dequantize_mxfp4_linear_weight(
        gate_packed,
        gate_scale,
        output_dtype=torch.bfloat16,
    )
    up_weight = dequantize_mxfp4_linear_weight(
        up_packed,
        up_scale,
        output_dtype=torch.bfloat16,
    )
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    del gate_weight, up_weight, gate_packed, gate_scale, up_packed, up_scale

    gate_up = torch.empty(
        (*gate.shape[:-1], gate.shape[-1] * 2),
        dtype=gate.dtype,
        device=device,
    )
    gate_up[..., 0::2] = gate
    gate_up[..., 1::2] = up
    activated = kimi_swiglu_gate_up(gate_up, output_dtype=torch.bfloat16)
    del gate, up, gate_up

    down_weight = dequantize_mxfp4_linear_weight(
        down_packed,
        down_scale,
        output_dtype=torch.bfloat16,
    )
    output = F.linear(activated, down_weight)
    del down_weight, down_packed, down_scale, activated
    assert torch.isfinite(output).all()
    assert float(output.abs().max().item()) > 0.0
    return output


def _router_logits(num_tokens: int, rank: int, device: torch.device) -> torch.Tensor:
    experts = torch.tensor(
        [0, 96, 192, 288, 95, 191, 287, 383, 1, 97, 193, 289],
        dtype=torch.long,
        device=device,
    ).roll(shifts=rank)
    logits = torch.full((num_tokens, 384), -12.0, dtype=torch.float32, device=device)
    descending = torch.linspace(10.0, -2.0, steps=experts.numel(), device=device)
    for token in range(num_tokens):
        logits[token, experts.roll(shifts=token)] = descending
    return logits.to(torch.bfloat16)


def _run_real_routed_case(
    layer,
    *,
    rank: int,
    world_size: int,
    kernel_calls: dict[str, list[dict]],
) -> torch.Tensor:
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig

    torch.manual_seed(19019 + rank)
    hidden_states = (
        torch.randn(1, layer.hidden_size, device=layer.w13_weight.device) * 0.002
    ).bfloat16()
    correction_bias = torch.zeros(layer.num_experts, device=hidden_states.device)
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=_router_logits(hidden_states.shape[0], rank, hidden_states.device),
        topk_config=TopKConfig(
            top_k=8,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            renormalize=True,
            correction_bias=correction_bias,
            routed_scaling_factor=2.827,
            apply_routed_scaling_factor_on_output=True,
        ),
    )
    output = layer(
        hidden_states,
        topk_output,
        num_global_tokens=hidden_states.shape[0] * world_size,
        max_num_tokens_per_gpu=hidden_states.shape[0],
    )
    torch.cuda.synchronize()

    assert output.shape == hidden_states.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert float(output.abs().max().item()) > 0.0
    assert any(
        call.get("expected_kernel_name") == "gluon_ep_metadata_gfx950"
        for call in kernel_calls["dispatch"]
    )
    assert any(
        call.get("expected_kernel_name") == "gluon_local_sum_reduce_gfx950"
        for call in kernel_calls["combine"]
    )
    assert len(kernel_calls["quantize_mxfp4"]) >= 2
    return output


def test_local_kimi_mxfp4_real_weight_shallow_tp_ep_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("iris")
    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    torch.cuda.reset_peak_memory_stats(device)

    model_config = _model_config(model)
    weight_map = _weight_map(model)
    contract = build_kimi_mxfp4_sharding_contract(
        model_config,
        tp_size=world_size,
        ep_size=world_size,
        tp_rank=rank,
        ep_rank=rank,
    )
    hidden_states = (
        torch.randn(1, contract.hidden_size, device=device) * 0.002
    ).bfloat16()

    with _SliceReader(model, weight_map) as reader:
        dense_output = _run_real_mlp_projection(
            reader,
            prefix="language_model.model.layers.0.mlp",
            contract=contract.dense,
            hidden_states=hidden_states,
            device=device,
        )
        shared_output = _run_real_mlp_projection(
            reader,
            prefix="language_model.model.layers.1.mlp.shared_experts",
            contract=contract.shared,
            hidden_states=hidden_states,
            device=device,
        )
        kernel_calls = _record_kernel_calls(monkeypatch)
        layer = _make_real_moe_layer(rank, world_size, device)
        _load_real_routed_weights(layer, reader, contract.routed, device=device)
        layer.process_weights_after_loading(layer)

    assert dense_output.shape == (1, contract.hidden_size)
    assert shared_output.shape == (1, contract.hidden_size)
    assert type(layer.backend).__name__ == "Mxfp4TritonKernelEPBackend"
    assert layer.backend.key.quant == "mxfp4"
    assert layer.backend.key.impl == "triton_kernel_ep"
    assert layer.expert_weight_format_signature.name == "mxfp4_e2m1_block32"
    assert layer.topk_output_format.is_bypassed()
    assert tuple(layer.w13_weight.shape) == contract.routed.rank_local_w13_weight_shape
    assert tuple(layer.w13_weight_scale.shape) == contract.routed.rank_local_w13_scale_shape
    assert tuple(layer.w2_weight.shape) == contract.routed.rank_local_w2_weight_shape
    assert tuple(layer.w2_weight_scale.shape) == contract.routed.rank_local_w2_scale_shape

    routed_output = _run_real_routed_case(
        layer,
        rank=rank,
        world_size=world_size,
        kernel_calls=kernel_calls,
    )
    workspace = layer.backend._ep_workspace
    assert workspace is not None
    assert workspace.backend == "iris"
    assert workspace.world_size == world_size
    assert workspace.rank == rank
    print(
        "rank=",
        rank,
        "dense_max=",
        float(dense_output.abs().max().item()),
        "shared_max=",
        float(shared_output.abs().max().item()),
        "routed_max=",
        float(routed_output.abs().max().item()),
        "max_memory_allocated=",
        torch.cuda.max_memory_allocated(device),
    )
    dist.barrier()
