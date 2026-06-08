"""4-rank shallow smoke for real local Kimi MXFP4 checkpoint weights."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

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


def _ensure_runtime_process_group(world_size: int) -> tuple[int, ...]:
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    group = tuple(range(world_size))
    pg_manager.init_process_group(group, backend="nccl")
    return group


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

    def read_tensor(self, name: str, *, device: torch.device) -> torch.Tensor:
        return self._handle(name).get_tensor(name).contiguous().to(device)


def _record_kernel_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    import tokenspeed_kernel

    kernel_calls: dict[str, list[dict]] = {
        "route": [],
        "dispatch": [],
        "combine": [],
        "quantize_mxfp4": [],
    }
    original_route = tokenspeed_kernel.moe_route
    original_dispatch = tokenspeed_kernel.moe_dispatch
    original_combine = tokenspeed_kernel.moe_combine
    original_quantize_mxfp4 = tokenspeed_kernel.quantize_mxfp4

    def _record_route(*args, **kwargs):
        result = original_route(*args, **kwargs)
        kernel_calls["route"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

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
    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _record_route)
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


def _make_real_mlp(
    model_config: dict,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    prefix: str,
    intermediate_size: int,
    is_shared_expert: bool,
):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization import Mxfp4Config
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP

    text_config = model_config["text_config"]
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        mlp = DeepseekV3MLP(
            hidden_size=int(text_config["hidden_size"]),
            intermediate_size=intermediate_size,
            hidden_act=str(text_config["hidden_act"]),
            mapping=Mapping(
                rank=rank,
                world_size=world_size,
                attn_tp_size=world_size,
                dense_tp_size=world_size,
                moe_tp_size=1,
                moe_ep_size=world_size,
            ),
            quant_config=Mxfp4Config.from_config(model_config["quantization_config"]),
            prefix=prefix,
            is_shared_expert=is_shared_expert,
        ).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
    return mlp


def _load_real_mlp_weights(layer, reader: _SliceReader, *, prefix: str, device: torch.device):
    for projection, shard_id in (("gate_proj", 0), ("up_proj", 1)):
        layer.gate_up_proj.weight_loader(
            layer.gate_up_proj.weight,
            reader.read_tensor(f"{prefix}.{projection}.weight", device=device),
            shard_id,
        )
        layer.gate_up_proj.weight_loader(
            layer.gate_up_proj.weight_scale,
            reader.read_tensor(f"{prefix}.{projection}.weight_scale", device=device),
            shard_id,
        )
    layer.down_proj.weight_loader(
        layer.down_proj.weight,
        reader.read_tensor(f"{prefix}.down_proj.weight", device=device),
    )
    layer.down_proj.weight_loader(
        layer.down_proj.weight_scale,
        reader.read_tensor(f"{prefix}.down_proj.weight_scale", device=device),
    )
    layer.gate_up_proj.quant_method.process_weights_after_loading(layer.gate_up_proj)
    layer.down_proj.quant_method.process_weights_after_loading(layer.down_proj)


def _run_loaded_mlp_matches_reference(
    layer,
    expected: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    output = layer(hidden_states)
    assert output.shape == expected.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=0.02,
        rtol=0.02,
        check_dtype=False,
    )
    return output


class _RealMLAKVPool:
    def __init__(
        self,
        device: torch.device,
        *,
        total_slots: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_layers: int = 1,
    ) -> None:
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.kv_buffer = [
            torch.zeros(
                total_slots,
                1,
                self.kv_cache_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(num_layers)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_buffer[layer_id]

    def set_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1).to(torch.bfloat16)
        self.kv_buffer[layer.layer_id][loc.long()] = cache_k


class _RealMLAPrefillBackend:
    spec_num_tokens = 1
    supports_fused_fp8_mla = False
    data_type = torch.bfloat16

    def __init__(self) -> None:
        self.chunked_prefill_metadata = None
        self.prefill_calls = 0

    def init_prefill_metadata(self, seq_lens: torch.Tensor) -> None:
        cum_seq_lens = torch.zeros(
            seq_lens.numel() + 1,
            device=seq_lens.device,
            dtype=torch.int32,
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])
        self.chunked_prefill_metadata = SimpleNamespace(
            extend_seq_lens=seq_lens,
            cum_extend_seq_lens=cum_seq_lens,
            max_extend_seq_len=int(seq_lens.max().item()),
            chunked_loop_num=0,
            chunk_kv_indices_list=[],
            chunked_seq_len=[],
            cu_chunked_seq_len=[],
            max_chunk_len_per_loop=[],
        )

    def forward_extend_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scaling: float,
        logits_soft_cap: float,
        *,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        causal: bool,
        out: torch.Tensor | None = None,
    ):
        del logits_soft_cap
        from tokenspeed_kernel.ops.attention import mla_prefill

        self.prefill_calls += 1
        return mla_prefill(
            q,
            k,
            v,
            seq_lens,
            cum_seq_lens_kv,
            max_kv_len,
            batch_size=batch_size,
            softmax_scale=scaling,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max_q_len,
            out=out,
        )


def _make_gluon_mla_decode_backend(
    attn,
    *,
    device: torch.device,
    seq_len: int,
    page_id: int = 0,
):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.gluon_mla import GluonMLABackend

    backend = GluonMLABackend(
        SimpleNamespace(
            device=device,
            num_attention_heads=attn.num_heads,
            attn_tp_size=attn.mapping.attn.tp_size,
            num_kv_heads=1,
            dtype=torch.bfloat16,
            head_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
            is_draft=False,
            speculative_num_draft_tokens=1,
            context_len=64,
            page_size=64,
            kv_lora_rank=attn.kv_lora_rank,
            qk_nope_head_dim=attn.qk_nope_head_dim,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            v_head_dim=attn.v_head_dim,
            kv_cache_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
            scaling=attn.scaling,
            kv_cache_dtype=torch.bfloat16,
        )
    )
    backend.init_forward_metadata(
        bs=1,
        num_extends=0,
        req_pool_indices=torch.zeros(1, dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        forward_mode=ForwardMode.DECODE,
        req_to_page=torch.full((1, 1), page_id, dtype=torch.int32, device=device),
    )
    return backend


def _make_real_attention(
    model_config: dict,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization import Mxfp4Config
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
    from tokenspeed.runtime.utils.env import global_server_args_dict

    text_config = dict(model_config["text_config"])
    text_config["rope_scaling"] = dict(text_config["rope_scaling"])
    config = SimpleNamespace(**text_config)
    old_dtype = torch.get_default_dtype()
    old_backend = global_server_args_dict.get("attention_backend")
    global_server_args_dict["attention_backend"] = "tokenspeed_mla"
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = DeepseekV3AttentionMLA(
            config=config,
            mapping=Mapping(
                rank=rank,
                world_size=world_size,
                attn_tp_size=world_size,
                dense_tp_size=world_size,
                moe_tp_size=1,
                moe_ep_size=world_size,
            ),
            hidden_size=int(text_config["hidden_size"]),
            num_heads=int(text_config["num_attention_heads"]),
            qk_nope_head_dim=int(text_config["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text_config["qk_rope_head_dim"]),
            v_head_dim=int(text_config["v_head_dim"]),
            q_lora_rank=int(text_config["q_lora_rank"]),
            kv_lora_rank=int(text_config["kv_lora_rank"]),
            rope_theta=float(text_config["rope_theta"]),
            rope_scaling=dict(text_config["rope_scaling"]),
            max_position_embeddings=int(text_config["max_position_embeddings"]),
            quant_config=Mxfp4Config.from_config(model_config["quantization_config"]),
            layer_id=0,
            prefix="model.layers.0.self_attn",
        ).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
        global_server_args_dict["attention_backend"] = old_backend
    return attn


def _load_real_attention_weights(
    attn,
    reader: _SliceReader,
    *,
    prefix: str,
    device: torch.device,
) -> None:
    fused = attn.fused_qkv_a_proj_with_mqa
    fused.weight_loader(
        fused.weight,
        reader.read_tensor(f"{prefix}.q_a_proj.weight", device=device),
        begin_size=0,
    )
    fused.weight_loader(
        fused.weight,
        reader.read_tensor(f"{prefix}.kv_a_proj_with_mqa.weight", device=device),
        begin_size=attn.q_lora_rank,
    )
    attn.q_a_layernorm.weight.data.copy_(
        reader.read_tensor(f"{prefix}.q_a_layernorm.weight", device=device)
    )
    attn.kv_a_layernorm.weight.data.copy_(
        reader.read_tensor(f"{prefix}.kv_a_layernorm.weight", device=device)
    )
    for module_name in ("q_b_proj", "kv_b_proj", "o_proj"):
        module = getattr(attn, module_name)
        module.weight_loader(
            module.weight,
            reader.read_tensor(f"{prefix}.{module_name}.weight", device=device),
        )
    w_kc, w_vc = attn.kv_b_proj.weight.unflatten(
        0,
        (-1, attn.qk_nope_head_dim + attn.v_head_dim),
    ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)
    attn.w_kc = w_kc.contiguous()
    attn.w_vc = w_vc.transpose(1, 2).contiguous()


def _rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _real_attention_reference(
    attn,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    qkv = F.linear(hidden_states, attn.fused_qkv_a_proj_with_mqa.weight)
    q_a, latent_cache = qkv.split(
        [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim],
        dim=-1,
    )
    kv_a = latent_cache[..., : attn.kv_lora_rank]
    k_pe = latent_cache[..., attn.kv_lora_rank :].unsqueeze(1).clone()
    q_norm = _rmsnorm_reference(
        q_a,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.variance_epsilon,
    )
    kv_a_norm = _rmsnorm_reference(
        kv_a,
        attn.kv_a_layernorm.weight,
        attn.kv_a_layernorm.variance_epsilon,
    )

    q = attn.q_b_proj.quant_method.apply(attn.q_b_proj, q_norm).view(
        -1,
        attn.num_local_heads,
        attn.qk_head_dim,
    )
    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    kv = attn.kv_b_proj.quant_method.apply(attn.kv_b_proj, kv_a_norm).view(
        -1,
        attn.num_local_heads,
        attn.qk_nope_head_dim + attn.v_head_dim,
    )
    k_nope = kv[..., : attn.qk_nope_head_dim]
    v = kv[..., attn.qk_nope_head_dim :]

    q_rope = q_pe.clone()
    k_rope = k_pe.clone()
    q_rope, k_rope = attn.rotary_emb(positions, q_rope, k_rope)
    q = torch.cat([q_nope, q_rope], dim=-1)
    k = torch.cat([k_nope, k_rope.expand(-1, attn.num_local_heads, -1)], dim=-1)

    q_heads = q.float().transpose(0, 1)
    k_heads = k.float().transpose(0, 1)
    v_heads = v.float().transpose(0, 1)
    scores = torch.matmul(q_heads, k_heads.transpose(1, 2)) * attn.attn_mha.scaling
    causal_mask = torch.triu(
        torch.ones(
            scores.shape[-2:],
            device=scores.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    context = torch.matmul(probs, v_heads).transpose(0, 1).contiguous()
    context = context.view(hidden_states.shape[0], attn.num_local_heads * attn.v_head_dim)
    partial = F.linear(context.to(attn.o_proj.weight.dtype), attn.o_proj.weight)
    dist.all_reduce(partial)
    return partial.to(hidden_states.dtype)


def _run_real_attention_case(
    attn,
    *,
    rank: int,
    world_size: int,
    hidden_states: torch.Tensor,
) -> None:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    seq_lens = torch.tensor([hidden_states.shape[0]], device=hidden_states.device)
    backend = _RealMLAPrefillBackend()
    backend.init_prefill_metadata(seq_lens)
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=_RealMLAKVPool(
            hidden_states.device,
            total_slots=hidden_states.shape[0],
            kv_lora_rank=attn.kv_lora_rank,
            qk_rope_head_dim=attn.qk_rope_head_dim,
        ),
        bs=1,
        num_extends=1,
        input_num_tokens=hidden_states.shape[0],
        forward_mode=ForwardMode.EXTEND,
    )
    positions = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.long,
    )
    output = attn(
        positions=positions,
        hidden_states=hidden_states,
        ctx=ctx,
        out_cache_loc=torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
            dtype=torch.int32,
        ),
        comm_manager=SimpleNamespace(pre_attn_comm=lambda states, _ctx: states),
    )
    expected = _real_attention_reference(attn, hidden_states, positions)
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=0.08,
        rtol=0.05,
        check_dtype=False,
    )
    assert backend.prefill_calls == 1
    assert output.shape == (hidden_states.shape[0], attn.hidden_size)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    dist.barrier()
    if rank == 0:
        print(
            "attention_max=",
            float(output.abs().max().item()),
            "world_size=",
            world_size,
        )


def _run_real_attention_decode_case(
    attn,
    *,
    rank: int,
    world_size: int,
    prefill_hidden_states: torch.Tensor,
    decode_hidden_states: torch.Tensor,
) -> None:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    seq_len = prefill_hidden_states.shape[0]
    total_slots = 64
    pool = _RealMLAKVPool(
        prefill_hidden_states.device,
        total_slots=total_slots,
        kv_lora_rank=attn.kv_lora_rank,
        qk_rope_head_dim=attn.qk_rope_head_dim,
    )
    prefill_backend = _RealMLAPrefillBackend()
    prefill_backend.init_prefill_metadata(
        torch.tensor([seq_len], device=prefill_hidden_states.device)
    )
    prefill_ctx = ForwardContext(
        attn_backend=prefill_backend,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=1,
        input_num_tokens=seq_len,
        forward_mode=ForwardMode.EXTEND,
    )
    prefill_positions = torch.arange(
        seq_len,
        device=prefill_hidden_states.device,
        dtype=torch.long,
    )
    _ = attn(
        positions=prefill_positions,
        hidden_states=prefill_hidden_states,
        ctx=prefill_ctx,
        out_cache_loc=torch.arange(
            seq_len,
            device=prefill_hidden_states.device,
            dtype=torch.int32,
        ),
        comm_manager=SimpleNamespace(pre_attn_comm=lambda states, _ctx: states),
    )

    decode_backend = _make_gluon_mla_decode_backend(
        attn,
        device=prefill_hidden_states.device,
        seq_len=seq_len + 1,
    )
    decode_ctx = ForwardContext(
        attn_backend=decode_backend,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
    )
    decode_positions = torch.tensor(
        [seq_len],
        device=prefill_hidden_states.device,
        dtype=torch.long,
    )
    output = attn(
        positions=decode_positions,
        hidden_states=decode_hidden_states,
        ctx=decode_ctx,
        out_cache_loc=torch.tensor(
            [seq_len],
            device=prefill_hidden_states.device,
            dtype=torch.int32,
        ),
        comm_manager=SimpleNamespace(pre_attn_comm=lambda states, _ctx: states),
    )

    expected = _real_attention_reference(
        attn,
        torch.cat([prefill_hidden_states, decode_hidden_states], dim=0),
        torch.arange(
            seq_len + 1,
            device=prefill_hidden_states.device,
            dtype=torch.long,
        ),
    )[-1:]
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=0.08,
        rtol=0.05,
        check_dtype=False,
    )
    assert torch.isfinite(output).all()
    dist.barrier()
    if rank == 0:
        print(
            "attention_decode_max=",
            float(output.abs().max().item()),
            "world_size=",
            world_size,
        )


def _load_decoder_layer_norms(layer, reader: _SliceReader, *, prefix: str, device):
    layer.input_layernorm.weight.data.copy_(
        reader.read_tensor(f"{prefix}.input_layernorm.weight", device=device)
    )
    layer.post_attention_layernorm.weight.data.copy_(
        reader.read_tensor(f"{prefix}.post_attention_layernorm.weight", device=device)
    )


def _rmsnorm_add_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed_float = x.float() + residual.float()
    variance = summed_float.pow(2).mean(dim=-1, keepdim=True)
    normed = summed_float * torch.rsqrt(variance + eps) * weight.float()
    return normed.to(x.dtype), summed_float.to(x.dtype)


def _rmsnorm_module_reference(norm, x: torch.Tensor) -> torch.Tensor:
    return norm(x.clone())


def _rmsnorm_add_module_reference(
    norm,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return norm(x.clone(), residual.clone())


def _print_top_abs_diffs(
    *,
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    limit: int = 5,
) -> None:
    diff = (actual.float() - expected.float()).abs().flatten()
    top_values, top_indices = torch.topk(diff, k=min(limit, diff.numel()))
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_index), actual.shape))
        print(
            label,
            "index=",
            index,
            "diff=",
            float(value),
            "actual=",
            float(actual_flat[flat_index].float().item()),
            "expected=",
            float(expected_flat[flat_index].float().item()),
        )


def _all_reduce_clone(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    dist.all_reduce(out)
    return out


def _load_prompt_embeddings(
    reader: _SliceReader,
    input_ids: torch.Tensor,
    *,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = [
        reader.read_2d(
            "language_model.model.embed_tokens.weight",
            (int(token_id), int(token_id) + 1),
            (0, hidden_size),
            device=device,
        )
        for token_id in input_ids.cpu().tolist()
    ]
    return torch.cat(rows, dim=0).to(torch.bfloat16)


def _make_real_decoder_layer(
    model_config: dict,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    layer_id: int,
    enable_allreduce_fusion: bool = False,
):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization import Mxfp4Config
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3DecoderLayer
    from tokenspeed.runtime.utils.env import global_server_args_dict

    text_config = dict(model_config["text_config"])
    text_config["rope_scaling"] = dict(text_config["rope_scaling"])
    config = SimpleNamespace(**text_config)
    old_dtype = torch.get_default_dtype()
    old_attention_backend = global_server_args_dict.get("attention_backend")
    old_enable_allreduce_fusion = global_server_args_dict.get("enable_allreduce_fusion")
    old_comm_fusion_max_num_tokens = global_server_args_dict.get(
        "comm_fusion_max_num_tokens"
    )
    global_server_args_dict["attention_backend"] = "tokenspeed_mla"
    global_server_args_dict["enable_allreduce_fusion"] = enable_allreduce_fusion
    global_server_args_dict["comm_fusion_max_num_tokens"] = 2048
    if not enable_allreduce_fusion:
        global_server_args_dict["comm_fusion_max_num_tokens"] = 0
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = DeepseekV3DecoderLayer(
            config=config,
            layer_id=layer_id,
            mapping=Mapping(
                rank=rank,
                world_size=world_size,
                attn_tp_size=world_size,
                dense_tp_size=world_size,
                moe_tp_size=1,
                moe_ep_size=world_size,
            ),
            quant_config=Mxfp4Config.from_config(model_config["quantization_config"]),
            prefix=f"model.layers.{layer_id}",
        ).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
        global_server_args_dict["attention_backend"] = old_attention_backend
        global_server_args_dict["enable_allreduce_fusion"] = old_enable_allreduce_fusion
        global_server_args_dict["comm_fusion_max_num_tokens"] = (
            old_comm_fusion_max_num_tokens
        )
    return layer


def _make_prefill_context(
    *,
    device: torch.device,
    seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_layers: int = 1,
    total_slots: int | None = None,
):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    seq_lens = torch.tensor([seq_len], device=device)
    backend = _RealMLAPrefillBackend()
    backend.init_prefill_metadata(seq_lens)
    return ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=_RealMLAKVPool(
            device,
            total_slots=seq_len if total_slots is None else total_slots,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            num_layers=num_layers,
        ),
        bs=1,
        num_extends=1,
        input_num_tokens=seq_len,
        forward_mode=ForwardMode.EXTEND,
    )


def _load_dense_decoder_layer(
    layer,
    reader: _SliceReader,
    *,
    prefix: str,
    device: torch.device,
) -> None:
    _load_decoder_layer_norms(layer, reader, prefix=prefix, device=device)
    _load_real_attention_weights(
        layer.self_attn,
        reader,
        prefix=f"{prefix}.self_attn",
        device=device,
    )
    _load_real_mlp_weights(layer.mlp, reader, prefix=f"{prefix}.mlp", device=device)


def _load_moe_decoder_layer(
    layer,
    reader: _SliceReader,
    *,
    contract: KimiMxfp4MlpShardContract,
    prefix: str,
    device: torch.device,
) -> None:
    _load_decoder_layer_norms(layer, reader, prefix=prefix, device=device)
    _load_real_attention_weights(
        layer.self_attn,
        reader,
        prefix=f"{prefix}.self_attn",
        device=device,
    )
    layer.mlp.gate.weight.data.copy_(
        reader.read_tensor(f"{prefix}.mlp.gate.weight", device=device)
    )
    if f"{prefix}.mlp.gate.e_score_correction_bias" in reader._weight_map:
        layer.mlp.gate.e_score_correction_bias.data.copy_(
            reader.read_tensor(
                f"{prefix}.mlp.gate.e_score_correction_bias",
                device=device,
            )
        )
    _load_real_mlp_weights(
        layer.mlp.shared_experts,
        reader,
        prefix=f"{prefix}.mlp.shared_experts",
        device=device,
    )
    _load_real_routed_weights(
        layer.mlp.experts,
        reader,
        contract,
        experts_prefix=f"{prefix}.mlp.experts",
        device=device,
    )
    layer.mlp.experts.process_weights_after_loading(layer.mlp.experts)


def _run_dense_decoder_layer_reference(
    layer,
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    defer_output_allreduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states
    normed = _rmsnorm_module_reference(layer.input_layernorm, hidden_states)
    attn = _real_attention_reference(layer.self_attn, normed, positions)
    normed, residual = _rmsnorm_add_module_reference(
        layer.post_attention_layernorm, attn, residual
    )
    mlp_partial = layer.mlp(normed)
    if defer_output_allreduce:
        return mlp_partial, residual
    return _all_reduce_clone(mlp_partial), residual


def _run_moe_decoder_layer_reference(
    layer,
    reader: _SliceReader,
    *,
    contract: KimiMxfp4MlpShardContract,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    input_is_deferred_partial: bool = False,
    defer_output_allreduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_is_deferred_partial:
        hidden_states = _all_reduce_clone(hidden_states)
    normed, residual = _rmsnorm_add_module_reference(
        layer.input_layernorm, hidden_states, residual
    )
    attn = _real_attention_reference(layer.self_attn, normed, positions)
    normed, residual = _rmsnorm_add_module_reference(
        layer.post_attention_layernorm, attn, residual
    )

    router_logits = layer.mlp.gate(normed)
    routed = _run_real_routed_reference(
        reader,
        contract=contract,
        experts_prefix=f"language_model.model.layers.{layer.layer_id}.mlp.experts",
        hidden_states=normed,
        router_logits=router_logits,
        correction_bias=layer.mlp.gate.e_score_correction_bias,
        device=normed.device,
    )
    routed_partial = layer.mlp._prepare_routed_output_for_post_moe_comm(routed)
    partial = routed_partial + layer.mlp.shared_experts(normed)
    if defer_output_allreduce:
        return partial, residual
    return _all_reduce_clone(partial), residual


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
    experts_prefix: str = "language_model.model.layers.1.mlp.experts",
    device: torch.device,
) -> None:
    ownership = contract.num_rank_local_experts
    assert ownership == layer.num_local_experts
    expert_start = layer.backend.spec.ep_rank * layer.num_local_experts
    for local_expert_id in range(layer.num_local_experts):
        global_expert_id = expert_start + local_expert_id
        prefix = f"{experts_prefix}.{global_expert_id}"
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


def _run_real_routed_reference(
    reader: _SliceReader,
    *,
    contract: KimiMxfp4MlpShardContract,
    experts_prefix: str = "language_model.model.layers.1.mlp.experts",
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    from tokenspeed.runtime.layers.dense.mxfp4 import dequantize_mxfp4_linear_weight
    from tokenspeed.runtime.layers.moe.backends.mxfp4.activation import (
        dequantize_mxfp4_activation,
        quantize_mxfp4_activation_reference,
    )
    from tokenspeed.runtime.layers.moe.backends.mxfp4.experts import (
        kimi_swiglu_gate_up,
    )
    from tokenspeed.runtime.layers.moe.backends.mxfp4.routing import (
        select_kimi_sigmoid_noaux_topk,
    )

    topk_weights, topk_ids = select_kimi_sigmoid_noaux_topk(
        router_logits.float().cpu(),
        top_k=8,
        correction_bias=correction_bias.cpu(),
        renormalize=True,
        routed_scaling_factor=2.827,
        apply_routed_scaling_factor_on_output=True,
    )
    topk_weights = topk_weights.to(device=device, dtype=torch.float32)

    packed_hidden, hidden_scale = quantize_mxfp4_activation_reference(hidden_states)
    dequant_hidden = dequantize_mxfp4_activation(
        packed_hidden,
        hidden_scale,
        logical_shape=tuple(hidden_states.shape),
        output_dtype=torch.float32,
    )

    returned_slots = torch.empty(
        (*topk_ids.shape, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=device,
    )
    for token in range(topk_ids.shape[0]):
        hidden = dequant_hidden[token : token + 1]
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            prefix = f"{experts_prefix}.{expert}"
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

            gate_weight = dequantize_mxfp4_linear_weight(gate_packed, gate_scale)
            up_weight = dequantize_mxfp4_linear_weight(up_packed, up_scale)
            gate = F.linear(hidden, gate_weight)
            up = F.linear(hidden, up_weight)
            gate_up = torch.empty(
                (1, gate.shape[-1] * 2),
                dtype=gate.dtype,
                device=device,
            )
            gate_up[..., 0::2] = gate
            gate_up[..., 1::2] = up
            activated = kimi_swiglu_gate_up(gate_up, output_dtype=hidden_states.dtype)
            del gate_weight, up_weight, gate, up, gate_up, gate_packed, gate_scale
            del up_packed, up_scale

            packed_intermediate, intermediate_scale = quantize_mxfp4_activation_reference(
                activated
            )
            dequant_intermediate = dequantize_mxfp4_activation(
                packed_intermediate,
                intermediate_scale,
                logical_shape=tuple(activated.shape),
                output_dtype=torch.float32,
            )
            down_weight = dequantize_mxfp4_linear_weight(down_packed, down_scale)
            down = F.linear(dequant_intermediate, down_weight)
            returned_slots[token, slot].copy_(down.to(returned_slots.dtype)[0])
            del down_weight, down, down_packed, down_scale
            del activated, packed_intermediate, intermediate_scale, dequant_intermediate

    weighted_slots = returned_slots.float() * topk_weights.unsqueeze(-1)
    return weighted_slots.sum(dim=1).to(hidden_states.dtype)


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
    reader: _SliceReader,
    contract: KimiMxfp4MlpShardContract,
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
    router_logits = _router_logits(hidden_states.shape[0], rank, hidden_states.device)
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
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
    expected = _run_real_routed_reference(
        reader,
        contract=contract,
        experts_prefix="language_model.model.layers.1.mlp.experts",
        hidden_states=hidden_states,
        router_logits=router_logits,
        correction_bias=correction_bias,
        device=hidden_states.device,
    )
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=0.35,
        rtol=0.18,
        check_dtype=False,
    )
    assert any(
        call.get("expected_kernel_name") == "gluon_grouped_biased_topk_gfx950"
        and call.get("traits", {}).get("output_type") == "topk"
        for call in kernel_calls["route"]
    )
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


def _run_replicated_routed_post_moe_allreduce_case(
    layer,
    *,
    reader: _SliceReader,
    contract: KimiMxfp4MlpShardContract,
    rank: int,
    world_size: int,
) -> None:
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MoE

    torch.manual_seed(20260608)
    hidden_states = (
        torch.randn(1, layer.hidden_size, device=layer.w13_weight.device) * 0.002
    ).bfloat16()
    correction_bias = torch.zeros(layer.num_experts, device=hidden_states.device)
    router_logits = _router_logits(hidden_states.shape[0], 0, hidden_states.device)
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
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
    expected = _run_real_routed_reference(
        reader,
        contract=contract,
        experts_prefix="language_model.model.layers.1.mlp.experts",
        hidden_states=hidden_states,
        router_logits=router_logits,
        correction_bias=correction_bias,
        device=hidden_states.device,
    )
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=0.35,
        rtol=0.18,
        check_dtype=False,
    )

    moe = object.__new__(DeepseekV3MoE)
    moe.mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
        moe_tp_size=1,
        moe_ep_size=world_size,
    )
    moe.experts = SimpleNamespace(backend=layer.backend)
    partial_output = moe._prepare_routed_output_for_post_moe_comm(output.clone())
    torch.testing.assert_close(
        partial_output.float(),
        (output / world_size).float(),
        atol=0.01,
        rtol=0.01,
        check_dtype=False,
    )
    dist.all_reduce(partial_output)
    torch.testing.assert_close(
        partial_output.float(),
        expected.float(),
        atol=0.35,
        rtol=0.18,
        check_dtype=False,
    )


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
        dense_layer = _make_real_mlp(
            model_config,
            rank=rank,
            world_size=world_size,
            device=device,
            prefix="model.layers.0.mlp",
            intermediate_size=int(model_config["text_config"]["intermediate_size"]),
            is_shared_expert=False,
        )
        _load_real_mlp_weights(
            dense_layer,
            reader,
            prefix="language_model.model.layers.0.mlp",
            device=device,
        )
        loaded_dense_output = _run_loaded_mlp_matches_reference(
            dense_layer,
            dense_output,
            hidden_states,
        )
        shared_layer = _make_real_mlp(
            model_config,
            rank=rank,
            world_size=world_size,
            device=device,
            prefix="model.layers.1.mlp.shared_experts",
            intermediate_size=int(model_config["text_config"]["moe_intermediate_size"]),
            is_shared_expert=True,
        )
        _load_real_mlp_weights(
            shared_layer,
            reader,
            prefix="language_model.model.layers.1.mlp.shared_experts",
            device=device,
        )
        loaded_shared_output = _run_loaded_mlp_matches_reference(
            shared_layer,
            shared_output,
            hidden_states,
        )
        kernel_calls = _record_kernel_calls(monkeypatch)
        layer = _make_real_moe_layer(rank, world_size, device)
        _load_real_routed_weights(layer, reader, contract.routed, device=device)
        layer.process_weights_after_loading(layer)

        assert dense_output.shape == (1, contract.hidden_size)
        assert shared_output.shape == (1, contract.hidden_size)
        assert loaded_dense_output.shape == dense_output.shape
        assert loaded_shared_output.shape == shared_output.shape
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
            reader=reader,
            contract=contract.routed,
            rank=rank,
            world_size=world_size,
            kernel_calls=kernel_calls,
        )
        _run_replicated_routed_post_moe_allreduce_case(
            layer,
            reader=reader,
            contract=contract.routed,
            rank=rank,
            world_size=world_size,
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


def test_local_kimi_mxfp4_real_lm_head_logits_tp_gather() -> None:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    group = _ensure_runtime_process_group(world_size)
    weight_map = _weight_map(model)
    config = _model_config(model)["text_config"]
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    rows_per_rank = vocab_size // world_size
    assert vocab_size % world_size == 0

    lm_head = ParallelLMHead(
        vocab_size,
        hidden_size,
        params_dtype=torch.bfloat16,
        tp_rank=rank,
        tp_size=world_size,
        tp_group=group,
    ).to(device)
    with _SliceReader(model, weight_map) as reader:
        lm_head.weight.data.copy_(
            reader.read_2d(
                "language_model.lm_head.weight",
                (rank * rows_per_rank, (rank + 1) * rows_per_rank),
                (0, hidden_size),
                device=device,
            )
        )

    torch.manual_seed(20260608)
    hidden_states = (torch.randn(2, hidden_size, device=device) * 0.01).bfloat16()
    local_expected = torch.matmul(hidden_states, lm_head.weight.T)

    gathered_expected = torch.empty(
        world_size * local_expected.shape[0],
        local_expected.shape[1],
        dtype=local_expected.dtype,
        device=device,
    )
    dist.all_gather_into_tensor(
        gathered_expected,
        local_expected,
        group=pg_manager.get_process_group("nccl", group),
    )
    expected = (
        gathered_expected.view(world_size, local_expected.shape[0], rows_per_rank)
        .transpose(0, 1)
        .contiguous()
        .view(local_expected.shape[0], vocab_size)
    )

    logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=vocab_size, model_type="kimi_k2"),
        tp_rank=rank,
        tp_size=world_size,
        tp_group=group,
    )
    output = logits_processor(
        torch.zeros(hidden_states.shape[0], dtype=torch.long, device=device),
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
    )

    torch.testing.assert_close(
        output.next_token_logits.float(),
        expected.float(),
        atol=0.02,
        rtol=0.02,
        check_dtype=False,
    )
    rank_slice = slice(rank * rows_per_rank, (rank + 1) * rows_per_rank)
    torch.testing.assert_close(
        output.next_token_logits[:, rank_slice].float(),
        local_expected.float(),
        atol=0.02,
        rtol=0.02,
        check_dtype=False,
    )
    assert torch.isfinite(output.next_token_logits).all()
    dist.barrier()


def test_local_kimi_mxfp4_real_attention_prefill_matches_reference() -> None:
    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    _ensure_runtime_process_group(world_size)
    model_config = _model_config(model)
    weight_map = _weight_map(model)
    torch.manual_seed(20260608)
    hidden_states = (
        torch.randn(3, int(model_config["text_config"]["hidden_size"]), device=device)
        * 0.01
    ).bfloat16()

    with _SliceReader(model, weight_map) as reader:
        attn = _make_real_attention(
            model_config,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        _load_real_attention_weights(
            attn,
            reader,
            prefix="language_model.model.layers.0.self_attn",
            device=device,
        )
        _run_real_attention_case(
            attn,
            rank=rank,
            world_size=world_size,
            hidden_states=hidden_states,
        )


def test_local_kimi_mxfp4_real_attention_decode_matches_reference() -> None:
    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    _ensure_runtime_process_group(world_size)
    model_config = _model_config(model)
    weight_map = _weight_map(model)
    torch.manual_seed(20260609)
    prefill_hidden_states = (
        torch.randn(3, int(model_config["text_config"]["hidden_size"]), device=device)
        * 0.01
    ).bfloat16()
    decode_hidden_states = (
        torch.randn(1, int(model_config["text_config"]["hidden_size"]), device=device)
        * 0.01
    ).bfloat16()

    with _SliceReader(model, weight_map) as reader:
        attn = _make_real_attention(
            model_config,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        _load_real_attention_weights(
            attn,
            reader,
            prefix="language_model.model.layers.0.self_attn",
            device=device,
        )
        _run_real_attention_decode_case(
            attn,
            rank=rank,
            world_size=world_size,
            prefill_hidden_states=prefill_hidden_states,
            decode_hidden_states=decode_hidden_states,
        )


def test_local_kimi_mxfp4_real_decoder_layers_match_reference() -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    _ensure_runtime_process_group(world_size)
    model_config = _model_config(model)
    weight_map = _weight_map(model)
    contract = build_kimi_mxfp4_sharding_contract(
        model_config,
        tp_size=world_size,
        ep_size=world_size,
        tp_rank=rank,
        ep_rank=rank,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    torch.manual_seed(20260608)
    seq_len = 3
    num_layers = int(
        os.environ.get("TOKENSPEED_KIMI_MXFP4_REAL_LAYER_PROBE_DEPTH", "4")
    )
    enable_allreduce_fusion = (
        os.environ.get(
            "TOKENSPEED_KIMI_MXFP4_REAL_LAYER_ENABLE_ALLREDUCE_FUSION", "0"
        )
        == "1"
    )
    assert 1 <= num_layers <= int(model_config["text_config"]["num_hidden_layers"])
    hidden_states = (
        torch.randn(seq_len, int(model_config["text_config"]["hidden_size"]), device=device)
        * 0.01
    ).bfloat16()
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    out_cache_loc = torch.arange(seq_len, device=device, dtype=torch.int32)

    old_enable_allreduce_fusion = global_server_args_dict.get("enable_allreduce_fusion")
    old_comm_fusion_max_num_tokens = global_server_args_dict.get(
        "comm_fusion_max_num_tokens"
    )
    global_server_args_dict["enable_allreduce_fusion"] = enable_allreduce_fusion
    global_server_args_dict["comm_fusion_max_num_tokens"] = 2048
    if not enable_allreduce_fusion:
        global_server_args_dict["comm_fusion_max_num_tokens"] = 0
    try:
        with _SliceReader(model, weight_map) as reader:
            actual_hidden = hidden_states
            actual_residual = None
            actual_hidden_is_deferred_partial = False
            for layer_id in range(num_layers):
                layer = _make_real_decoder_layer(
                    model_config,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    layer_id=layer_id,
                    enable_allreduce_fusion=enable_allreduce_fusion,
                )
                prefix = f"language_model.model.layers.{layer_id}"
                if layer.is_moe_layer:
                    _load_moe_decoder_layer(
                        layer,
                        reader,
                        contract=contract.routed,
                        prefix=prefix,
                        device=device,
                    )
                else:
                    _load_dense_decoder_layer(
                        layer,
                        reader,
                        prefix=prefix,
                        device=device,
                    )
                reference_hidden = actual_hidden
                reference_residual = actual_residual
                output_deferred_partial = layer.comm_manager.should_fuse(
                    reference_hidden.shape[0]
                )
                if layer.is_moe_layer:
                    assert reference_residual is not None
                    expected_hidden, expected_residual = (
                        _run_moe_decoder_layer_reference(
                            layer,
                            reader,
                            contract=contract.routed,
                            positions=positions,
                            hidden_states=reference_hidden,
                            residual=reference_residual,
                            input_is_deferred_partial=actual_hidden_is_deferred_partial,
                            defer_output_allreduce=output_deferred_partial,
                        )
                    )
                    atol, rtol = 0.35, 0.18
                else:
                    assert reference_residual is None
                    expected_hidden, expected_residual = (
                        _run_dense_decoder_layer_reference(
                            layer,
                            positions=positions,
                            hidden_states=reference_hidden,
                            defer_output_allreduce=output_deferred_partial,
                        )
                    )
                    atol, rtol = 0.12, 0.08

                actual_hidden, actual_residual = layer(
                    positions,
                    actual_hidden,
                    _make_prefill_context(
                        device=device,
                        seq_len=seq_len,
                        kv_lora_rank=layer.self_attn.kv_lora_rank,
                        qk_rope_head_dim=layer.self_attn.qk_rope_head_dim,
                        num_layers=num_layers,
                    ),
                    out_cache_loc,
                    residual=actual_residual,
                )
                torch.cuda.synchronize()
                hidden_diff = (actual_hidden.float() - expected_hidden.float()).abs().max()
                residual_diff = (
                    actual_residual.float() - expected_residual.float()
                ).abs().max()
                if rank == 0:
                    print(
                        "decoder_layer=",
                        layer.layer_id,
                        "hidden_diff=",
                        float(hidden_diff.item()),
                        "residual_diff=",
                        float(residual_diff.item()),
                        "hidden_abs_max=",
                        float(actual_hidden.float().abs().max().item()),
                    )
                if hidden_diff > atol:
                    _print_top_abs_diffs(
                        label=f"rank={rank} decoder_layer={layer.layer_id} hidden",
                        actual=actual_hidden,
                        expected=expected_hidden,
                    )
                if residual_diff > atol:
                    _print_top_abs_diffs(
                        label=f"rank={rank} decoder_layer={layer.layer_id} residual",
                        actual=actual_residual,
                        expected=expected_residual,
                    )
                torch.testing.assert_close(
                    actual_hidden.float(),
                    expected_hidden.float(),
                    atol=atol,
                    rtol=rtol,
                    check_dtype=False,
                )
                torch.testing.assert_close(
                    actual_residual.float(),
                    expected_residual.float(),
                    atol=atol,
                    rtol=rtol,
                    check_dtype=False,
                )
                assert torch.isfinite(actual_hidden).all()
                assert torch.isfinite(actual_residual).all()
                actual_hidden_is_deferred_partial = output_deferred_partial
                del layer
                torch.cuda.empty_cache()
    finally:
        global_server_args_dict["enable_allreduce_fusion"] = old_enable_allreduce_fusion
        global_server_args_dict["comm_fusion_max_num_tokens"] = (
            old_comm_fusion_max_num_tokens
        )
    dist.barrier()


def test_local_kimi_mxfp4_real_manual_full_prefill_next_token_smoke() -> None:
    if os.environ.get("TOKENSPEED_KIMI_MXFP4_REAL_FULL_PREFILL", "0") != "1":
        pytest.skip("set TOKENSPEED_KIMI_MXFP4_REAL_FULL_PREFILL=1 to run full probe")

    from transformers import AutoTokenizer

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
    from tokenspeed.runtime.utils.env import global_server_args_dict

    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    group = _ensure_runtime_process_group(world_size)
    model_config = _model_config(model)
    text_config = model_config["text_config"]
    weight_map = _weight_map(model)
    contract = build_kimi_mxfp4_sharding_contract(
        model_config,
        tp_size=world_size,
        ep_size=world_size,
        tp_rank=rank,
        ep_rank=rank,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
        dtype=torch.long,
    )
    seq_len = int(input_ids.numel())
    num_layers = int(text_config["num_hidden_layers"])
    hidden_size = int(text_config["hidden_size"])
    vocab_size = int(text_config["vocab_size"])
    rows_per_rank = vocab_size // world_size
    assert vocab_size % world_size == 0

    old_enable_allreduce_fusion = global_server_args_dict.get("enable_allreduce_fusion")
    old_comm_fusion_max_num_tokens = global_server_args_dict.get(
        "comm_fusion_max_num_tokens"
    )
    enable_allreduce_fusion = (
        os.environ.get(
            "TOKENSPEED_KIMI_MXFP4_REAL_FULL_ENABLE_ALLREDUCE_FUSION", "1"
        )
        == "1"
    )
    global_server_args_dict["enable_allreduce_fusion"] = enable_allreduce_fusion
    global_server_args_dict["comm_fusion_max_num_tokens"] = 2048
    if not enable_allreduce_fusion:
        global_server_args_dict["comm_fusion_max_num_tokens"] = 0

    try:
        with _SliceReader(model, weight_map) as reader:
            hidden_states = _load_prompt_embeddings(
                reader,
                input_ids,
                hidden_size=hidden_size,
                device=device,
            )
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            out_cache_loc = torch.arange(seq_len, device=device, dtype=torch.int32)
            ctx = _make_prefill_context(
                device=device,
                seq_len=seq_len,
                kv_lora_rank=int(text_config["kv_lora_rank"]),
                qk_rope_head_dim=int(text_config["qk_rope_head_dim"]),
                num_layers=num_layers,
            )
            residual = None
            final_comm_manager = None
            for layer_id in range(num_layers):
                layer = _make_real_decoder_layer(
                    model_config,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    layer_id=layer_id,
                    enable_allreduce_fusion=enable_allreduce_fusion,
                )
                prefix = f"language_model.model.layers.{layer_id}"
                if layer.is_moe_layer:
                    _load_moe_decoder_layer(
                        layer,
                        reader,
                        contract=contract.routed,
                        prefix=prefix,
                        device=device,
                    )
                else:
                    _load_dense_decoder_layer(
                        layer,
                        reader,
                        prefix=prefix,
                        device=device,
                    )
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    ctx,
                    out_cache_loc,
                    residual=residual,
                )
                final_comm_manager = layer.comm_manager
                del layer
                torch.cuda.empty_cache()

            assert final_comm_manager is not None
            final_norm = RMSNorm(
                hidden_size,
                eps=float(text_config["rms_norm_eps"]),
            ).to(device=device, dtype=torch.bfloat16)
            final_norm.weight.data.copy_(
                reader.read_tensor("language_model.model.norm.weight", device=device)
            )
            hidden_states = final_comm_manager.final_norm(
                hidden_states,
                residual,
                ctx,
                final_norm,
            )

            lm_head = ParallelLMHead(
                vocab_size,
                hidden_size,
                params_dtype=torch.bfloat16,
                tp_rank=rank,
                tp_size=world_size,
                tp_group=group,
            ).to(device)
            lm_head.weight.data.copy_(
                reader.read_2d(
                    "language_model.lm_head.weight",
                    (rank * rows_per_rank, (rank + 1) * rows_per_rank),
                    (0, hidden_size),
                    device=device,
                )
            )
            logits_processor = LogitsProcessor(
                SimpleNamespace(vocab_size=vocab_size, model_type="kimi_k25"),
                tp_rank=rank,
                tp_size=world_size,
                tp_group=group,
            )
            logits_output = logits_processor(
                input_ids,
                hidden_states,
                lm_head,
                LogitsMetadata(
                    forward_mode=ForwardMode.EXTEND,
                    extend_seq_lens=torch.tensor([seq_len], device=device),
                ),
            )
            logits = logits_output.next_token_logits.float()
            assert logits.shape == (1, vocab_size)
            assert torch.isfinite(logits).all()
            assert float((logits.max() - logits.min()).item()) > 0.0
            top_values, top_ids = torch.topk(logits[0], k=8)
            if rank == 0:
                print(
                    "manual_full_prefill_argmax=",
                    int(top_ids[0].item()),
                    "decoded=",
                    tokenizer.decode([int(top_ids[0].item())]),
                    "top_ids=",
                    [int(x) for x in top_ids.tolist()],
                    "top_values=",
                    [float(x) for x in top_values.tolist()],
                    "top_decoded=",
                    [tokenizer.decode([int(x)]) for x in top_ids.tolist()],
                )
    finally:
        global_server_args_dict["enable_allreduce_fusion"] = old_enable_allreduce_fusion
        global_server_args_dict["comm_fusion_max_num_tokens"] = (
            old_comm_fusion_max_num_tokens
        )
    dist.barrier()


def _load_official_kimi_mxfp4_model(
    model: Path,
    *,
    rank: int,
    world_size: int,
):
    from tokenspeed.runtime.configs.device_config import DeviceConfig
    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.model_loader import get_model
    from tokenspeed.runtime.utils.env import global_server_args_dict_update
    from tokenspeed.runtime.utils.server_args import ServerArgs

    server_args = ServerArgs(
        model=str(model),
        tokenizer=str(model),
        trust_remote_code=True,
        quantization="mxfp4",
        language_model_only=True,
        world_size=world_size,
        nprocs_per_node=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
        enable_expert_parallel=True,
        max_model_len=4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        max_cudagraph_capture_size=1,
    )
    server_args.mapping.rank = rank
    server_args.attention_backend = "tokenspeed_mla"
    global_server_args_dict_update(server_args)

    model_config = ModelConfig(
        model_path=str(model),
        trust_remote_code=True,
        context_length=4096,
        model_override_args="{}",
        dtype="auto",
        quantization="mxfp4",
        server_args=server_args,
    )
    loaded_model = get_model(
        model_config=model_config,
        load_config=LoadConfig(load_format=server_args.load_format),
        device_config=DeviceConfig("cuda"),
    )
    return loaded_model


def _assert_first_moe_weights_match(official_moe, manual_moe, *, rank: int) -> None:
    if rank == 0:
        print(
            "official_manual_moe_state=",
            {
                "official_ep_rank": official_moe.experts.ep_rank,
                "manual_ep_rank": manual_moe.experts.ep_rank,
                "official_ep_size": official_moe.experts.ep_size,
                "manual_ep_size": manual_moe.experts.ep_size,
                "official_backend": type(official_moe.experts.backend).__name__,
                "manual_backend": type(manual_moe.experts.backend).__name__,
            },
        )
    for attr in (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ):
        official_tensor = getattr(official_moe.experts, attr)
        manual_tensor = getattr(manual_moe.experts, attr)
        if official_tensor.dtype == torch.uint8:
            diff = (
                official_tensor.to(torch.int16) - manual_tensor.to(torch.int16)
            ).abs()
            max_diff = int(diff.max().item())
        else:
            max_diff = float((official_tensor - manual_tensor).abs().max().item())
        if rank == 0 or max_diff != 0:
            print(
                "official_manual_moe_weight_diff=",
                {"rank": rank, "attr": attr, "max_diff": max_diff},
            )
        assert torch.equal(official_tensor, manual_tensor), (
            f"official/manual first MoE {attr} mismatch on rank {rank}"
        )


def test_local_kimi_mxfp4_real_official_load_prefill_next_token_smoke() -> None:
    if os.environ.get("TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_LOAD_PREFILL", "0") != "1":
        pytest.skip(
            "set TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_LOAD_PREFILL=1 to run "
            "official loader probe"
        )

    from transformers import AutoTokenizer

    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    _ensure_runtime_process_group(world_size)

    loaded_model = _load_official_kimi_mxfp4_model(
        model,
        rank=rank,
        world_size=world_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
        dtype=torch.long,
    )
    seq_len = int(input_ids.numel())
    text_config = _model_config(model)["text_config"]
    ctx = _make_prefill_context(
        device=device,
        seq_len=seq_len,
        kv_lora_rank=int(text_config["kv_lora_rank"]),
        qk_rope_head_dim=int(text_config["qk_rope_head_dim"]),
        num_layers=int(text_config["num_hidden_layers"]),
    )
    with torch.no_grad():
        logits_output = loaded_model(
            ctx=ctx,
            input_ids=input_ids,
            positions=torch.arange(seq_len, device=device, dtype=torch.long),
            out_cache_loc=torch.arange(seq_len, device=device, dtype=torch.int32),
            input_lengths=torch.tensor([seq_len], device=device),
        )
    logits = logits_output.next_token_logits.float()
    assert logits.shape == (1, int(text_config["vocab_size"]))
    assert torch.isfinite(logits).all()
    top_values, top_ids = torch.topk(logits[0], k=8)
    if rank == 0:
        print(
            "official_load_prefill_argmax=",
            int(top_ids[0].item()),
            "decoded=",
            tokenizer.decode([int(top_ids[0].item())]),
            "top_ids=",
            [int(x) for x in top_ids.tolist()],
            "top_values=",
            [float(x) for x in top_values.tolist()],
            "top_decoded=",
            [tokenizer.decode([int(x)]) for x in top_ids.tolist()],
        )
    assert int(top_ids[0].item()) == 41607
    dist.barrier()


def test_local_kimi_mxfp4_real_official_prefill_then_decode_smoke() -> None:
    if (
        os.environ.get("TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_PREFILL_DECODE", "0")
        != "1"
    ):
        pytest.skip(
            "set TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_PREFILL_DECODE=1 to run "
            "official prefill/decode probe"
        )

    from transformers import AutoTokenizer

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.utils.env import global_server_args_dict

    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    _ensure_runtime_process_group(world_size)
    loaded_model = _load_official_kimi_mxfp4_model(
        model,
        rank=rank,
        world_size=world_size,
    )
    old_enable_allreduce_fusion = global_server_args_dict.get("enable_allreduce_fusion")
    old_comm_fusion_max_num_tokens = global_server_args_dict.get(
        "comm_fusion_max_num_tokens"
    )
    language_model = getattr(loaded_model, "language_model", loaded_model)

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
        dtype=torch.long,
    )
    seq_len = int(input_ids.numel())
    text_config = _model_config(model)["text_config"]
    page_size = 64
    cache_page_id = 1
    cache_slot_start = page_size * cache_page_id
    prefill_cache_locs = torch.arange(
        cache_slot_start,
        cache_slot_start + seq_len,
        device=device,
        dtype=torch.int32,
    )
    decode_cache_loc = torch.tensor(
        [cache_slot_start + seq_len],
        device=device,
        dtype=torch.int32,
    )
    prefill_ctx = _make_prefill_context(
        device=device,
        seq_len=seq_len,
        total_slots=max(page_size * 2, cache_slot_start + seq_len + 1),
        kv_lora_rank=int(text_config["kv_lora_rank"]),
        qk_rope_head_dim=int(text_config["qk_rope_head_dim"]),
        num_layers=int(text_config["num_hidden_layers"]),
    )
    with torch.no_grad():
        prefill_output = loaded_model(
            ctx=prefill_ctx,
            input_ids=input_ids,
            positions=torch.arange(seq_len, device=device, dtype=torch.long),
            out_cache_loc=prefill_cache_locs,
            input_lengths=torch.tensor([seq_len], device=device),
        )
    prefill_logits = prefill_output.next_token_logits.float()
    first_top_values, first_top_ids = torch.topk(prefill_logits[0], k=8)
    first_token = int(first_top_ids[0].item())
    assert first_token != 0

    decode_backend = _make_gluon_mla_decode_backend(
        language_model.model.layers[0].self_attn,
        device=device,
        seq_len=seq_len + 1,
        page_id=cache_page_id,
    )
    decode_ctx = ForwardContext(
        attn_backend=decode_backend,
        token_to_kv_pool=prefill_ctx.token_to_kv_pool,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
    )
    decode_input_ids = torch.tensor([first_token], device=device, dtype=torch.long)
    decode_nonfinite_layers: list[int] = []

    def _make_decode_finite_hook(layer_id: int):
        def _hook(_module, _inputs, output) -> None:
            hidden, residual = output
            hidden_finite = torch.isfinite(hidden).all()
            residual_finite = torch.isfinite(residual).all()
            if hidden_finite and residual_finite:
                return
            if not decode_nonfinite_layers:
                print(
                    "official_prefill_decode_nonfinite_layer=",
                    {
                        "rank": rank,
                        "layer_id": layer_id,
                        "hidden_finite": bool(hidden_finite.item()),
                        "residual_finite": bool(residual_finite.item()),
                        "hidden_nan": int(torch.isnan(hidden).sum().item()),
                        "residual_nan": int(torch.isnan(residual).sum().item()),
                    },
                )
            decode_nonfinite_layers.append(layer_id)

        return _hook

    hooks = [
        layer.register_forward_hook(_make_decode_finite_hook(layer_id))
        for layer_id, layer in enumerate(language_model.model.layers)
    ]
    with torch.no_grad():
        try:
            decode_output = loaded_model(
                ctx=decode_ctx,
                input_ids=decode_input_ids,
                positions=torch.tensor([seq_len], device=device, dtype=torch.long),
                out_cache_loc=decode_cache_loc,
                input_lengths=torch.tensor([seq_len + 1], device=device),
            )
        finally:
            global_server_args_dict["enable_allreduce_fusion"] = (
                old_enable_allreduce_fusion
            )
            global_server_args_dict["comm_fusion_max_num_tokens"] = (
                old_comm_fusion_max_num_tokens
            )
            for hook in hooks:
                hook.remove()
    decode_logits = decode_output.next_token_logits.float()
    assert decode_logits.shape == prefill_logits.shape
    assert decode_nonfinite_layers == []
    assert torch.isfinite(decode_logits).all()
    second_top_values, second_top_ids = torch.topk(decode_logits[0], k=8)
    if rank == 0:
        print(
            "official_prefill_decode_first=",
            first_token,
            "first_decoded=",
            tokenizer.decode([first_token]),
            "first_top_ids=",
            [int(x) for x in first_top_ids.tolist()],
            "first_top_values=",
            [float(x) for x in first_top_values.tolist()],
            "second_argmax=",
            int(second_top_ids[0].item()),
            "second_decoded=",
            tokenizer.decode([int(second_top_ids[0].item())]),
            "second_top_ids=",
            [int(x) for x in second_top_ids.tolist()],
            "second_top_values=",
            [float(x) for x in second_top_values.tolist()],
            "second_top_decoded=",
            [tokenizer.decode([int(x)]) for x in second_top_ids.tolist()],
        )
    assert int(second_top_ids[0].item()) != 0
    dist.barrier()


def test_local_kimi_mxfp4_real_official_layers_match_manual_load() -> None:
    if (
        os.environ.get("TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_LAYER_COMPARE", "0")
        != "1"
    ):
        pytest.skip(
            "set TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_LAYER_COMPARE=1 to run "
            "official/manual loader comparison probe"
        )

    from transformers import AutoTokenizer

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead

    model = _require_local_model()
    rank, world_size, device = _require_four_rank_cdna4_gpu()
    group = _ensure_runtime_process_group(world_size)
    model_config = _model_config(model)
    text_config = model_config["text_config"]
    weight_map = _weight_map(model)
    contract = build_kimi_mxfp4_sharding_contract(
        model_config,
        tp_size=world_size,
        ep_size=world_size,
        tp_rank=rank,
        ep_rank=rank,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    loaded_model = _load_official_kimi_mxfp4_model(
        model,
        rank=rank,
        world_size=world_size,
    )
    language_model = getattr(loaded_model, "language_model", loaded_model)

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
        dtype=torch.long,
    )
    seq_len = int(input_ids.numel())
    hidden_size = int(text_config["hidden_size"])
    vocab_size = int(text_config["vocab_size"])
    num_model_layers = int(text_config["num_hidden_layers"])
    num_layers = int(
        os.environ.get(
            "TOKENSPEED_KIMI_MXFP4_REAL_OFFICIAL_LAYER_COMPARE_DEPTH",
            str(num_model_layers),
        )
    )
    assert 1 <= num_layers <= num_model_layers

    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    out_cache_loc = torch.arange(seq_len, device=device, dtype=torch.int32)
    official_ctx = _make_prefill_context(
        device=device,
        seq_len=seq_len,
        kv_lora_rank=int(text_config["kv_lora_rank"]),
        qk_rope_head_dim=int(text_config["qk_rope_head_dim"]),
        num_layers=num_model_layers,
    )
    manual_ctx = _make_prefill_context(
        device=device,
        seq_len=seq_len,
        kv_lora_rank=int(text_config["kv_lora_rank"]),
        qk_rope_head_dim=int(text_config["qk_rope_head_dim"]),
        num_layers=num_model_layers,
    )

    with _SliceReader(model, weight_map) as reader:
        official_hidden = language_model.model.embed_tokens(input_ids)
        manual_hidden = _load_prompt_embeddings(
            reader,
            input_ids,
            hidden_size=hidden_size,
            device=device,
        )
        torch.cuda.synchronize()
        embed_diff = (official_hidden.float() - manual_hidden.float()).abs().max()
        if rank == 0:
            print("official_manual_embed_diff=", float(embed_diff.item()))
        torch.testing.assert_close(
            official_hidden.float(),
            manual_hidden.float(),
            atol=0.01,
            rtol=0.01,
            check_dtype=False,
        )

        official_residual = None
        manual_residual = None
        manual_comm_manager = None
        for layer_id in range(num_layers):
            manual_layer = _make_real_decoder_layer(
                model_config,
                rank=rank,
                world_size=world_size,
                device=device,
                layer_id=layer_id,
                enable_allreduce_fusion=True,
            )
            prefix = f"language_model.model.layers.{layer_id}"
            if manual_layer.is_moe_layer:
                _load_moe_decoder_layer(
                    manual_layer,
                    reader,
                    contract=contract.routed,
                    prefix=prefix,
                    device=device,
                )
            else:
                _load_dense_decoder_layer(
                    manual_layer,
                    reader,
                    prefix=prefix,
                    device=device,
                )

            official_layer = language_model.model.layers[layer_id]
            if layer_id == 1:
                _assert_first_moe_weights_match(
                    official_layer.mlp,
                    manual_layer.mlp,
                    rank=rank,
                )
            atol, rtol = (0.35, 0.18) if official_layer.is_moe_layer else (0.12, 0.08)
            official_hidden, official_residual = official_layer(
                positions,
                official_hidden,
                official_ctx,
                out_cache_loc,
                residual=official_residual,
            )
            manual_hidden, manual_residual = manual_layer(
                positions,
                manual_hidden,
                manual_ctx,
                out_cache_loc,
                residual=manual_residual,
            )
            manual_comm_manager = manual_layer.comm_manager
            torch.cuda.synchronize()
            hidden_diff = (official_hidden.float() - manual_hidden.float()).abs().max()
            residual_diff = (
                official_residual.float() - manual_residual.float()
            ).abs().max()
            if rank == 0:
                print(
                    "official_manual_layer=",
                    layer_id,
                    "hidden_diff=",
                    float(hidden_diff.item()),
                    "residual_diff=",
                    float(residual_diff.item()),
                )
            if hidden_diff > atol:
                _print_top_abs_diffs(
                    label=f"rank={rank} official_manual_layer={layer_id} hidden",
                    actual=official_hidden,
                    expected=manual_hidden,
                )
            if residual_diff > atol:
                _print_top_abs_diffs(
                    label=f"rank={rank} official_manual_layer={layer_id} residual",
                    actual=official_residual,
                    expected=manual_residual,
                )
            torch.testing.assert_close(
                official_hidden.float(),
                manual_hidden.float(),
                atol=atol,
                rtol=rtol,
                check_dtype=False,
            )
            torch.testing.assert_close(
                official_residual.float(),
                manual_residual.float(),
                atol=atol,
                rtol=rtol,
                check_dtype=False,
            )
            del manual_layer
            torch.cuda.empty_cache()

        if num_layers == num_model_layers:
            assert manual_comm_manager is not None
            manual_norm = RMSNorm(
                hidden_size,
                eps=float(text_config["rms_norm_eps"]),
            ).to(device=device, dtype=torch.bfloat16)
            manual_norm.weight.data.copy_(
                reader.read_tensor("language_model.model.norm.weight", device=device)
            )
            official_hidden = language_model.model.layers[-1].comm_manager.final_norm(
                official_hidden,
                official_residual,
                official_ctx,
                language_model.model.norm,
            )
            manual_hidden = manual_comm_manager.final_norm(
                manual_hidden,
                manual_residual,
                manual_ctx,
                manual_norm,
            )
            torch.cuda.synchronize()
            final_hidden_diff = (
                official_hidden.float() - manual_hidden.float()
            ).abs().max()
            if rank == 0:
                print(
                    "official_manual_final_hidden_diff=",
                    float(final_hidden_diff.item()),
                )
            torch.testing.assert_close(
                official_hidden.float(),
                manual_hidden.float(),
                atol=0.03,
                rtol=0.02,
                check_dtype=False,
            )

            rows_per_rank = vocab_size // world_size
            assert vocab_size % world_size == 0
            manual_lm_head = ParallelLMHead(
                vocab_size,
                hidden_size,
                params_dtype=torch.bfloat16,
                tp_rank=rank,
                tp_size=world_size,
                tp_group=group,
            ).to(device)
            manual_lm_head.weight.data.copy_(
                reader.read_2d(
                    "language_model.lm_head.weight",
                    (rank * rows_per_rank, (rank + 1) * rows_per_rank),
                    (0, hidden_size),
                    device=device,
                )
            )
            manual_logits_processor = LogitsProcessor(
                SimpleNamespace(vocab_size=vocab_size, model_type="kimi_k25"),
                tp_rank=rank,
                tp_size=world_size,
                tp_group=group,
            )
            metadata = LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=torch.tensor([seq_len], device=device),
            )
            official_logits = language_model.logits_processor(
                input_ids,
                official_hidden,
                language_model.lm_head,
                metadata,
            ).next_token_logits.float()
            manual_logits = manual_logits_processor(
                input_ids,
                manual_hidden,
                manual_lm_head,
                metadata,
            ).next_token_logits.float()
            logits_diff = (official_logits - manual_logits).abs().max()
            if rank == 0:
                official_top_values, official_top_ids = torch.topk(
                    official_logits[0], k=8
                )
                manual_top_values, manual_top_ids = torch.topk(manual_logits[0], k=8)
                print(
                    "official_manual_logits_diff=",
                    float(logits_diff.item()),
                    "official_top_ids=",
                    [int(x) for x in official_top_ids.tolist()],
                    "official_top_values=",
                    [float(x) for x in official_top_values.tolist()],
                    "manual_top_ids=",
                    [int(x) for x in manual_top_ids.tolist()],
                    "manual_top_values=",
                    [float(x) for x in manual_top_values.tolist()],
                )
            torch.testing.assert_close(
                official_logits,
                manual_logits,
                atol=0.03,
                rtol=0.02,
                check_dtype=False,
            )
    dist.barrier()
