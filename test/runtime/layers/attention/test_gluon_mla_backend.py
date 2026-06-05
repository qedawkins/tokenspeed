from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention


_NUM_HEADS = 4
_KV_LORA_RANK = 256
_ROPE_DIM = 8
_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM
_VALUE_HEAD_DIM = 64
_SCALING = _HEAD_DIM**-0.5


def _require_cdna4_gpu() -> None:
    from tokenspeed_kernel.platform import current_platform

    if not torch.cuda.is_available() or not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for Gluon MLA backend runtime tests")


def _make_config(device: str) -> MLAConfig:
    return MLAConfig(
        device=device,
        backend_name="tokenspeed_mla",
        num_attention_heads=_NUM_HEADS,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        page_size=64,
        context_len=128,
        max_bs=2,
        max_graph_bs=2,
        kv_cache_quant_method="none",
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        v_head_dim=_VALUE_HEAD_DIM,
        scaling=_SCALING,
        kv_cache_dim=_HEAD_DIM,
    )


def _make_layer() -> PagedAttention:
    return PagedAttention(
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        scaling=_SCALING,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=_VALUE_HEAD_DIM,
    )


def test_gluon_mla_backend_prefill_and_decode_smoke() -> None:
    _require_cdna4_gpu()
    device = "cuda"

    from tokenspeed.runtime.layers.attention.backends.gluon_mla import GluonMLABackend

    torch.manual_seed(20260605)
    config = _make_config(device)
    backend = GluonMLABackend(config)
    assert backend.supports_fused_fp8_mla is False

    seq_lens = torch.tensor([3, 2], device=device, dtype=torch.int32)
    cum_seq_lens = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    q = torch.randn(5, _NUM_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.125
    k = torch.randn(5, _NUM_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.125
    v = (
        torch.randn(5, _NUM_HEADS, _VALUE_HEAD_DIM, device=device, dtype=torch.bfloat16)
        * 0.125
    )

    out, lse = backend.forward_extend_chunked(
        q,
        k,
        v,
        _SCALING,
        0.0,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        max_q_len=3,
        max_kv_len=3,
        seq_lens=seq_lens,
        batch_size=2,
        causal=True,
    )
    torch.cuda.synchronize()

    assert out.shape == (5, _NUM_HEADS, _VALUE_HEAD_DIM)
    assert lse.shape == (5, _NUM_HEADS)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()

    layer = _make_layer()
    pool = config.create_pool(
        num_layers=1,
        max_total_num_tokens=128,
        rank=0,
        enable_memory_saver=False,
    )
    prior_locs = torch.tensor([0, 1, 2, 64, 65, 66, 67, 68], device=device)
    prior_k = torch.randn(
        prior_locs.numel(), 1, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    pool.set_mla_kv_buffer(
        layer,
        prior_locs,
        prior_k[..., :_KV_LORA_RANK],
        prior_k[..., _KV_LORA_RANK:],
    )

    req_pool_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    req_to_page = torch.tensor([[0, 0], [1, 0]], device=device, dtype=torch.int32)
    decode_seq_lens = torch.tensor([4, 6], device=device, dtype=torch.int32)
    backend.init_forward_metadata(
        bs=2,
        num_extends=0,
        req_pool_indices=req_pool_indices,
        seq_lens=decode_seq_lens,
        forward_mode=ForwardMode.DECODE,
        req_to_page=req_to_page,
    )

    query = torch.randn(2, _NUM_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.125
    current_k = torch.randn(2, 1, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.125
    decode_out = backend.forward_decode(
        query,
        current_k,
        None,
        layer,
        torch.tensor([3, 69], device=device),
        pool,
        bs=2,
        save_kv_cache=True,
    )
    torch.cuda.synchronize()

    assert decode_out.shape == (2, _NUM_HEADS * _VALUE_HEAD_DIM)
    assert torch.isfinite(decode_out).all()
