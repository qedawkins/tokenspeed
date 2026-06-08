from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3MoE,
    _qkv_a_quant_block_size,
)


def test_qkv_a_quant_block_size_defaults_for_mxfp4_config() -> None:
    assert _qkv_a_quant_block_size(Mxfp4Config()) == 1


def test_qkv_a_quant_block_size_uses_fp8_weight_block_size() -> None:
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[16, 32],
    )

    assert _qkv_a_quant_block_size(quant_config) == 16


def test_mxfp4_ep_replicated_routed_output_is_made_partial_before_moe_allreduce():
    moe = object.__new__(DeepseekV3MoE)
    moe.mapping = Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=4,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
    )
    moe.experts = SimpleNamespace(
        backend=SimpleNamespace(returns_replicated_routed_output=True)
    )
    routed = torch.tensor([[4.0, 8.0]], dtype=torch.float32)

    actual = moe._prepare_routed_output_for_post_moe_comm(routed)

    torch.testing.assert_close(actual, routed / 4)


def test_non_replicated_routed_output_is_not_changed_before_moe_allreduce():
    moe = object.__new__(DeepseekV3MoE)
    moe.mapping = Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=4,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
    )
    moe.experts = SimpleNamespace(
        backend=SimpleNamespace(returns_replicated_routed_output=False)
    )
    routed = torch.tensor([[4.0, 8.0]], dtype=torch.float32)

    actual = moe._prepare_routed_output_for_post_moe_comm(routed)

    assert actual is routed


def test_moe_allreduce_reports_replicated_token_capacity():
    mapping = Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=4,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
    )
    manager = CommManager(mapping=mapping, layer_id=1, is_moe=True, prev_is_moe=False)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1,
        input_num_tokens=15,
        forward_mode=ForwardMode.EXTEND,
    )

    assert manager.get_num_tokens(ctx) == (60, 15)


def test_moe_allreduce_reports_replicated_draft_pruned_capacity():
    mapping = Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=4,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
    )
    manager = CommManager(mapping=mapping, layer_id=1, is_moe=True, prev_is_moe=False)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=8,
        forward_mode=ForwardMode.DECODE,
        draft_first_step_reduce=True,
    )

    assert manager.get_num_tokens(ctx) == (8, 2)


def test_absorb_decode_copies_returned_rope_outputs_into_q_and_k_cache():
    attention = object.__new__(DeepseekV3AttentionMLA)
    attention.num_local_heads = 2
    attention.qk_nope_head_dim = 2
    attention.qk_rope_head_dim = 2
    attention.qk_head_dim = 4
    attention.kv_lora_rank = 3
    attention.w_kc = torch.zeros(2, 2, 3)
    attention.attn_mqa = SimpleNamespace(k_scale_float=2.0, layer_id=0)
    attention.attention_backend = "tokenspeed_mla"
    attention.use_fused_set_kv_buffer = False

    class ReturningRotary:
        def __call__(
            self,
            positions,
            query,
            key,
            *,
            fused_set_kv_buffer_arg=None,
            output_q_rope=None,
        ):
            assert fused_set_kv_buffer_arg is None
            assert output_q_rope is not None
            return query + 100, key + 200

    class RecordingKVPool:
        def __init__(self):
            self.cache_k_nope = None
            self.cache_k_rope = None

        def set_mla_kv_buffer(
            self,
            layer,
            loc,
            *,
            cache_k_nope,
            cache_k_rope,
        ):
            self.cache_k_nope = cache_k_nope.clone()
            self.cache_k_rope = cache_k_rope.clone()

    attention.rotary_emb = ReturningRotary()
    kv_pool = RecordingKVPool()
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            supports_fused_fp8_mla=False,
            data_type=torch.bfloat16,
        ),
        token_to_kv_pool=kv_pool,
    )
    q = torch.arange(8, dtype=torch.float32).view(1, 8)
    latent_cache = torch.arange(5, dtype=torch.float32).view(1, 5)
    q_pe = q.view(1, 2, 4)[..., 2:].clone()
    k_pe = latent_cache.unsqueeze(1)[..., 3:].clone()

    Q, K = DeepseekV3AttentionMLA.forward_absorb_qkv_proj(
        attention,
        q,
        latent_cache,
        torch.tensor([0]),
        ctx,
        torch.tensor([0], dtype=torch.int32),
    )

    torch.testing.assert_close(Q[..., 3:], q_pe + 100)
    torch.testing.assert_close(K[..., 3:], k_pe + 200)
    torch.testing.assert_close(kv_pool.cache_k_rope, k_pe + 200)
