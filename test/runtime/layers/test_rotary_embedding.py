import torch

import pytest

from tokenspeed.runtime.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    get_rope,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _deepseek_checkpoint_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).transpose(-1, -2).reshape_as(x)
    return x * cos + _rotate_half(x) * sin


def test_deepseek_scaling_rotary_embedding_constructs_on_amd() -> None:
    rope = DeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=True,
        scaling_factor=4.0,
        dtype=torch.float32,
        device="cpu",
    )
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    query = torch.randn((1, 2, 1, 8), dtype=torch.float32)
    key = torch.randn((1, 2, 1, 8), dtype=torch.float32)

    query_out, key_out = rope(positions, query, key)

    assert query_out.shape == query.shape
    assert key_out.shape == key.shape
    assert torch.isfinite(query_out).all()
    assert torch.isfinite(key_out).all()


def test_deepseek_scaling_rotary_embedding_writes_output_buffers_on_amd_path() -> None:
    rope = DeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=False,
        scaling_factor=4.0,
        dtype=torch.float32,
        device="cpu",
    )
    positions = torch.tensor([0, 1], dtype=torch.long)
    query = torch.randn((2, 1, 8), dtype=torch.float32)
    key = torch.randn((2, 1, 8), dtype=torch.float32)
    output_query = torch.empty_like(query)
    output_key = torch.empty_like(key)

    query_out, key_out = rope(
        positions,
        query,
        key,
        output_q_rope=output_query,
        output_k_rope=output_key,
    )

    assert query_out.data_ptr() == output_query.data_ptr()
    assert key_out.data_ptr() == output_key.data_ptr()
    torch.testing.assert_close(query_out, output_query)
    torch.testing.assert_close(key_out, output_key)
    assert torch.isfinite(query_out).all()
    assert torch.isfinite(key_out).all()


def test_deepseek_scaling_rotary_embedding_gptj_flag_matches_checkpoint_layout() -> None:
    rope = DeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=False,
        scaling_factor=4.0,
        dtype=torch.float32,
        device="cpu",
    )
    positions = torch.tensor([0, 1], dtype=torch.long)
    query = torch.randn((2, 1, 8), dtype=torch.float32)
    key = torch.randn((2, 1, 8), dtype=torch.float32)

    query_out, key_out = rope(positions, query, key)

    cos, sin = rope.cos_sin_cache[positions].chunk(2, dim=-1)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    expected_query = _deepseek_checkpoint_rope(query, cos, sin)
    expected_key = _deepseek_checkpoint_rope(key, cos, sin)
    torch.testing.assert_close(query_out, expected_query)
    torch.testing.assert_close(key_out, expected_key)


def test_yarn_rope_with_mscale_fields_uses_deepseek_scaling() -> None:
    rope = get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=16,
        base=10000,
        is_neox_style=False,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        dtype=torch.float32,
    )

    assert isinstance(rope, DeepseekScalingRotaryEmbedding)
    assert rope.mscale == pytest.approx(1.0)
