import torch

from tokenspeed.runtime.layers.rotary_embedding import DeepseekScalingRotaryEmbedding


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
