from __future__ import annotations

import pytest
import torch


def test_deepseek_scaling_rotary_embedding_exposes_native_forward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for rotary embedding platform detection")

    from tokenspeed.runtime.layers.rotary_embedding import (
        DeepseekScalingRotaryEmbedding,
    )

    rope = DeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=True,
        scaling_factor=1.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert rope._forward_method.__func__ is rope.forward_native.__func__

    positions = torch.arange(4, device="cuda", dtype=torch.int64).reshape(1, 4)
    query = torch.randn(1, 4, 1, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 4, 1, 8, device="cuda", dtype=torch.bfloat16)

    q_out, k_out = rope.forward_native(positions, query, key)

    assert q_out.shape == query.shape
    assert k_out.shape == key.shape
    assert q_out.dtype == query.dtype
    assert k_out.dtype == key.dtype
