# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from s1_component_utils import (
    assert_finite_close,
    assert_selected_amd_kernel_not_reference,
    extract_s1_text_model_dims,
)
from tokenspeed_kernel.ops.attention import (
    MLAPrefixChunk,
    mla_chunked_prefill,
    mla_prefill,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


PAGE_SIZE = 64


def _tiny_s1_dims():
    config = SimpleNamespace(
        hidden_size=1024,
        n_routed_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        num_attention_heads=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=8,
        v_head_dim=32,
        kv_lora_rank=64,
    )
    return extract_s1_text_model_dims(config, tp_size=4)


def _cum_seq_lens(seq_lens: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(seq_lens, dtype=torch.int32).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for MLA chunked-prefill component tests")


def _slice_single_sequence(
    tensor: torch.Tensor,
    *,
    prefix_len: int,
    chunk_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = tensor[:prefix_len]
    current = tensor[prefix_len : prefix_len + chunk_len]
    full = tensor[: prefix_len + chunk_len]
    return prefix, current, full


def _prefix_chunks(
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    *,
    chunk_size: int,
    device: str,
) -> list[MLAPrefixChunk]:
    chunks = []
    for start in range(0, k_prefix.shape[0], chunk_size):
        end = min(start + chunk_size, k_prefix.shape[0])
        seq_lens = (end - start,)
        chunks.append(
            MLAPrefixChunk(
                k=k_prefix[start:end],
                v=v_prefix[start:end],
                seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
                cum_seq_lens=_cum_seq_lens(seq_lens, device),
                max_seq_len=max(seq_lens),
            )
        )
    return chunks


@pytest.mark.parametrize(
    "prefix_len,chunk_len,prefix_chunk_size",
    [
        (64, 32, 32),
        (64, 64, 64),
        (64, 80, 80),
    ],
)
def test_mla_chunked_prefill_matches_non_chunked_prefill(
    device: str,
    prefix_len: int,
    chunk_len: int,
    prefix_chunk_size: int,
) -> None:
    _require_cdna4_gpu()
    dims = _tiny_s1_dims()
    query_dim = dims.D_nope + dims.D_rope
    total_len = prefix_len + chunk_len
    assert prefix_len % PAGE_SIZE == 0

    torch.manual_seed(9000 + chunk_len)
    q_full = (
        torch.randn(total_len, dims.A_r, query_dim, device=device, dtype=torch.bfloat16)
        * 0.125
    )
    k_full = (
        torch.randn(total_len, dims.A_r, query_dim, device=device, dtype=torch.bfloat16)
        * 0.125
    )
    v_full = (
        torch.randn(total_len, dims.A_r, dims.D_v, device=device, dtype=torch.bfloat16)
        * 0.125
    )
    _, q_current, _ = _slice_single_sequence(
        q_full, prefix_len=prefix_len, chunk_len=chunk_len
    )
    k_prefix, k_current, k_visible = _slice_single_sequence(
        k_full, prefix_len=prefix_len, chunk_len=chunk_len
    )
    v_prefix, v_current, v_visible = _slice_single_sequence(
        v_full, prefix_len=prefix_len, chunk_len=chunk_len
    )

    chunk_seq_lens = (chunk_len,)
    visible_seq_lens = (total_len,)
    chunk_seq_lens_t = torch.tensor(chunk_seq_lens, device=device, dtype=torch.int32)
    visible_seq_lens_t = torch.tensor(visible_seq_lens, device=device, dtype=torch.int32)
    chunk_cum = _cum_seq_lens(chunk_seq_lens, device)
    visible_cum = _cum_seq_lens(visible_seq_lens, device)
    softmax_scale = 1.0 / query_dim**0.5

    chunked_out, chunked_lse = mla_chunked_prefill(
        q_current,
        k_current,
        v_current,
        chunk_seq_lens_t,
        chunk_cum,
        chunk_len,
        batch_size=1,
        softmax_scale=softmax_scale,
        prefix_chunks=_prefix_chunks(
            k_prefix,
            v_prefix,
            chunk_size=prefix_chunk_size,
            device=device,
        ),
        return_lse=True,
    )
    expected_out, expected_lse = mla_prefill(
        q_current,
        k_visible,
        v_visible,
        visible_seq_lens_t,
        visible_cum,
        total_len,
        batch_size=1,
        softmax_scale=softmax_scale,
        is_causal=True,
        return_lse=True,
        cum_seq_lens_q=chunk_cum,
        max_seq_len_q=chunk_len,
    )
    torch.cuda.synchronize()

    assert chunked_out.shape == (chunk_len, dims.A_r, dims.D_v)
    assert chunked_lse.shape == (chunk_len, dims.A_r)
    assert_finite_close(
        chunked_out, expected_out, atol=0.04, rtol=0.02, name="chunked_prefill"
    )
    assert_finite_close(
        chunked_lse, expected_lse, atol=0.04, rtol=0.02, name="chunked_lse"
    )


def test_mla_chunked_prefill_selection_reuses_gluon_prefill(mi350_platform) -> None:
    dims = _tiny_s1_dims()
    for is_causal in (True, False):
        selected = assert_selected_amd_kernel_not_reference(
            "attention",
            "mla_prefill",
            platform=mi350_platform,
            format_signature=format_signature(
                q=dense_tensor_format(torch.bfloat16),
                k=dense_tensor_format(torch.bfloat16),
                v=dense_tensor_format(torch.bfloat16),
            ),
            features=frozenset({"mla"}),
            traits={
                "num_q_heads": dims.A_r,
                "num_kv_heads": dims.A_r,
                "query_dim": dims.D_nope + dims.D_rope,
                "value_head_dim": dims.D_v,
                "is_causal": is_causal,
                "return_lse": True,
            },
            expected_solution="gluon",
        )
        assert selected.name == "gluon_mla_prefill_gfx950"
