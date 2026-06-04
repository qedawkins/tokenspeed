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
    s1_decode_token_counts,
)
from tokenspeed_kernel.ops.attention import mla_decode_with_kvcache
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


PAGE_SIZE = 64
MAX_SEQ_LEN = 65


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


def _to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.clamp(-2.0, 2.0).to(dtype)
    return tensor.to(dtype)


def _make_decode_case(
    *,
    decode_rows: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    device: str,
    noncontiguous_q: bool = False,
):
    dims = _tiny_s1_dims()
    query_dim = dims.R_kv + dims.D_rope
    max_pages = 2
    num_pages = max(max_pages, decode_rows * max_pages)

    torch.manual_seed(1234 + decode_rows)
    if noncontiguous_q:
        q_source = _to_dtype(
            torch.randn(
                decode_rows,
                dims.A_r,
                query_dim * 2,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.125,
            q_dtype,
        )
        q = q_source[..., ::2]
        assert not q.is_contiguous() or q.numel() == 0
    else:
        q = _to_dtype(
            torch.randn(
                decode_rows,
                dims.A_r,
                query_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.125,
            q_dtype,
        )

    kv_cache = _to_dtype(
        torch.randn(
            num_pages,
            PAGE_SIZE,
            query_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125,
        kv_dtype,
    )
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).reshape(
        decode_rows if decode_rows else 1, max_pages
    )[:decode_rows]
    cache_seqlens = torch.full(
        (decode_rows,), MAX_SEQ_LEN, device=device, dtype=torch.int32
    )
    return dims, q, kv_cache, page_table, cache_seqlens


def _reference_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    value_head_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    output = torch.empty(
        (q.shape[0], q.shape[1], value_head_dim), device=q.device, dtype=out_dtype
    )
    for row in range(q.shape[0]):
        seq_len = int(cache_seqlens[row].item())
        if seq_len == 0:
            output[row].zero_()
            continue
        rows = []
        for token in range(seq_len):
            page_id = token // PAGE_SIZE
            token_slot = token % PAGE_SIZE
            rows.append(kv_cache[page_table[row, page_id], token_slot])
        kv_rows = torch.stack(rows).float()
        scores = torch.einsum("hd,sd->hs", q[row].float(), kv_rows) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        values = kv_rows[:, :value_head_dim]
        output[row] = torch.einsum("hs,sv->hv", probs, values).to(out_dtype)
    return output


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon MLA decode component test")


@pytest.mark.parametrize("decode_rows", s1_decode_token_counts())
def test_gluon_mla_decode_matches_reference_bf16(
    device: str, decode_rows: int
) -> None:
    _require_cdna4_gpu()
    dims, q, kv_cache, page_table, cache_seqlens = _make_decode_case(
        decode_rows=decode_rows,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        device=device,
    )
    assert cache_seqlens.numel() == 0 or int(cache_seqlens.max()) > PAGE_SIZE

    softmax_scale = 1.0 / (dims.R_kv + dims.D_rope) ** 0.5
    actual = mla_decode_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        MAX_SEQ_LEN,
        kv_lora_rank=dims.R_kv,
        qk_rope_head_dim=dims.D_rope,
        value_head_dim=dims.D_v,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    expected = _reference_mla_decode(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        value_head_dim=dims.D_v,
        softmax_scale=softmax_scale,
    )
    assert actual.shape == (decode_rows, dims.A_r, dims.D_v)
    assert actual.dtype == torch.bfloat16
    assert_finite_close(actual, expected, atol=0.04, rtol=0.02, name="mla_decode")


def test_gluon_mla_decode_accepts_noncontiguous_query(device: str) -> None:
    _require_cdna4_gpu()
    dims, q, kv_cache, page_table, cache_seqlens = _make_decode_case(
        decode_rows=8,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        device=device,
        noncontiguous_q=True,
    )
    softmax_scale = 1.0 / (dims.R_kv + dims.D_rope) ** 0.5

    actual = mla_decode_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        MAX_SEQ_LEN,
        kv_lora_rank=dims.R_kv,
        qk_rope_head_dim=dims.D_rope,
        value_head_dim=dims.D_v,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    expected = _reference_mla_decode(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        value_head_dim=dims.D_v,
        softmax_scale=softmax_scale,
    )
    assert_finite_close(actual, expected, atol=0.04, rtol=0.02, name="strided_query")


def test_gluon_mla_decode_matches_reference_fp8(device: str) -> None:
    _require_cdna4_gpu()
    dims, q, kv_cache, page_table, cache_seqlens = _make_decode_case(
        decode_rows=8,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        device=device,
    )
    softmax_scale = 1.0 / (dims.R_kv + dims.D_rope) ** 0.5

    actual = mla_decode_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        MAX_SEQ_LEN,
        kv_lora_rank=dims.R_kv,
        qk_rope_head_dim=dims.D_rope,
        value_head_dim=dims.D_v,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    expected = _reference_mla_decode(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        value_head_dim=dims.D_v,
        softmax_scale=softmax_scale,
    )
    assert actual.dtype == torch.bfloat16
    assert_finite_close(actual, expected, atol=0.12, rtol=0.04, name="fp8_mla_decode")


def test_amd_mla_decode_selection_uses_gluon(mi350_platform) -> None:
    dims = _tiny_s1_dims()
    selected = assert_selected_amd_kernel_not_reference(
        "attention",
        "mla_decode_with_kvcache",
        platform=mi350_platform,
        format_signature=format_signature(
            q=dense_tensor_format(torch.bfloat16),
            kv_cache=dense_tensor_format(torch.bfloat16),
        ),
        features=frozenset({"paged", "mla"}),
        traits={
            "num_q_heads": dims.A_r,
            "query_dim": dims.R_kv + dims.D_rope,
            "kv_lora_rank": dims.R_kv,
            "qk_rope_head_dim": dims.D_rope,
            "value_head_dim": dims.D_v,
            "page_size": PAGE_SIZE,
        },
        expected_solution="gluon",
    )
    assert selected.name == "gluon_mla_decode_gfx950"
