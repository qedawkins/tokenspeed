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

import math
from types import SimpleNamespace

import pytest
import torch
from s1_component_utils import (
    assert_finite_close,
    assert_selected_amd_kernel_not_reference,
    extract_s1_text_model_dims,
    s1_prefill_token_counts,
    s1_varlen_prefill_sequences,
)
from tokenspeed_kernel.ops.attention import mla_prefill
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


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


def _cum_seq_lens(seq_lens: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(seq_lens, dtype=torch.int32).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )


def _make_prefill_case(
    *,
    seq_lens: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
):
    dims = _tiny_s1_dims()
    total = sum(seq_lens)
    query_dim = dims.D_nope + dims.D_rope

    torch.manual_seed(4321 + total)
    q = _to_dtype(
        torch.randn(total, dims.A_r, query_dim, device=device, dtype=torch.bfloat16)
        * 0.125,
        dtype,
    )
    k = _to_dtype(
        torch.randn(total, dims.A_r, query_dim, device=device, dtype=torch.bfloat16)
        * 0.125,
        dtype,
    )
    v = _to_dtype(
        torch.randn(total, dims.A_r, dims.D_v, device=device, dtype=torch.bfloat16)
        * 0.125,
        dtype,
    )
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    cum = _cum_seq_lens(seq_lens, device)
    return dims, q, k, v, seq_lens_t, cum


def _reference_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens_q: tuple[int, ...],
    seq_lens_kv: tuple[int, ...],
    softmax_scale: float,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lses = []
    q_offset = 0
    kv_offset = 0
    group_size = q.shape[1] // k.shape[1]
    for cur_q_len, cur_kv_len in zip(seq_lens_q, seq_lens_kv):
        cur_q = q[q_offset : q_offset + cur_q_len].float()
        cur_k = k[kv_offset : kv_offset + cur_kv_len].float()
        cur_v = v[kv_offset : kv_offset + cur_kv_len].float()
        seq_out = []
        seq_lse = []
        for q_head in range(q.shape[1]):
            kv_head = q_head // group_size
            scores = torch.matmul(cur_q[:, q_head], cur_k[:, kv_head].T)
            scores = scores * softmax_scale
            if is_causal:
                q_idx = torch.arange(cur_q_len, device=q.device).view(-1, 1)
                k_idx = torch.arange(cur_kv_len, device=q.device).view(1, -1)
                offset = cur_kv_len - cur_q_len
                scores = scores.masked_fill(k_idx > q_idx + offset, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            seq_out.append(torch.matmul(probs, cur_v[:, kv_head]))
            seq_lse.append(torch.logsumexp(scores, dim=-1) * math.log2(math.e))
        outputs.append(torch.stack(seq_out, dim=1))
        lses.append(torch.stack(seq_lse, dim=1))
        q_offset += cur_q_len
        kv_offset += cur_kv_len
    return torch.cat(outputs, dim=0).to(torch.bfloat16), torch.cat(lses, dim=0)


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon MLA prefill component test")


PREFILL_SEQUENCES = (*s1_varlen_prefill_sequences(), s1_prefill_token_counts())


@pytest.mark.parametrize("seq_lens", PREFILL_SEQUENCES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_gluon_mla_prefill_matches_reference_bf16(
    device: str, seq_lens: tuple[int, ...], is_causal: bool
) -> None:
    _require_cdna4_gpu()
    dims, q, k, v, seq_lens_t, cum = _make_prefill_case(
        seq_lens=seq_lens,
        dtype=torch.bfloat16,
        device=device,
    )
    softmax_scale = 1.0 / (dims.D_nope + dims.D_rope) ** 0.5

    actual, actual_lse = mla_prefill(
        q,
        k,
        v,
        seq_lens_t,
        cum,
        max(seq_lens),
        batch_size=len(seq_lens),
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        return_lse=True,
    )
    torch.cuda.synchronize()

    expected, expected_lse = _reference_mla_prefill(
        q,
        k,
        v,
        seq_lens,
        seq_lens,
        softmax_scale,
        is_causal=is_causal,
    )
    assert actual.shape == (sum(seq_lens), dims.A_r, dims.D_v)
    assert actual.dtype == torch.bfloat16
    assert actual_lse.shape == (sum(seq_lens), dims.A_r)
    assert actual_lse.dtype == torch.float32
    assert_finite_close(actual, expected, atol=0.04, rtol=0.02, name="mla_prefill")
    assert_finite_close(actual_lse, expected_lse, atol=0.04, rtol=0.02, name="lse")


def test_gluon_mla_prefill_matches_reference_fp8(device: str) -> None:
    _require_cdna4_gpu()
    seq_lens = (17, 9, 12)
    dims, q, k, v, seq_lens_t, cum = _make_prefill_case(
        seq_lens=seq_lens,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    softmax_scale = 1.0 / (dims.D_nope + dims.D_rope) ** 0.5

    actual, actual_lse = mla_prefill(
        q,
        k,
        v,
        seq_lens_t,
        cum,
        max(seq_lens),
        batch_size=len(seq_lens),
        softmax_scale=softmax_scale,
        is_causal=True,
        return_lse=True,
    )
    torch.cuda.synchronize()

    expected, expected_lse = _reference_mla_prefill(
        q,
        k,
        v,
        seq_lens,
        seq_lens,
        softmax_scale,
        is_causal=True,
    )
    assert actual.dtype == torch.bfloat16
    assert_finite_close(actual, expected, atol=0.12, rtol=0.04, name="fp8_prefill")
    assert_finite_close(actual_lse, expected_lse, atol=0.12, rtol=0.04, name="fp8_lse")


def test_gluon_mla_prefill_accepts_preallocated_output(device: str) -> None:
    _require_cdna4_gpu()
    seq_lens = (1,)
    dims, q, k, v, seq_lens_t, cum = _make_prefill_case(
        seq_lens=seq_lens,
        dtype=torch.bfloat16,
        device=device,
    )
    out = torch.empty(sum(seq_lens), dims.A_r, dims.D_v, device=device, dtype=torch.bfloat16)
    softmax_scale = 1.0 / (dims.D_nope + dims.D_rope) ** 0.5

    actual = mla_prefill(
        q,
        k,
        v,
        seq_lens_t,
        cum,
        max(seq_lens),
        batch_size=len(seq_lens),
        softmax_scale=softmax_scale,
        is_causal=True,
        return_lse=False,
        out=out,
    )
    torch.cuda.synchronize()

    expected, _ = _reference_mla_prefill(
        q,
        k,
        v,
        seq_lens,
        seq_lens,
        softmax_scale,
        is_causal=True,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert_finite_close(out, expected, atol=0.04, rtol=0.02, name="preallocated_out")


def test_amd_mla_prefill_selection_uses_gluon(mi350_platform) -> None:
    dims = _tiny_s1_dims()
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
            "is_causal": True,
            "return_lse": True,
        },
        expected_solution="gluon",
    )
    assert selected.name == "gluon_mla_prefill_gfx950"
