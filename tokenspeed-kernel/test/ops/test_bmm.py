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

import pytest
import tokenspeed_kernel
import tokenspeed_kernel.ops.gemm as gemm_ops
import torch
from tokenspeed_kernel.selection import SelectedKernel


def _reference_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha: torch.Tensor | float | None = None,
) -> torch.Tensor:
    output = torch.bmm(a, b.transpose(1, 2))
    if alpha is not None:
        output = output * alpha
    if bias is not None:
        if bias.ndim == 1:
            bias = bias.view(1, 1, -1)
        else:
            bias = bias.view(bias.shape[0], 1, bias.shape[1])
        output = output + bias
    return output


def test_bmm_dense_reference_matches_torch() -> None:
    torch.manual_seed(0)
    a = torch.randn((3, 5, 7), dtype=torch.float32)
    b = torch.randn((3, 11, 7), dtype=torch.float32)

    actual = tokenspeed_kernel.bmm(a, b, override="torch_bmm")
    expected = _reference_bmm(a, b)
    torch.testing.assert_close(actual, expected)


def test_bmm_dense_reference_writes_strided_out() -> None:
    torch.manual_seed(0)
    heads, tokens, n, k = 3, 5, 11, 7
    a = torch.randn((heads, tokens, k), dtype=torch.float32)
    b = torch.randn((heads, n, k), dtype=torch.float32)
    bias = torch.randn((heads, n), dtype=torch.float32)
    alpha = torch.tensor(1.5, dtype=torch.float32)
    output_base = torch.empty((tokens, heads, n), dtype=torch.float32)
    out = output_base.transpose(0, 1)

    actual = tokenspeed_kernel.bmm(
        a,
        b,
        bias=bias,
        alpha=alpha,
        out=out,
        override="torch_bmm",
    )
    expected = _reference_bmm(a, b, bias=bias, alpha=alpha)

    assert actual is out
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(output_base, expected.transpose(0, 1))


def test_bmm_dense_rejects_batch_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((3, 16, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch mismatch"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_dense_rejects_rank2_weights() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"B with shape \[B, N, K\]"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_dense_rejects_bad_out_layout() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    out = torch.empty((2, 16, 4), dtype=torch.bfloat16).transpose(1, 2)
    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.bmm(a, b, out=out)


def test_bmm_dense_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.float32)
    b = torch.empty((2, 16, 8), dtype=torch.float32)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_bmm out= requires out_dtype"):
        tokenspeed_kernel.bmm(a, b, out=out, override="torch_bmm")


def test_bmm_forwards_pdl_to_opted_in_kernel(monkeypatch) -> None:
    torch.manual_seed(1)
    a = torch.randn((2, 4, 8), dtype=torch.float32)
    b = torch.randn((2, 16, 8), dtype=torch.float32)
    captured: dict[str, bool] = {}

    def pdl_kernel(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        assert A_scales is None
        assert B_scales is None
        assert alpha is None
        assert block_size is None
        assert out is None
        captured["enable_pdl"] = enable_pdl
        return _reference_bmm(A, B).to(out_dtype)

    def select_pdl_kernel(*args, **kwargs) -> SelectedKernel:
        return SelectedKernel("flashinfer_mm_nvfp4", pdl_kernel)

    monkeypatch.setattr(gemm_ops, "select_kernel", select_pdl_kernel)

    actual = tokenspeed_kernel.bmm(a, b, enable_pdl=True)
    expected = _reference_bmm(a, b)

    assert captured["enable_pdl"] is True
    torch.testing.assert_close(actual, expected)
