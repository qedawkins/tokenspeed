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
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import SelectedKernel


def _copy_out_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert A_scales is None
    assert B_scales is None
    assert block_size is None
    output = A @ B.T
    if alpha is not None:
        output = output * alpha.to(dtype=output.dtype)
    output = output.to(out_dtype)
    if out is not None:
        out.copy_(output)
        return out
    return output


def _require_gpu(vendor: str | None = None) -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    platform = current_platform()
    if vendor is not None and platform.vendor != vendor:
        pytest.skip(f"{vendor} GPU required")


def test_mm_dense_reference_writes_strided_out() -> None:
    torch.manual_seed(0)
    m, n, k = 5, 11, 7
    a = torch.randn((m, k), dtype=torch.float32)
    b = torch.randn((n, k), dtype=torch.float32)
    bias = torch.randn((n,), dtype=torch.float32)
    output_base = torch.empty((m, n + 3), dtype=torch.float32)
    out = output_base[:, :n]

    actual = tokenspeed_kernel.mm(a, b, bias=bias, out=out, override="torch_mm")
    expected = torch.nn.functional.linear(a, b, bias)

    assert actual is out
    torch.testing.assert_close(out, expected)


def test_mm_dense_rejects_bad_out_layout() -> None:
    a = torch.empty((4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)
    out = torch.empty((16, 4), dtype=torch.bfloat16).transpose(0, 1)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.mm(a, b, out=out)


def test_mm_dense_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((4, 8), dtype=torch.float32)
    b = torch.empty((16, 8), dtype=torch.float32)
    out = torch.empty((4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_mm out= requires out_dtype"):
        tokenspeed_kernel.mm(a, b, out=out, override="torch_mm")


def test_mm_non_native_out_kernel_copies_to_out(monkeypatch) -> None:
    torch.manual_seed(1)
    a = torch.randn((4, 8), dtype=torch.float32)
    b = torch.randn((16, 8), dtype=torch.float32)
    out = torch.empty((4, 16), dtype=torch.float32)

    def select_copy_out_kernel(*args, **kwargs) -> SelectedKernel:
        return SelectedKernel("test_mm_copy_out_kernel", _copy_out_kernel)

    monkeypatch.setattr(gemm_ops, "select_kernel", select_copy_out_kernel)

    actual = tokenspeed_kernel.mm(a, b, out=out, override="test_mm_copy_out_kernel")
    expected = a @ b.T

    assert actual is out
    torch.testing.assert_close(out, expected)


def test_gluon_dense16_fallback_writes_strided_out(monkeypatch) -> None:
    _require_gpu("amd")
    from tokenspeed_kernel.ops.gemm import gluon as gluon_gemm

    if not hasattr(gluon_gemm, "gluon_mm_a16w16_gfx950"):
        pytest.skip("Gluon dense16 wrapper is not registered")

    def skip_dense16_impl(*args, **kwargs):
        return None

    monkeypatch.setattr(gluon_gemm, "_dense16_impl", skip_dense16_impl)

    torch.manual_seed(2)
    m, n, k = 3, 128, 64
    dtype = torch.bfloat16
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25
    alpha = torch.tensor(1.5, device="cuda", dtype=torch.float32)
    backing = torch.empty((m, n + 17), device="cuda", dtype=dtype)
    out = backing[:, :n]

    actual = gluon_gemm.gluon_mm_a16w16_gfx950(
        a,
        b,
        None,
        None,
        dtype,
        alpha=alpha,
        out=out,
    )
    expected = torch.mm(a, b.T) * alpha.to(dtype=dtype)

    assert actual is out
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


def test_triton_fp8_blockscale_writes_strided_out() -> None:
    _require_gpu()
    from tokenspeed_kernel.ops.gemm import triton as triton_gemm

    torch.manual_seed(3)
    m, n, k = 7, 73, 128
    fp8_dtype = current_platform().fp8e4m3fn.dtype
    a = (torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.25).to(fp8_dtype)
    b = (torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.25).to(fp8_dtype)
    a_scales = torch.rand((m, 1), device="cuda", dtype=torch.float32) + 0.5
    b_scales = torch.rand((1, 1), device="cuda", dtype=torch.float32) + 0.5
    out_dtype = torch.bfloat16

    expected = triton_gemm.triton_mm_fp8_blockscale(
        a,
        b,
        a_scales,
        b_scales,
        out_dtype,
        block_size=[128, 128],
    )
    backing = torch.full((m, n + 17), -123, device="cuda", dtype=out_dtype)
    out = backing[:, :n]

    actual = triton_gemm.triton_mm_fp8_blockscale(
        a,
        b,
        a_scales,
        b_scales,
        out_dtype,
        block_size=[128, 128],
        out=out,
    )

    assert actual is out
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        backing[:, n:],
        torch.full_like(backing[:, n:], -123),
    )


def test_triton_mxfp4_writes_strided_out() -> None:
    _require_gpu("amd")
    from tokenspeed_kernel.ops.gemm import triton as triton_gemm

    torch.manual_seed(4)
    m, n, k = 5, 37, 64
    a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    a_scales = torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8)
    b_scales = torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8)
    out_dtype = torch.bfloat16

    expected = triton_gemm.triton_mm_mxfp4(
        a,
        b,
        a_scales,
        b_scales,
        out_dtype,
    )
    backing = torch.full((m, n + 17), -123, device="cuda", dtype=out_dtype)
    out = backing[:, :n]

    actual = triton_gemm.triton_mm_mxfp4(
        a,
        b,
        a_scales,
        b_scales,
        out_dtype,
        out=out,
    )

    assert actual is out
    torch.testing.assert_close(out, expected, equal_nan=True)
    torch.testing.assert_close(
        backing[:, n:],
        torch.full_like(backing[:, n:], -123),
    )


def test_triton_fp8_scaled_writes_strided_out() -> None:
    _require_gpu("nvidia")
    from tokenspeed_kernel.ops.gemm import triton as triton_gemm

    torch.manual_seed(5)
    m, n, k = 7, 41, 64
    fp8_dtype = current_platform().fp8e4m3fn.dtype
    a = (torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.25).to(fp8_dtype)
    b = (torch.randn((k, n), device="cuda", dtype=torch.float32) * 0.25).to(fp8_dtype)
    a_scales = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    b_scales = torch.tensor([1.25], device="cuda", dtype=torch.float32)
    out_dtype = torch.bfloat16

    expected = triton_gemm.triton_mm_fp8_scaled(
        a,
        b,
        a_scales,
        b_scales,
        out_dtype,
    )
    backing = torch.full((m, n + 17), -123, device="cuda", dtype=out_dtype)
    out = backing[:, :n]

    actual = triton_gemm.triton_mm_fp8_scaled(
        a,
        b,
        a_scales,
        b_scales,
        out_dtype,
        out=out,
    )

    assert actual is out
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        backing[:, n:],
        torch.full_like(backing[:, n:], -123),
    )
