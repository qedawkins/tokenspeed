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

"""Initial mHC streams preserve embeddings exactly in eager and graph forwards."""

import pytest
import torch
from tokenspeed_kernel.ops.residual import mhc_init
from tokenspeed_kernel.ops.residual.triton import _mhc_init_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _check(hidden, actual, hc_mult):
    residual, pre = actual
    expected = hidden[:, None, :].repeat(1, hc_mult, 1)
    expected_pre = torch.zeros(hidden.shape[0], hc_mult, device=hidden.device)
    expected_pre[:, 0] = 1
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
    assert residual.is_contiguous() and pre.is_contiguous()
    # Streams must be independent, including when the source was an expanded view.
    if hidden.shape[0] and hc_mult > 1:
        untouched = residual[:, 1].clone()
        residual[:, 0].zero_()
        torch.testing.assert_close(residual[:, 1], untouched, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 6, 12, 24, 48, 96, 192, 2049])
@pytest.mark.parametrize("width", [129, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_mhc_init(rows, width, dtype, hc_mult):
    hidden = torch.randn(rows, width, device="cuda", dtype=dtype)
    _check(hidden, mhc_init(hidden, hc_mult), hc_mult)


@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
def test_mhc_init_strided(layout):
    if layout == "transpose":
        hidden = torch.randn(5120, 24, device="cuda", dtype=torch.bfloat16).T
    elif layout == "slice":
        hidden = torch.randn(48, 10240, device="cuda", dtype=torch.bfloat16)[::2, ::2]
    else:
        hidden = torch.randn(1, 5120, device="cuda", dtype=torch.bfloat16).expand(
            24, -1
        )
    _check(hidden, mhc_init(hidden, 4), 4)


def test_mhc_init_runtime_rows_and_strides_do_not_compile():
    # Warm once, then vary row count and both embedding strides.
    hidden = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
    mhc_init(hidden, 4)
    with assert_no_triton_compile(_mhc_init_kernel):
        for rows in [12, 24, 48, 96, 192]:
            hidden = torch.randn(rows * 2, 10240, device="cuda", dtype=torch.bfloat16)[
                ::2, ::2
            ]
            _check(hidden, mhc_init(hidden, 4), 4)


def test_mhc_init_graph_replay_then_eager():
    hidden = torch.randn(24, 5120, device="cuda", dtype=torch.bfloat16)
    mhc_init(hidden, 4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_init(hidden, 4)
    hidden.normal_()
    graph.replay()
    # Eager outputs survive another graph replay and own independent storage.
    eager = mhc_init(hidden, 4)
    expected = hidden.clone()
    hidden.normal_()
    graph.replay()
    _check(expected, eager, 4)
    _check(hidden, captured, 4)
