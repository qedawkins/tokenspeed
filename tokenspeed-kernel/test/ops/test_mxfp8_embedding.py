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

"""MXFP8 embedding gathers from Engram layer slices without copying IDs."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.embedding import mxfp8_embedding
from tokenspeed_kernel.ops.embedding.triton import _mxfp8_embedding_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _table() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(83)
    weight = torch.randn((64, 256), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(
        118, 132, (64, 8), device="cuda", dtype=torch.uint8, generator=generator
    )
    return weight, scales


def _indices(rows: int, layout: str, dtype: torch.dtype) -> torch.Tensor:
    if layout == "layer":
        ids = torch.arange(rows * 2 * 24, device="cuda", dtype=dtype).reshape(
            rows, 2, 24
        )[:, 1, :]
    elif layout == "transpose":
        ids = torch.arange(rows * 24, device="cuda", dtype=dtype).reshape(24, rows).t()
    elif layout == "columns":
        ids = torch.arange(rows * 48, device="cuda", dtype=dtype).reshape(rows, 48)[
            :, ::2
        ]
    elif layout == "broadcast":
        return (torch.arange(24, device="cuda", dtype=dtype) * 4).expand(rows, -1)
    else:
        ids = torch.arange(rows * 24, device="cuda", dtype=dtype).reshape(rows, 2, 12)
    # Update in place to retain the sliced/strided layout.
    ids.copy_(ids % 90)
    return ids


def _reference(weight, scales, ids):
    local = (ids >= 17) & (ids < 70)
    offset = (ids.long() - 17).clamp(0, 63)
    codes = weight.view(torch.uint8)[offset].view(torch.float8_e4m3fn).float()
    factors = (
        scales[offset].view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=-1)
    )
    return (codes * factors).to(torch.bfloat16).masked_fill(~local[..., None], 0)


@pytest.mark.parametrize("rows", [0, 6, 12, 24, 48, 96, 192, 1031])
@pytest.mark.parametrize(
    "layout", ["layer", "transpose", "columns", "broadcast", "contiguous"]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_embedding_reads_strided_ids(rows, layout, dtype):
    weight, scales = _table()
    ids = _indices(rows, layout, dtype)
    actual = mxfp8_embedding(weight, scales, ids, 17, 70)
    expected = _reference(weight, scales, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


def test_embedding_replays_updated_layer_slice():
    weight, scales = _table()
    ids = _indices(6, "layer", torch.int64)
    mxfp8_embedding(weight, scales, ids, 17, 70)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mxfp8_embedding(weight, scales, ids, 17, 70)
    for shift in (1, 3, 11):
        ids.copy_((ids + shift) % 90)
        graph.replay()
        torch.testing.assert_close(
            captured, _reference(weight, scales, ids), rtol=0, atol=0
        )


def test_embedding_id_strides_do_not_specialize():
    weight, scales = _table()
    ids = torch.arange(48, device="cuda", dtype=torch.int64).reshape(2, 24)
    storage = torch.zeros((12, 3, 48), device="cuda", dtype=torch.int64)
    first = storage[:2, 1, :24]
    first.copy_(ids)
    mxfp8_embedding(weight, scales, first, 17, 70)
    second = storage[:, 2, ::2].t()
    with assert_no_triton_compile(_mxfp8_embedding_kernel):
        actual = mxfp8_embedding(weight, scales, second, 17, 70)
    torch.testing.assert_close(
        actual, _reference(weight, scales, second), rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", [2048, 2049, 65536])
def test_embedding_strided_large_row_dispatch(rows):
    weight, scales = _table()
    ids = torch.arange(rows * 2, device="cuda", dtype=torch.int64).reshape(rows, 2)[
        :, 1:2
    ]
    ids.copy_(ids % 90)
    actual = mxfp8_embedding(weight, scales, ids, 17, 70)
    torch.testing.assert_close(actual, _reference(weight, scales, ids), rtol=0, atol=0)
