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
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TokenSpeed kernel selection imports require GPU platform detection",
)


def _w8a8_signature():
    from tokenspeed_kernel.signature import (
        ScaleFormat,
        dense_tensor_format,
        format_signature,
        tensor_format,
    )

    fp8_dtype = getattr(torch, "float8_e4m3fn", None) or getattr(
        torch,
        "float8_e4m3fnuz",
    )
    return format_signature(
        x=dense_tensor_format(torch.bfloat16),
        weight=tensor_format(
            "scaled-fp8",
            fp8_dtype,
            scale=ScaleFormat(storage_dtype=torch.float32, granularity="channel"),
        ),
    )


def _mxfp4_signature():
    from tokenspeed_kernel.signature import (
        ScaleFormat,
        dense_tensor_format,
        format_signature,
        tensor_format,
    )

    return format_signature(
        x=dense_tensor_format(torch.bfloat16),
        weight=tensor_format(
            "mxfp4",
            torch.uint8,
            scale=ScaleFormat(
                storage_dtype=torch.uint8,
                granularity="block",
                block_shape=(32,),
            ),
        ),
    )


def test_w4a8_quark_experts_signature_is_distinct_from_w8a8_and_mxfp4() -> None:
    from tokenspeed_kernel.ops.moe import (
        WEIGHT_W4A8_QUARK,
        w4a8_quark_experts_format_signature,
    )

    signature = w4a8_quark_experts_format_signature(torch.bfloat16)
    weight_format = signature.format_for("weight")
    activation_format = signature.format_for("x")

    assert activation_format is not None
    assert weight_format is not None
    assert weight_format.scale is not None
    assert activation_format.format == "dense"
    assert activation_format.storage_dtype == torch.bfloat16
    assert weight_format.format == WEIGHT_W4A8_QUARK
    assert weight_format.storage_dtype == torch.uint8
    assert weight_format.scale.storage_dtype == torch.float32
    assert weight_format.scale.granularity == "channel"
    assert signature != _w8a8_signature()
    assert signature != _mxfp4_signature()


def test_w4a8_quark_selection_fails_with_missing_kernel_message() -> None:
    from tokenspeed_kernel.ops.moe import (
        W4A8_QUARK_MISSING_KERNEL_MESSAGE,
        select_w4a8_quark_experts_kernel,
    )
    from tokenspeed_kernel.selection import NoKernelFoundError

    with pytest.raises(
        NoKernelFoundError,
        match="Quark W4A8 MoE expert kernel is not registered",
    ) as exc_info:
        select_w4a8_quark_experts_kernel(
            dtype=torch.bfloat16,
            features={"dispatch_sorted"},
        )

    message = str(exc_info.value)
    assert W4A8_QUARK_MISSING_KERNEL_MESSAGE in message
    assert "quark-w4a8-int4" in message
    assert "scaled-fp8" not in message
    assert "mxfp4" not in message


def test_existing_w8a8_and_mxfp4_selection_contracts_still_resolve() -> None:
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    w8a8 = select_kernel(
        "moe",
        "experts",
        _w8a8_signature(),
        features=frozenset({"dispatch_sorted"}),
    )
    assert w8a8.name == "gluon_fp8_local_experts_gfx950"

    route = select_kernel(
        "moe",
        "route",
        format_signature(logits=dense_tensor_format(torch.bfloat16)),
        traits={"output_type": "ragged_metadata"},
    )
    dispatch_gemm = select_kernel(
        "moe",
        "experts",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=frozenset({"ragged_metadata", "dispatch_gemm"}),
    )

    assert route.name == "triton_kernels_routing"
    assert dispatch_gemm.name == "triton_kernels_dispatch_gemm"
