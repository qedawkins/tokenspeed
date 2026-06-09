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
import torch
from s1_component_utils import (
    assert_finite_close,
    assert_selected_amd_kernel_not_reference,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon MoE combine test")


def _call_combine(
    expert_output: torch.Tensor,
    out: torch.Tensor,
    routed_scaling_factor: float,
    *,
    num_tokens_trait: int | None = None,
) -> None:
    traits = {"comm_strategy": None}
    if num_tokens_trait is not None:
        traits["num_tokens"] = num_tokens_trait
    tokenspeed_kernel.moe_combine(
        expert_output,
        out,
        routed_scaling_factor,
        dtype=expert_output.dtype,
        traits=traits,
        expected_kernel_name="gluon_local_sum_reduce_gfx950",
    )


def _weighted_slots(
    num_tokens: int,
    topk: int,
    hidden: int,
    *,
    device: str,
    dtype: torch.dtype,
    include_invalid: bool,
) -> torch.Tensor:
    if num_tokens == 0:
        return torch.empty((0, topk, hidden), device=device, dtype=dtype)

    values = (
        torch.arange(num_tokens * topk * hidden, device=device, dtype=torch.float32)
        .view(num_tokens, topk, hidden)
        .remainder(17)
        - 8
    )
    weights = torch.linspace(
        0.0,
        1.0,
        steps=max(1, num_tokens * topk),
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, topk)
    if include_invalid and num_tokens > 0:
        valid = torch.ones((num_tokens, topk), device=device, dtype=torch.float32)
        valid[::2, -1] = 0.0
        if topk > 1:
            valid[-1, 0] = 0.0
        weights = weights * valid
    return (values * weights.unsqueeze(-1)).to(dtype)


@pytest.mark.parametrize(
    ("num_tokens", "topk", "hidden"),
    [
        (0, 1, 16),
        (1, 1, 16),
        (8, 2, 32),
        (32, 8, 64),
        (257, 8, 32),
    ],
)
def test_gluon_local_combine_matches_weighted_slot_reference(
    device: str,
    num_tokens: int,
    topk: int,
    hidden: int,
) -> None:
    _require_cdna4_gpu()
    expert_output = _weighted_slots(
        num_tokens,
        topk,
        hidden,
        device=device,
        dtype=torch.bfloat16,
        include_invalid=True,
    )
    out = torch.empty((num_tokens, hidden), device=device, dtype=torch.bfloat16)

    _call_combine(expert_output, out, 1.0)
    torch.cuda.synchronize()

    expected = expert_output.float().sum(dim=1).to(torch.bfloat16)
    assert_finite_close(out, expected, atol=0.0, rtol=0.0)


def test_gluon_local_combine_accumulates_fp32_before_scaling(device: str) -> None:
    _require_cdna4_gpu()
    torch.manual_seed(7201)
    num_tokens = 8
    topk = 8
    hidden = 96
    expert_output = (
        torch.randn(num_tokens, topk, hidden, device=device, dtype=torch.float32) * 0.25
    ).to(torch.bfloat16)
    expert_output[:, 3, :] = 0.0
    out = torch.empty((num_tokens, hidden), device=device, dtype=torch.bfloat16)
    routed_scaling_factor = 0.75

    _call_combine(
        expert_output,
        out,
        routed_scaling_factor,
        num_tokens_trait=num_tokens,
    )
    torch.cuda.synchronize()

    expected = (expert_output.float().sum(dim=1) * routed_scaling_factor).to(
        torch.bfloat16
    )
    assert_finite_close(out, expected, atol=0.0, rtol=0.0)


def test_amd_local_combine_small_selection_uses_gluon(mi350_platform) -> None:
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "combine",
        platform=mi350_platform,
        format_signature=format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"num_tokens": 8, "comm_strategy": None},
        expected_solution="gluon",
    )
    assert selected.name == "gluon_local_sum_reduce_gfx950"


def test_amd_local_combine_prefill_selection_is_not_reference(mi350_platform) -> None:
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "combine",
        platform=mi350_platform,
        format_signature=format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"num_tokens": 257, "comm_strategy": None},
    )
    assert selected.name in {"gluon_local_sum_reduce_gfx950", "triton_moe_sum_reduce"}
