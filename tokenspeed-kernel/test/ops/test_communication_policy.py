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

import torch
from tokenspeed_kernel.ops import communication


def test_allreduce_fusion_lane_owns_single_row_eligibility(monkeypatch) -> None:
    monkeypatch.setattr(communication, "_ALLREDUCE_FUSION_LANE", None)
    one_row = torch.ones(1, 4)

    lane = communication.allreduce_fusion_lane(one_row, 6)

    assert lane is not None
    assert lane.shape == (1, 6)
    assert communication.allreduce_fusion_lane(torch.ones(2, 4), 6) is None
    assert communication.allreduce_fusion_lane(one_row, 6, enabled=False) is None


def test_latent_norm_support_requires_nvidia_preparation(monkeypatch) -> None:
    group = SimpleNamespace(size=lambda: 8)
    monkeypatch.setattr(
        communication,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )

    assert communication.allreduce_lane_latent_norm_supported(
        torch.ones(1, 4), group=group, max_token_num=8, prepared=True
    )
    assert not communication.allreduce_lane_latent_norm_supported(
        torch.ones(2, 4), group=group, max_token_num=1, prepared=True
    )
    assert not communication.allreduce_lane_latent_norm_supported(
        torch.ones(1, 4), group=group, max_token_num=8, prepared=False
    )

    monkeypatch.setattr(
        communication,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=False),
    )
    assert not communication.allreduce_lane_latent_norm_supported(
        torch.ones(1, 4), group=group, max_token_num=8, prepared=True
    )
