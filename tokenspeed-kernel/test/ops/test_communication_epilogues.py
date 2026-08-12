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

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tokenspeed_kernel.ops import communication


def _attnres_args():
    tensor = torch.empty(1, 4)
    return {
        "input_tensor": tensor,
        "residual": torch.empty_like(tensor),
        "res_weight": torch.empty(4),
        "rms_weight": torch.empty(4),
        "score_weight": torch.empty(4),
        "output_weight": torch.empty(4),
        "scratch": (torch.empty(1), torch.empty(1), torch.empty(1, 4)),
        "rank": 0,
        "group": SimpleNamespace(size=lambda: 8),
        "local_world_size": 8,
        "eps": 1e-5,
        "max_token_num": 8,
        "enabled": True,
        "prepared": True,
    }


def test_nvidia_attnres_dispatch_uses_raw_weights(monkeypatch):
    args = _attnres_args()
    monkeypatch.setattr(
        communication,
        "current_platform",
        lambda: SimpleNamespace(is_amd=False, is_nvidia=True),
    )
    implementation = Mock(return_value=(Mock(), Mock()))
    monkeypatch.setattr(
        communication,
        "_trtllm_allreduce_residual_attnres_combine",
        implementation,
    )

    communication.allreduce_residual_attnres_combine(**args)

    positional = implementation.call_args.args
    assert positional[2] is args["res_weight"]
    assert positional[3] is args["rms_weight"]
    assert positional[4] is args["output_weight"]


def test_amd_attnres_dispatch_uses_precombined_score_weight(monkeypatch):
    from tokenspeed_kernel.ops.communication import triton

    args = _attnres_args()
    monkeypatch.setattr(
        communication,
        "current_platform",
        lambda: SimpleNamespace(is_amd=True, is_nvidia=False),
    )
    monkeypatch.setattr(
        triton,
        "allreduce_residual_attnres_combine_supported",
        lambda *_args, **_kwargs: True,
    )
    implementation = Mock(return_value=(Mock(), Mock()))
    monkeypatch.setattr(
        triton,
        "allreduce_residual_attnres_combine",
        implementation,
    )

    communication.allreduce_residual_attnres_combine(**args)

    positional = implementation.call_args.args
    assert positional[2] is args["score_weight"]
    assert positional[3] is args["output_weight"]


def test_nvidia_residual_rmsnorm_returns_sentinel_without_workspace(monkeypatch):
    from tokenspeed_kernel.ops.communication import trtllm

    monkeypatch.setattr(
        communication,
        "current_platform",
        lambda: SimpleNamespace(is_amd=False, is_nvidia=True),
    )
    monkeypatch.setattr(
        trtllm,
        "ensure_workspace_initialized",
        lambda **_kwargs: False,
        raising=False,
    )
    implementation = Mock()
    monkeypatch.setattr(
        communication,
        "_trtllm_allreduce_residual_rmsnorm",
        implementation,
    )
    tensor = torch.empty(1, 4)

    result = communication.allreduce_residual_rmsnorm(
        tensor,
        torch.empty_like(tensor),
        torch.empty(4),
        rank=0,
        group=SimpleNamespace(size=lambda: 8),
        max_token_num=8,
    )

    assert result == (None, None, None, None)
    implementation.assert_not_called()
