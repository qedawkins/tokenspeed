# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""S8 long-tail shape matrix for synthetic Kimi scenario fixtures.

This bead records explicit pass/fail status for the S1/S4/S5/S6 synthetic
scenario paths. Unsupported rows are concrete gaps for later S8 beads; passing
rows run both MLA and MoE through the existing Kimi language-only fixture paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

from test.runtime import test_kimi_fp8_ep_scenario as s4
from test.runtime import test_kimi_fp8_tp_scenario as s1
from test.runtime import test_kimi_prerouted_fused_ep_scenario as s5
from test.runtime import test_kimi_self_routing_fused_ep_scenario as s6


_SCENARIOS = (
    "s1_fp8_tp",
    "s4_fp8_ep",
    "s5_prerouted_fused_ep",
    "s6_self_routing_fused_ep",
)
_REQUIRED_CATEGORIES = frozenset(
    {
        "m_zero",
        "m_le_32",
        "mixed_prefill_decode",
        "chunked_prefill",
        "hot_experts",
        "empty_experts",
        "skewed_rank_traffic",
    }
)


@dataclass(frozen=True)
class _ShapeCase:
    name: str
    forward_mode: ForwardMode
    input_lengths: tuple[int, ...]
    categories: frozenset[str]
    ep_size: int = 8
    ep_rank: int = 7

    @property
    def is_prefill(self) -> bool:
        return self.forward_mode.is_extend()

    @property
    def total_tokens(self) -> int:
        if self.forward_mode.is_decode():
            return len(self.input_lengths)
        return sum(self.input_lengths)


@dataclass(frozen=True)
class _MatrixStatus:
    scenario: str
    shape: str
    categories: frozenset[str]
    status: str
    reason: str = ""


_COMMON_PASS_SHAPES = (
    _ShapeCase(
        name="m0-prefill",
        forward_mode=ForwardMode.EXTEND,
        input_lengths=(),
        categories=frozenset({"m_zero"}),
    ),
    _ShapeCase(
        name="hot-empty-prefill-m8",
        forward_mode=ForwardMode.EXTEND,
        input_lengths=(2, 2, 2, 2),
        categories=frozenset({"m_le_32", "hot_experts", "empty_experts"}),
    ),
    _ShapeCase(
        name="small-decode-m4",
        forward_mode=ForwardMode.DECODE,
        input_lengths=(1, 1, 1, 1),
        categories=frozenset({"m_le_32"}),
    ),
)

_S1_CHUNKED_PASS_SHAPE = _ShapeCase(
    name="chunked-prefill-m32",
    forward_mode=ForwardMode.EXTEND,
    input_lengths=(17, 8, 7),
    categories=frozenset(
        {
            "m_le_32",
            "chunked_prefill",
            "hot_experts",
            "empty_experts",
        }
    ),
)

_EP_SKEW_PASS_SHAPE = _ShapeCase(
    name="chunked-prefill-skew-m32",
    forward_mode=ForwardMode.EXTEND,
    input_lengths=(17, 8, 7),
    categories=frozenset(
        {
            "m_le_32",
            "chunked_prefill",
            "hot_experts",
            "empty_experts",
            "skewed_rank_traffic",
        }
    ),
)

_PASS_MATRIX: tuple[tuple[str, _ShapeCase], ...] = (
    *((scenario, shape) for scenario in _SCENARIOS for shape in _COMMON_PASS_SHAPES),
    ("s1_fp8_tp", _S1_CHUNKED_PASS_SHAPE),
    *(
        (scenario, _EP_SKEW_PASS_SHAPE)
        for scenario in (
            "s4_fp8_ep",
            "s5_prerouted_fused_ep",
            "s6_self_routing_fused_ep",
        )
    ),
)


_STATUS_MATRIX: tuple[_MatrixStatus, ...] = tuple(
    _MatrixStatus(scenario, shape.name, shape.categories, "pass")
    for scenario, shape in _PASS_MATRIX
) + (
    *(
        _MatrixStatus(
            scenario,
            "mixed-prefill-decode",
            frozenset({"mixed_prefill_decode"}),
            "xfail",
            "The S1/S4/S5/S6 test-local MLA fixture initializes either prefill "
            "or decode metadata, but not the scheduler-provided mixed metadata "
            "needed to validate true prefill plus decode rows in one forward.",
        )
        for scenario in _SCENARIOS
    ),
    _MatrixStatus(
        "s1_fp8_tp",
        "skewed-rank-traffic",
        frozenset({"skewed_rank_traffic"}),
        "not_applicable",
        "S1 is tensor-parallel only; EP rank skew is covered by S4/S5/S6.",
    ),
)


def _status_for(scenario: str, shape: str) -> _MatrixStatus:
    matches = [
        status
        for status in _STATUS_MATRIX
        if status.scenario == scenario and status.shape == shape
    ]
    assert len(matches) == 1
    return matches[0]


def _input_lengths(device: torch.device, shape: _ShapeCase) -> torch.Tensor:
    return torch.tensor(shape.input_lengths, device=device, dtype=torch.int32)


def _seeded_input_embeds(
    device: torch.device,
    *,
    scenario: str,
    shape: _ShapeCase,
) -> torch.Tensor:
    seed = 9400 + _SCENARIOS.index(scenario) * 101 + len(shape.name)
    torch.manual_seed(seed)
    return (torch.randn(shape.total_tokens, 32, device=device) * 0.10).bfloat16()


def _force_s1_hot_empty_router(model) -> None:
    layer = model.language_model.model.layers[0]
    with torch.no_grad():
        layer.mlp.gate.weight.zero_()
        bias = layer.mlp.gate.e_score_correction_bias
        assert bias is not None
        bias.copy_(
            torch.tensor(
                [-20.0, -19.0, 18.0, 20.0],
                device=bias.device,
                dtype=bias.dtype,
            )
        )


def _assert_attention_path(run, shape: _ShapeCase) -> None:
    if shape.is_prefill:
        assert run.prefill_calls >= 1
        assert run.decode_calls == 0
    else:
        assert run.decode_calls >= 1
        assert run.prefill_calls == 0


def _run_s1(monkeypatch: pytest.MonkeyPatch, shape: _ShapeCase) -> None:
    import tokenspeed_kernel

    route_calls: list[dict] = []
    original_route = tokenspeed_kernel.moe_route

    def _record_route(*args, **kwargs):
        route_calls.append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return original_route(*args, **kwargs)

    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _record_route)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(31415 + len(shape.name))
    model = s1._make_tiny_kimi_language_model(device)
    s1._init_kimi_language_weights(model)
    _force_s1_hot_empty_router(model)
    input_lengths = _input_lengths(device, shape)

    logits = s1._run_kimi_language_logits_case(
        model,
        total_tokens=shape.total_tokens,
        input_lengths=input_lengths,
        is_prefill=shape.is_prefill,
        route_calls=route_calls,
    )

    assert logits.shape == (len(shape.input_lengths), 64)


def _run_s4(monkeypatch: pytest.MonkeyPatch, shape: _ShapeCase) -> None:
    kernel_calls = s4._record_kernel_calls(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4110 + shape.ep_size + shape.ep_rank + len(shape.name))
    model = s4._make_tiny_kimi_ep_language_model(
        device,
        ep_size=shape.ep_size,
        ep_rank=shape.ep_rank,
    )
    s4._init_kimi_ep_language_weights(model)
    input_lengths = _input_lengths(device, shape)

    logits = s4._run_kimi_ep_language_logits_case(
        model,
        total_tokens=shape.total_tokens,
        input_lengths=input_lengths,
        is_prefill=shape.is_prefill,
        kernel_calls=kernel_calls,
    )

    assert logits.shape == (len(shape.input_lengths), 64)


def _run_s5(monkeypatch: pytest.MonkeyPatch, shape: _ShapeCase) -> None:
    kernel_calls = s4._record_kernel_calls(monkeypatch)
    path_calls = s5._record_pre_routed_path_calls(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    fused_model, _baseline_model = s5._make_model_pair(
        device,
        ep_size=shape.ep_size,
        ep_rank=shape.ep_rank,
    )
    input_lengths = _input_lengths(device, shape)
    input_embeds = _seeded_input_embeds(
        device,
        scenario="s5_prerouted_fused_ep",
        shape=shape,
    )

    run = s5._run_model_case(
        fused_model,
        total_tokens=shape.total_tokens,
        input_lengths=input_lengths,
        input_embeds=input_embeds,
        is_prefill=shape.is_prefill,
        kernel_calls=kernel_calls,
        path_calls=path_calls,
    )

    if shape.total_tokens == 0:
        assert not run.route_calls
        assert not run.dispatch_calls
        assert not run.expert_calls
        assert not run.combine_calls
        assert all(value == 0 for value in run.path_delta.values())
    else:
        s5._assert_common_ep_calls(run, require_expected_reduce=True)
        s5._assert_pre_routed_fused_path(run)
    _assert_attention_path(run, shape)


def _run_s6(monkeypatch: pytest.MonkeyPatch, shape: _ShapeCase) -> None:
    kernel_calls, path_calls = s6._record_s6_calls(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    self_routing_model, _pre_routed_model = s6._make_model_pair(
        device,
        ep_size=shape.ep_size,
        ep_rank=shape.ep_rank,
    )
    input_lengths = _input_lengths(device, shape)
    input_embeds = _seeded_input_embeds(
        device,
        scenario="s6_self_routing_fused_ep",
        shape=shape,
    )

    run = s6._run_model_case(
        self_routing_model,
        total_tokens=shape.total_tokens,
        input_lengths=input_lengths,
        input_embeds=input_embeds,
        is_prefill=shape.is_prefill,
        self_routing=True,
        kernel_calls=kernel_calls,
        path_calls=path_calls,
    )

    if shape.total_tokens == 0:
        assert not run.route_calls
        assert not run.dispatch_calls
        assert not run.expert_calls
        assert not run.combine_calls
        assert all(value == 0 for value in run.path_delta.values())
    else:
        s6._assert_route_call_shape(run, inside_self_gate_up=True)
        s6._assert_ep_kernel_calls(run)
        s6._assert_self_routing_path(run)
    _assert_attention_path(run, shape)


def test_kimi_s8_shape_matrix_declares_every_category() -> None:
    for scenario in _SCENARIOS:
        covered: set[str] = set()
        for status in _STATUS_MATRIX:
            if status.scenario == scenario:
                covered.update(status.categories)
                if status.status != "pass":
                    assert status.reason
        assert _REQUIRED_CATEGORIES <= covered, scenario

    for status in _STATUS_MATRIX:
        assert status.status in {"pass", "xfail", "not_applicable"}


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(
            status,
            marks=pytest.mark.xfail(reason=status.reason, strict=True),
            id=f"{status.scenario}-{status.shape}",
        )
        for status in _STATUS_MATRIX
        if status.status == "xfail"
    ],
)
def test_kimi_s8_explicit_failed_shape_status(status: _MatrixStatus) -> None:
    # These are status rows, not executable repros: true mixed prefill/decode
    # requires scheduler-owned metadata that these synthetic fixtures do not
    # construct.
    assert False, status.reason


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(
            status,
            id=f"{status.scenario}-{status.shape}",
        )
        for status in _STATUS_MATRIX
        if status.status == "not_applicable"
    ],
)
def test_kimi_s8_explicit_not_applicable_shape_status(
    status: _MatrixStatus,
) -> None:
    pytest.skip(status.reason)


def test_kimi_s8_empty_mla_wrappers_return_initialized_outputs() -> None:
    s1._require_cdna4_gpu()
    from tokenspeed_kernel.ops.attention import mla_decode_with_kvcache, mla_prefill

    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.empty((0, 4, 72), device=device, dtype=torch.bfloat16)
    k = torch.empty((0, 4, 72), device=device, dtype=torch.bfloat16)
    v = torch.empty((0, 4, 32), device=device, dtype=torch.bfloat16)
    seq_lens = torch.empty((0,), device=device, dtype=torch.int32)
    cum_seq_lens = torch.zeros((1,), device=device, dtype=torch.int32)
    out = torch.empty((0, 4, 32), device=device, dtype=torch.bfloat16)

    actual, actual_lse = mla_prefill(
        q,
        k,
        v,
        seq_lens,
        cum_seq_lens,
        max_seq_len=0,
        batch_size=0,
        softmax_scale=1.0,
        return_lse=True,
        out=out,
    )

    assert actual is out
    assert actual.shape == (0, 4, 32)
    assert actual_lse is not None
    assert actual_lse.shape == (0, 4)
    assert actual_lse.dtype == torch.float32

    decode = mla_decode_with_kvcache(
        q,
        torch.empty((0, 64, 72), device=device, dtype=torch.bfloat16),
        torch.empty((0, 0), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.int32),
        max_seqlen_k=0,
        kv_lora_rank=64,
        qk_rope_head_dim=8,
        value_head_dim=32,
    )

    assert decode.shape == (0, 4, 32)
    assert decode.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("scenario", "shape"),
    [
        pytest.param(scenario, shape, id=f"{scenario}-{shape.name}")
        for scenario, shape in _PASS_MATRIX
    ],
)
def test_kimi_s8_supported_longtail_shape_runs_mla_and_moe(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    shape: _ShapeCase,
) -> None:
    s1._require_cdna4_gpu()
    assert _status_for(scenario, shape.name).status == "pass"

    if scenario == "s1_fp8_tp":
        _run_s1(monkeypatch, shape)
    elif scenario == "s4_fp8_ep":
        _run_s4(monkeypatch, shape)
    elif scenario == "s5_prerouted_fused_ep":
        _run_s5(monkeypatch, shape)
    elif scenario == "s6_self_routing_fused_ep":
        _run_s6(monkeypatch, shape)
    else:
        raise AssertionError(f"unknown scenario: {scenario}")
