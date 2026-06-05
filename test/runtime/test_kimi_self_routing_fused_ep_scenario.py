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

"""Synthetic S6 self-routing fused EP scenario coverage.

The test compares self-routing fused EP against the S5 pre-routed fused path on
matching synthetic Kimi language-only models. It proves the external TopK call
is bypassed by recording whether each route kernel call occurs while inside the
self-routing fused helper. As in S4/S5, this is single-process coverage: Iris
D2D is not exercised because torch.distributed is not initialized, and the test
asserts the EP workspace stays on the torch backend.
"""

from __future__ import annotations

import math
import time
from collections.abc import MutableMapping
from dataclasses import dataclass

import pytest
import torch

from test.runtime.test_kimi_fp8_ep_scenario import (
    _expected_logits,
    _init_kimi_ep_language_weights,
    _make_tiny_kimi_ep_language_model,
    _reference_fp8_ep_moe,
    _reference_kimi_ep_language_logits,
    _require_cdna4_gpu,
)
from test.runtime.test_kimi_prerouted_fused_ep_scenario import (
    _input_embeds_for_mode,
    _input_lengths_for_mode,
    _make_context,
)


@dataclass(frozen=True)
class _S6Run:
    logits: torch.Tensor
    elapsed_ms: float
    route_calls: list[dict]
    dispatch_calls: list[dict]
    expert_calls: list[dict]
    combine_calls: list[dict]
    path_delta: dict[str, int]
    prefill_calls: int
    decode_calls: int


def _make_model_pair(
    device: torch.device,
    *,
    ep_size: int,
    ep_rank: int,
):
    torch.manual_seed(8300 + ep_size + ep_rank)
    pre_routed_model = _make_tiny_kimi_ep_language_model(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    _init_kimi_ep_language_weights(pre_routed_model)

    self_routing_model = _make_tiny_kimi_ep_language_model(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    _init_kimi_ep_language_weights(self_routing_model)
    self_routing_model.load_state_dict(pre_routed_model.state_dict())
    _enable_self_routing_fused_ep(self_routing_model)
    return self_routing_model, pre_routed_model


def _enable_self_routing_fused_ep(model) -> None:
    from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat

    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts
    experts.backend.routing_config = {"moe_fused_features": {"self_routing"}}
    assert experts.backend.topk_output_format == TopKOutputFormat.BYPASSED
    layer.mlp.topk.topk_config.output_format = experts.backend.topk_output_format


def _reference_kimi_self_routing_logits(
    model,
    input_embeds: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    is_prefill: bool,
    attention_hidden_states: torch.Tensor,
) -> torch.Tensor:
    from tokenspeed.runtime.layers.moe.backends.ep_self_routing import (
        self_routing_topk_from_bypassed,
    )

    layer = model.language_model.model.layers[0]
    experts = layer.mlp.experts
    hidden_states, residual = layer.comm_manager.input_reduce_norm(input_embeds, None)
    assert hidden_states.shape == attention_hidden_states.shape
    hidden_states = attention_hidden_states
    hidden_states, residual = layer.comm_manager.post_attn_reduce_norm(
        hidden_states,
        residual,
        None,
    )
    hidden_states = layer.comm_manager.pre_mlp_comm(hidden_states, None)
    router_logits = layer.mlp.gate(hidden_states)
    topk_output = layer.mlp.topk(hidden_states, router_logits)
    assert topk_output.format.is_bypassed()
    routed_topk = self_routing_topk_from_bypassed(topk_output)
    moe_hidden = _reference_fp8_ep_moe(
        hidden_states,
        experts,
        routed_topk.topk_ids,
        routed_topk.topk_weights,
    )
    final_hidden = layer.comm_manager.final_norm(
        moe_hidden,
        residual,
        None,
        model.language_model.model.norm,
    )
    return _expected_logits(
        final_hidden,
        input_lengths,
        model.language_model.lm_head,
        is_prefill=is_prefill,
    )


def _record_s6_calls(monkeypatch: pytest.MonkeyPatch):
    import tokenspeed_kernel
    from tokenspeed.runtime.layers.moe.backends import (
        ep_fused_down_combine,
        ep_fused_gate_up,
        ep_fused_metadata,
    )

    state = {"self_gate_up_depth": 0}
    kernel_calls: dict[str, list[dict]] = {
        "route": [],
        "dispatch": [],
        "experts": [],
        "combine": [],
    }
    path_calls = {
        "metadata": 0,
        "pre_gate_up": 0,
        "pre_down_combine": 0,
        "self_gate_up": 0,
        "self_down_combine": 0,
    }
    original_route = tokenspeed_kernel.moe_route
    original_dispatch = tokenspeed_kernel.moe_dispatch
    original_experts = tokenspeed_kernel.moe_experts
    original_combine = tokenspeed_kernel.moe_combine

    def _record_route(*args, **kwargs):
        result = original_route(*args, **kwargs)
        kernel_calls["route"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
                "inside_self_gate_up": state["self_gate_up_depth"] > 0,
            }
        )
        return result

    def _record_dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        kernel_calls["dispatch"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def _record_experts(*args, **kwargs):
        result = original_experts(*args, **kwargs)
        kernel_calls["experts"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "features": set(kwargs.get("features") or set()),
            }
        )
        return result

    def _record_combine(*args, **kwargs):
        result = original_combine(*args, **kwargs)
        kernel_calls["combine"].append(
            {
                "expected_kernel_name": kwargs.get("expected_kernel_name"),
                "traits": dict(kwargs.get("traits") or {}),
            }
        )
        return result

    def wrap(module, attr: str, key: str, *, marks_self_gate_up: bool = False) -> None:
        original = getattr(module, attr)

        def _wrapped(*args, **kwargs):
            path_calls[key] += 1
            if not marks_self_gate_up:
                return original(*args, **kwargs)
            state["self_gate_up_depth"] += 1
            try:
                return original(*args, **kwargs)
            finally:
                state["self_gate_up_depth"] -= 1

        monkeypatch.setattr(module, attr, _wrapped)

    monkeypatch.setattr(tokenspeed_kernel, "moe_route", _record_route)
    monkeypatch.setattr(tokenspeed_kernel, "moe_dispatch", _record_dispatch)
    monkeypatch.setattr(tokenspeed_kernel, "moe_experts", _record_experts)
    monkeypatch.setattr(tokenspeed_kernel, "moe_combine", _record_combine)
    wrap(ep_fused_metadata, "build_pre_routed_fused_ep_metadata", "metadata")
    wrap(ep_fused_gate_up, "build_pre_routed_fused_ep_metadata", "metadata")
    wrap(ep_fused_gate_up, "pre_routed_fused_dispatch_gate_up", "pre_gate_up")
    wrap(ep_fused_down_combine, "pre_routed_fused_down_combine", "pre_down_combine")
    wrap(
        ep_fused_gate_up,
        "self_routing_fused_dispatch_gate_up",
        "self_gate_up",
        marks_self_gate_up=True,
    )
    wrap(ep_fused_down_combine, "self_routing_fused_down_combine", "self_down_combine")
    return kernel_calls, path_calls


def _delta(values: dict[str, int], start: dict[str, int]) -> dict[str, int]:
    return {key: values[key] - start[key] for key in values}


def _run_model_case(
    model,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    input_embeds: torch.Tensor,
    is_prefill: bool,
    self_routing: bool,
    kernel_calls: MutableMapping[str, list[dict]],
    path_calls: dict[str, int],
) -> _S6Run:
    device = input_lengths.device
    ctx, out_cache_loc, attn_backend = _make_context(
        device,
        total_tokens=total_tokens,
        input_lengths=input_lengths,
        is_prefill=is_prefill,
    )
    input_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
    positions = torch.arange(total_tokens, device=device, dtype=torch.long)
    route_start = len(kernel_calls["route"])
    dispatch_start = len(kernel_calls["dispatch"])
    experts_start = len(kernel_calls["experts"])
    combine_start = len(kernel_calls["combine"])
    path_start = dict(path_calls)

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(
        ctx,
        input_ids,
        positions,
        out_cache_loc,
        input_lengths,
        input_embeds=input_embeds,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    route_calls = kernel_calls["route"][route_start:]
    dispatch_calls = kernel_calls["dispatch"][dispatch_start:]
    expert_calls = kernel_calls["experts"][experts_start:]
    combine_calls = kernel_calls["combine"][combine_start:]
    path_delta = _delta(path_calls, path_start)
    reference = (
        _reference_kimi_self_routing_logits
        if self_routing
        else _reference_kimi_ep_language_logits
    )
    expected = reference(
        model,
        input_embeds,
        input_lengths,
        is_prefill=is_prefill,
        attention_hidden_states=model.language_model.model.layers[
            0
        ].comm_manager.last_attention_output,
    )

    assert output.next_token_logits.shape == (input_lengths.numel(), 64)
    assert output.next_token_logits.dtype == torch.bfloat16
    assert output.next_token_logits.isfinite().all()
    torch.testing.assert_close(
        output.next_token_logits.float(),
        expected.float(),
        atol=0.90,
        rtol=0.18,
        check_dtype=False,
    )
    experts = model.language_model.model.layers[0].mlp.experts
    workspace = experts.backend._ep_workspace
    if total_tokens == 0:
        assert not route_calls
        assert not dispatch_calls
        assert not expert_calls
        assert not combine_calls
        assert all(value == 0 for value in path_delta.values())
        assert workspace is None
    else:
        assert workspace is not None
        assert workspace.backend == "torch"
        assert workspace.world_size == experts.ep_size
        assert workspace.rank == experts.ep_rank
    assert elapsed_ms > 0.0
    assert math.isfinite(elapsed_ms)
    return _S6Run(
        logits=output.next_token_logits,
        elapsed_ms=elapsed_ms,
        route_calls=route_calls,
        dispatch_calls=dispatch_calls,
        expert_calls=expert_calls,
        combine_calls=combine_calls,
        path_delta=path_delta,
        prefill_calls=attn_backend.prefill_calls,
        decode_calls=attn_backend.decode_calls,
    )


def _assert_route_call_shape(run: _S6Run, *, inside_self_gate_up: bool) -> None:
    assert run.route_calls
    assert all(
        call["inside_self_gate_up"] is inside_self_gate_up for call in run.route_calls
    )
    assert any(
        call.get("expected_kernel_name") == "gluon_grouped_biased_topk_gfx950"
        and call.get("traits", {}).get("biased") is True
        and call.get("traits", {}).get("grouped") is True
        and call.get("traits", {}).get("ep") is True
        for call in run.route_calls
    )


def _assert_ep_kernel_calls(run: _S6Run) -> None:
    assert any(
        call.get("expected_kernel_name") == "gluon_ep_metadata_gfx950"
        and call.get("traits", {}).get("comm_strategy") == "ep_metadata"
        for call in run.dispatch_calls
    )
    assert sum(
        call.get("expected_kernel_name") == "gluon_fp8_local_experts_gfx950"
        for call in run.expert_calls
    ) >= 2
    assert any(
        call.get("expected_kernel_name") == "gluon_local_sum_reduce_gfx950"
        and call.get("traits", {}).get("comm_strategy") is None
        for call in run.combine_calls
    )


def _assert_self_routing_path(run: _S6Run) -> None:
    assert run.path_delta["self_gate_up"] >= 1
    assert run.path_delta["self_down_combine"] >= 1
    assert run.path_delta["metadata"] >= 1
    assert run.path_delta["pre_gate_up"] >= 1
    assert run.path_delta["pre_down_combine"] >= 1


def _assert_pre_routed_path(run: _S6Run) -> None:
    assert run.path_delta["self_gate_up"] == 0
    assert run.path_delta["self_down_combine"] == 0
    assert run.path_delta["metadata"] >= 1
    assert run.path_delta["pre_gate_up"] >= 1
    assert run.path_delta["pre_down_combine"] >= 1


@pytest.mark.parametrize(
    ("ep_size", "ep_rank"),
    [(4, 1), (8, 3)],
    ids=["4-rank-ep", "8-rank-ep"],
)
def test_kimi_self_routing_fused_ep_matches_prerouted_and_records_timings(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
    ep_rank: int,
) -> None:
    _require_cdna4_gpu()
    kernel_calls, path_calls = _record_s6_calls(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    self_model, pre_model = _make_model_pair(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    self_topk = self_model.language_model.model.layers[0].mlp.topk.topk_config
    pre_topk = pre_model.language_model.model.layers[0].mlp.topk.topk_config
    assert self_topk.output_format.is_bypassed()
    assert pre_topk.output_format.is_standard()

    for is_prefill, mode in ((True, "prefill"), (False, "decode")):
        input_lengths = _input_lengths_for_mode(device, is_prefill=is_prefill)
        total_tokens = int(input_lengths.sum().item()) if is_prefill else 2
        input_embeds = _input_embeds_for_mode(
            device,
            total_tokens=total_tokens,
            ep_size=ep_size,
            ep_rank=ep_rank,
            is_prefill=is_prefill,
        )

        _run_model_case(
            self_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            self_routing=True,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )
        _run_model_case(
            pre_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            self_routing=False,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )

        self_run = _run_model_case(
            self_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            self_routing=True,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )
        pre_run = _run_model_case(
            pre_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            self_routing=False,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )

        _assert_route_call_shape(self_run, inside_self_gate_up=True)
        _assert_route_call_shape(pre_run, inside_self_gate_up=False)
        _assert_ep_kernel_calls(self_run)
        _assert_ep_kernel_calls(pre_run)
        _assert_self_routing_path(self_run)
        _assert_pre_routed_path(pre_run)
        if is_prefill:
            assert self_run.prefill_calls >= 1
            assert pre_run.prefill_calls >= 1
        else:
            assert self_run.decode_calls >= 1
            assert pre_run.decode_calls >= 1
        torch.testing.assert_close(
            self_run.logits.float(),
            pre_run.logits.float(),
            atol=0.90,
            rtol=0.18,
            check_dtype=False,
        )
        ratio = self_run.elapsed_ms / pre_run.elapsed_ms
        assert ratio > 0.0
        assert math.isfinite(ratio)
        print(
            "KIMI-S6-SRFUSED-005 "
            f"ep_size={ep_size} ep_rank={ep_rank} mode={mode} "
            f"self_routing_ms={self_run.elapsed_ms:.3f} "
            f"pre_routed_ms={pre_run.elapsed_ms:.3f} ratio={ratio:.3f}",
            flush=True,
        )
