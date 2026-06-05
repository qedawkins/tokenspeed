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

"""Synthetic S5 pre-routed fused EP scenario coverage.

This scenario uses the same artifact-gated single-process Kimi language-only
setup as the S4 FP8 EP scenario, but compares the normal pre-routed fused EP
path against the S4-style path obtained by disabling the fused gate on a model
with identical weights and inputs. The timing checks characterize the tiny
synthetic prefill/decode runs; they intentionally do not assert or claim a
stable speedup.

As with the S4 synthetic scenario, Iris D2D is not exercised because
torch.distributed is not initialized; the test asserts the EP workspace stays
on the single-process torch backend.
"""

from __future__ import annotations

import math
import time
from collections.abc import MutableMapping
from dataclasses import dataclass

import pytest
import torch

from test.runtime.test_kimi_fp8_ep_scenario import (
    _GluonMLABackend,
    _init_kimi_ep_language_weights,
    _make_mla_kv_pool,
    _make_tiny_kimi_ep_language_model,
    _record_kernel_calls,
    _reference_kimi_ep_language_logits,
    _require_cdna4_gpu,
)


@dataclass(frozen=True)
class _ScenarioRun:
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
    torch.manual_seed(6200 + ep_size + ep_rank)
    fused_model = _make_tiny_kimi_ep_language_model(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    _init_kimi_ep_language_weights(fused_model)

    baseline_model = _make_tiny_kimi_ep_language_model(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    _init_kimi_ep_language_weights(baseline_model)
    baseline_model.load_state_dict(fused_model.state_dict())
    _disable_pre_routed_fused_ep(baseline_model)
    return fused_model, baseline_model


def _disable_pre_routed_fused_ep(model) -> None:
    experts = model.language_model.model.layers[0].mlp.experts
    experts.backend._can_use_pre_routed_fused_ep = (
        lambda layer, hidden_states: False
    )


def _record_pre_routed_path_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    from tokenspeed.runtime.layers.moe.backends import (
        ep_fused_down_combine,
        ep_fused_gate_up,
        ep_fused_metadata,
    )

    path_calls = {
        "metadata": 0,
        "pre_gate_up": 0,
        "pre_down_combine": 0,
        "self_gate_up": 0,
        "self_down_combine": 0,
    }

    def wrap(module, attr: str, key: str) -> None:
        original = getattr(module, attr)

        def _wrapped(*args, **kwargs):
            path_calls[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, attr, _wrapped)

    wrap(ep_fused_metadata, "build_pre_routed_fused_ep_metadata", "metadata")
    wrap(ep_fused_gate_up, "pre_routed_fused_dispatch_gate_up", "pre_gate_up")
    wrap(ep_fused_down_combine, "pre_routed_fused_down_combine", "pre_down_combine")
    wrap(ep_fused_gate_up, "self_routing_fused_dispatch_gate_up", "self_gate_up")
    wrap(ep_fused_down_combine, "self_routing_fused_down_combine", "self_down_combine")
    return path_calls


def _make_context(
    device: torch.device,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    is_prefill: bool,
):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    attn_backend = _GluonMLABackend(page_size=64, max_context_len=64)
    token_to_kv_pool = _make_mla_kv_pool(device)
    if is_prefill:
        attn_backend.init_prefill_metadata(input_lengths)
        out_cache_loc = torch.arange(total_tokens, device=device, dtype=torch.int32)
    else:
        page_table = torch.arange(
            input_lengths.numel(),
            device=device,
            dtype=torch.int32,
        ).view(-1, 1)
        attn_backend.init_decode_metadata(page_table, input_lengths)
        out_cache_loc = page_table.flatten() * 64
    ctx = ForwardContext(
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        bs=input_lengths.numel(),
        num_extends=input_lengths.numel() if is_prefill else 0,
        input_num_tokens=total_tokens,
        forward_mode=forward_mode,
    )
    return ctx, out_cache_loc, attn_backend


def _path_delta(path_calls: dict[str, int], start: dict[str, int]) -> dict[str, int]:
    return {key: path_calls[key] - start[key] for key in path_calls}


def _run_model_case(
    model,
    *,
    total_tokens: int,
    input_lengths: torch.Tensor,
    input_embeds: torch.Tensor,
    is_prefill: bool,
    kernel_calls: MutableMapping[str, list[dict]],
    path_calls: dict[str, int],
) -> _ScenarioRun:
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
    delta = _path_delta(path_calls, path_start)
    expected = _reference_kimi_ep_language_logits(
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
    assert workspace is not None
    assert workspace.backend == "torch"
    assert workspace.world_size == experts.ep_size
    assert workspace.rank == experts.ep_rank
    assert elapsed_ms > 0.0
    assert math.isfinite(elapsed_ms)
    return _ScenarioRun(
        logits=output.next_token_logits,
        elapsed_ms=elapsed_ms,
        route_calls=route_calls,
        dispatch_calls=dispatch_calls,
        expert_calls=expert_calls,
        combine_calls=combine_calls,
        path_delta=delta,
        prefill_calls=attn_backend.prefill_calls,
        decode_calls=attn_backend.decode_calls,
    )


def _assert_common_ep_calls(
    run: _ScenarioRun,
    *,
    require_expected_reduce: bool,
) -> None:
    assert any(
        call.get("expected_kernel_name") == "gluon_grouped_biased_topk_gfx950"
        and call.get("traits", {}).get("biased") is True
        and call.get("traits", {}).get("grouped") is True
        and call.get("traits", {}).get("ep") is True
        for call in run.route_calls
    )
    assert any(
        call.get("expected_kernel_name") == "gluon_ep_metadata_gfx950"
        and call.get("traits", {}).get("comm_strategy") == "ep_metadata"
        for call in run.dispatch_calls
    )
    assert sum(
        call.get("expected_kernel_name") == "gluon_fp8_local_experts_gfx950"
        for call in run.expert_calls
    ) >= 2
    if require_expected_reduce:
        assert any(
            call.get("expected_kernel_name") == "gluon_local_sum_reduce_gfx950"
            and call.get("traits", {}).get("comm_strategy") is None
            for call in run.combine_calls
        )
    else:
        assert any(
            call.get("traits", {}).get("comm_strategy") is None
            for call in run.combine_calls
        )


def _assert_pre_routed_fused_path(run: _ScenarioRun) -> None:
    assert run.path_delta["metadata"] >= 1
    assert run.path_delta["pre_gate_up"] >= 1
    assert run.path_delta["pre_down_combine"] >= 1
    assert run.path_delta["self_gate_up"] == 0
    assert run.path_delta["self_down_combine"] == 0


def _assert_s4_style_baseline_path(run: _ScenarioRun) -> None:
    assert run.path_delta["metadata"] == 0
    assert run.path_delta["pre_gate_up"] == 0
    assert run.path_delta["pre_down_combine"] == 0
    assert run.path_delta["self_gate_up"] == 0
    assert run.path_delta["self_down_combine"] == 0


def _input_lengths_for_mode(device: torch.device, *, is_prefill: bool) -> torch.Tensor:
    if is_prefill:
        return torch.tensor([3, 2], device=device, dtype=torch.int32)
    return torch.ones(2, device=device, dtype=torch.int32)


def _input_embeds_for_mode(
    device: torch.device,
    *,
    total_tokens: int,
    ep_size: int,
    ep_rank: int,
    is_prefill: bool,
) -> torch.Tensor:
    torch.manual_seed(7100 + ep_size * 17 + ep_rank * 3 + int(is_prefill))
    return (torch.randn(total_tokens, 32, device=device) * 0.10).bfloat16()


@pytest.mark.parametrize(
    ("ep_size", "ep_rank"),
    [(4, 1), (8, 3)],
    ids=["4-rank-ep", "8-rank-ep"],
)
def test_kimi_prerouted_fused_ep_matches_s4_baseline_and_records_timings(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
    ep_rank: int,
) -> None:
    _require_cdna4_gpu()
    kernel_calls = _record_kernel_calls(monkeypatch)
    path_calls = _record_pre_routed_path_calls(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    fused_model, baseline_model = _make_model_pair(
        device,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

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
            fused_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )
        _run_model_case(
            baseline_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )

        fused = _run_model_case(
            fused_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )
        baseline = _run_model_case(
            baseline_model,
            total_tokens=total_tokens,
            input_lengths=input_lengths,
            input_embeds=input_embeds,
            is_prefill=is_prefill,
            kernel_calls=kernel_calls,
            path_calls=path_calls,
        )

        _assert_common_ep_calls(fused, require_expected_reduce=True)
        _assert_common_ep_calls(baseline, require_expected_reduce=False)
        _assert_pre_routed_fused_path(fused)
        _assert_s4_style_baseline_path(baseline)
        if is_prefill:
            assert fused.prefill_calls >= 1
            assert baseline.prefill_calls >= 1
        else:
            assert fused.decode_calls >= 1
            assert baseline.decode_calls >= 1
        torch.testing.assert_close(
            fused.logits.float(),
            baseline.logits.float(),
            atol=0.90,
            rtol=0.18,
            check_dtype=False,
        )
        ratio = fused.elapsed_ms / baseline.elapsed_ms
        assert ratio > 0.0
        assert math.isfinite(ratio)
        print(
            "KIMI-S5-PRFUSED-005 "
            f"ep_size={ep_size} ep_rank={ep_rank} mode={mode} "
            f"fused_ms={fused.elapsed_ms:.3f} "
            f"s4_unfused_ms={baseline.elapsed_ms:.3f} ratio={ratio:.3f}",
            flush=True,
        )
