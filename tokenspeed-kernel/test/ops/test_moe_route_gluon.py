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

import pytest
import tokenspeed_kernel
import torch
from s1_component_utils import (
    assert_finite_close,
    assert_selected_amd_kernel_not_reference,
    extract_s1_text_model_dims,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _tiny_s1_dims():
    config = SimpleNamespace(
        hidden_size=1024,
        n_routed_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        num_attention_heads=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=8,
        v_head_dim=32,
        kv_lora_rank=64,
    )
    return extract_s1_text_model_dims(config, tp_size=4)


def _require_cdna4_gpu() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for the Gluon MoE route component test")


def _route_traits(*, num_expert_group: int, topk_group: int, topk: int) -> dict:
    return {
        "output_type": "topk",
        "biased": True,
        "grouped": True,
        "ep": True,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "topk": topk,
        "num_fused_shared_experts": 0,
    }


def _stable_topk_indices(values: torch.Tensor, k: int) -> list[int]:
    candidates = [(float(value), idx) for idx, value in enumerate(values.tolist())]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in candidates[:k]]


def _reference_grouped_biased_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    num_token_non_padded: torch.Tensor | None = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = router_logits.float().sigmoid()
    scores_for_choice = scores + correction_bias.float().unsqueeze(0)
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // num_expert_group

    topk_ids = torch.empty((num_tokens, topk), device=router_logits.device, dtype=torch.int32)
    topk_weights = torch.empty(
        (num_tokens, topk), device=router_logits.device, dtype=torch.float32
    )
    for token in range(num_tokens):
        row_choice = scores_for_choice[token]
        group_scores = []
        for group_id in range(num_expert_group):
            start = group_id * experts_per_group
            group_values = row_choice[start : start + experts_per_group]
            local = _stable_topk_indices(group_values, 2)
            group_scores.append(
                (
                    float(group_values[local[0]] + group_values[local[1]]),
                    group_id,
                )
            )
        group_scores.sort(key=lambda item: (-item[0], item[1]))
        selected_groups = {group_id for _, group_id in group_scores[:topk_group]}

        masked = row_choice.clone()
        for expert_id in range(num_experts):
            if expert_id // experts_per_group not in selected_groups:
                masked[expert_id] = -float("inf")
        ids = _stable_topk_indices(masked, topk)
        weights = scores[token, ids]
        if renormalize:
            weights = weights / weights.sum()
            if apply_routed_scaling_factor_on_output:
                weights = weights * routed_scaling_factor
        topk_ids[token] = torch.tensor(ids, device=router_logits.device, dtype=torch.int32)
        topk_weights[token] = weights

    if num_token_non_padded is not None:
        topk_ids[int(num_token_non_padded.item()) :] = -1
    return topk_weights, topk_ids


def _call_moe_route(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    num_token_non_padded: torch.Tensor | None = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tokenspeed_kernel.moe_route(
        hidden_states,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=None,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        dtype=router_logits.dtype,
        traits=_route_traits(
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            topk=topk,
        ),
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 32, 257])
def test_gluon_grouped_biased_route_matches_reference_token_counts(
    device: str, num_tokens: int
) -> None:
    _require_cdna4_gpu()
    dims = _tiny_s1_dims()
    num_expert_group = 4
    topk_group = 2
    topk = dims.K

    torch.manual_seed(5000 + num_tokens)
    hidden_states = torch.randn(num_tokens, dims.H, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(num_tokens, dims.E, device=device, dtype=torch.float32)
    correction_bias = torch.linspace(-0.15, 0.2, dims.E, device=device)

    actual_weights, actual_ids = _call_moe_route(
        hidden_states,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )
    torch.cuda.synchronize()

    expected_weights, expected_ids = _reference_grouped_biased_topk(
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )
    assert actual_ids.dtype == torch.int32
    assert actual_weights.dtype == torch.float32
    assert torch.equal(actual_ids, expected_ids)
    assert_finite_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)


def test_gluon_grouped_biased_route_handles_group_bias_ties_and_scaling(
    device: str,
) -> None:
    _require_cdna4_gpu()
    hidden_states = torch.zeros(1, 16, device=device, dtype=torch.bfloat16)
    router_logits = torch.zeros(1, 8, device=device, dtype=torch.float32)
    correction_bias = torch.tensor(
        [0.20, 0.20, 0.15, 0.15, 0.90, 0.90, 0.89, 0.89],
        device=device,
        dtype=torch.float32,
    )
    topk = 4
    num_expert_group = 4
    topk_group = 2
    routed_scaling_factor = 2.5

    actual_weights, actual_ids = _call_moe_route(
        hidden_states,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    torch.cuda.synchronize()

    expected_weights, expected_ids = _reference_grouped_biased_topk(
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    assert torch.equal(actual_ids, expected_ids)
    assert actual_ids.tolist() == [[4, 5, 6, 7]]
    assert_finite_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)


def test_gluon_grouped_biased_route_supports_kimi_like_shape(device: str) -> None:
    _require_cdna4_gpu()
    num_tokens = 8
    hidden_size = 64
    num_experts = 384
    topk = 8
    num_expert_group = 8
    topk_group = 4

    torch.manual_seed(6000)
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.bfloat16)
    correction_bias = torch.randn(num_experts, device=device, dtype=torch.float32) * 0.05

    actual_weights, actual_ids = _call_moe_route(
        hidden_states,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=False,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )
    torch.cuda.synchronize()

    expected_weights, expected_ids = _reference_grouped_biased_topk(
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=False,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )
    assert torch.equal(actual_ids, expected_ids)
    assert_finite_close(actual_weights, expected_weights, atol=5e-4, rtol=5e-4)


def test_gluon_grouped_biased_route_masks_padded_ids(device: str) -> None:
    _require_cdna4_gpu()
    num_tokens = 8
    num_experts = 16
    topk = 4
    num_expert_group = 4
    topk_group = 2
    hidden_states = torch.zeros(num_tokens, 32, device=device, dtype=torch.bfloat16)
    router_logits = torch.arange(
        num_tokens * num_experts,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, num_experts) / 100.0
    correction_bias = torch.zeros(num_experts, device=device, dtype=torch.float32)
    num_token_non_padded = torch.tensor(5, device=device, dtype=torch.int32)

    actual_weights, actual_ids = _call_moe_route(
        hidden_states,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        num_token_non_padded=num_token_non_padded,
    )
    torch.cuda.synchronize()

    expected_weights, expected_ids = _reference_grouped_biased_topk(
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        num_token_non_padded=num_token_non_padded,
    )
    assert torch.equal(actual_ids, expected_ids)
    assert actual_ids[5:].eq(-1).all()
    assert_finite_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)


def test_amd_grouped_biased_route_selection_uses_gluon(mi350_platform) -> None:
    selected = assert_selected_amd_kernel_not_reference(
        "moe",
        "route",
        platform=mi350_platform,
        format_signature=format_signature(logits=dense_tensor_format(torch.float32)),
        traits=_route_traits(num_expert_group=8, topk_group=4, topk=8),
        expected_solution="gluon",
    )
    assert selected.name == "gluon_grouped_biased_topk_gfx950"
