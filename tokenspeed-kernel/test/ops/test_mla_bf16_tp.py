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
from s1_component_utils import (
    assert_selected_amd_kernel_not_reference,
    extract_s1_text_model_dims,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


PAGE_SIZE = 64


def _tiny_s2_dims():
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


def test_bf16_mla_tp_reuses_existing_gluon_backend_names(mi350_platform) -> None:
    dims = _tiny_s2_dims()

    selected_decode = assert_selected_amd_kernel_not_reference(
        "attention",
        "mla_decode_with_kvcache",
        platform=mi350_platform,
        format_signature=format_signature(
            q=dense_tensor_format(torch.bfloat16),
            kv_cache=dense_tensor_format(torch.bfloat16),
        ),
        features=frozenset({"paged", "mla"}),
        traits={
            "num_q_heads": dims.A_r,
            "query_dim": dims.R_kv + dims.D_rope,
            "kv_lora_rank": dims.R_kv,
            "qk_rope_head_dim": dims.D_rope,
            "value_head_dim": dims.D_v,
            "page_size": PAGE_SIZE,
        },
        expected_solution="gluon",
    )

    selected_prefill = assert_selected_amd_kernel_not_reference(
        "attention",
        "mla_prefill",
        platform=mi350_platform,
        format_signature=format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k=dense_tensor_format(torch.bfloat16),
            v=dense_tensor_format(torch.bfloat16),
        ),
        features=frozenset({"mla"}),
        traits={
            "num_q_heads": dims.A_r,
            "num_kv_heads": dims.A_r,
            "query_dim": dims.D_nope + dims.D_rope,
            "value_head_dim": dims.D_v,
            "is_causal": True,
            "return_lse": True,
        },
        expected_solution="gluon",
    )

    selected_chunk_replay = assert_selected_amd_kernel_not_reference(
        "attention",
        "mla_prefill",
        platform=mi350_platform,
        format_signature=format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k=dense_tensor_format(torch.bfloat16),
            v=dense_tensor_format(torch.bfloat16),
        ),
        features=frozenset({"mla"}),
        traits={
            "num_q_heads": dims.A_r,
            "num_kv_heads": dims.A_r,
            "query_dim": dims.D_nope + dims.D_rope,
            "value_head_dim": dims.D_v,
            "is_causal": False,
            "return_lse": True,
        },
        expected_solution="gluon",
    )

    assert selected_decode.name == "gluon_mla_decode_gfx950"
    assert selected_prefill.name == "gluon_mla_prefill_gfx950"
    assert selected_chunk_replay.name == "gluon_mla_prefill_gfx950"
