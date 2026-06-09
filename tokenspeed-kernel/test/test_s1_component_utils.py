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
import torch
from s1_component_utils import (
    assert_exact_metadata_equal,
    assert_finite_close,
    assert_selected_amd_kernel_not_reference,
    extract_s1_text_model_dims,
    s1_decode_token_counts,
    s1_prefill_token_counts,
    s1_varlen_prefill_sequences,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import dummy_impl

pytestmark = pytest.mark.usefixtures("fresh_registry")


def _tiny_text_config(**overrides):
    fields = {
        "hidden_size": 7168,
        "n_routed_experts": 384,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 64,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "kv_lora_rank": 512,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_extract_s1_text_model_dims_from_direct_config():
    dims = extract_s1_text_model_dims(_tiny_text_config(), tp_size=8)

    assert dims.H == 7168
    assert dims.E == 384
    assert dims.K == 8
    assert dims.I == 2048
    assert dims.A == 64
    assert dims.D_nope == 128
    assert dims.D_rope == 64
    assert dims.D_v == 128
    assert dims.R_kv == 512
    assert dims.A_r == 8


def test_extract_s1_text_model_dims_from_wrapper_config():
    wrapper = SimpleNamespace(text_config=_tiny_text_config(num_attention_heads=32))

    dims = extract_s1_text_model_dims(wrapper, tp_size=4)

    assert dims.A == 32
    assert dims.A_r == 8


def test_extract_s1_text_model_dims_from_mapping_config():
    text_config = _tiny_text_config().__dict__

    dims = extract_s1_text_model_dims({"text_config": text_config}, tp_size=16)

    assert dims.H == 7168
    assert dims.A_r == 4


def test_extract_s1_text_model_dims_reports_missing_field():
    config = _tiny_text_config()
    delattr(config, "kv_lora_rank")

    with pytest.raises(
        ValueError,
        match="missing required text config field 'kv_lora_rank'",
    ):
        extract_s1_text_model_dims(config, tp_size=8)


def test_extract_s1_text_model_dims_reports_bad_tp_size():
    with pytest.raises(ValueError, match="tp_size must be positive"):
        extract_s1_text_model_dims(_tiny_text_config(), tp_size=0)

    with pytest.raises(
        ValueError,
        match="num_attention_heads=63 is not divisible by tp_size=8",
    ):
        extract_s1_text_model_dims(_tiny_text_config(num_attention_heads=63), tp_size=8)


def test_s1_token_count_helpers_are_deterministic_and_generic():
    assert s1_decode_token_counts() == (0, 1, 8, 32)
    assert s1_prefill_token_counts() == (257,)
    assert s1_varlen_prefill_sequences() == ((1,), (17, 9, 12), (64, 128, 32))


def test_assert_exact_metadata_equal_accepts_nested_integer_metadata():
    actual = {
        "ids": torch.tensor([1, 3, 5], dtype=torch.int32),
        "pages": (0, 64, 128),
    }
    expected = {
        "ids": torch.tensor([1, 3, 5], dtype=torch.int32),
        "pages": (0, 64, 128),
    }

    assert_exact_metadata_equal(actual, expected)


def test_assert_exact_metadata_equal_rejects_drift():
    with pytest.raises(AssertionError, match="metadata.ids: tensor values differ"):
        assert_exact_metadata_equal(
            {"ids": torch.tensor([1, 3], dtype=torch.int32)},
            {"ids": torch.tensor([1, 4], dtype=torch.int32)},
        )


def test_assert_finite_close_accepts_bf16_and_fp8_tolerances():
    bf16_actual = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    bf16_expected = torch.tensor([1.001, 1.999], dtype=torch.float32)
    assert_finite_close(bf16_actual, bf16_expected, atol=0.01, rtol=0.01)

    fp8_actual = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
    fp8_expected = fp8_actual.to(torch.float32)
    assert_finite_close(fp8_actual, fp8_expected, atol=0.0, rtol=0.0)


def test_assert_finite_close_rejects_nan_and_inf():
    with pytest.raises(AssertionError, match="actual contains NaN or Inf"):
        assert_finite_close(
            torch.tensor([float("nan")]),
            torch.tensor([0.0]),
            atol=0.0,
            rtol=0.0,
        )
    with pytest.raises(AssertionError, match="expected contains NaN or Inf"):
        assert_finite_close(
            torch.tensor([0.0]),
            torch.tensor([float("inf")]),
            atol=0.0,
            rtol=0.0,
        )


def test_assert_selected_amd_kernel_not_reference_accepts_real_kernel(mi350_platform):
    signature = format_signature(x=dense_tensor_format(torch.bfloat16))
    registry = KernelRegistry.get()
    registry.register(
        KernelSpec(
            name="s1_test_reference_decode",
            family="attention",
            mode="decode",
            solution="reference",
            capability=CapabilityRequirement(vendors=frozenset({"amd"})),
            format_signatures=frozenset({signature}),
            priority=0,
        ),
        dummy_impl("s1_test_reference_decode"),
    )
    registry.register(
        KernelSpec(
            name="s1_test_gluon_decode",
            family="attention",
            mode="decode",
            solution="gluon",
            capability=CapabilityRequirement(vendors=frozenset({"amd"})),
            format_signatures=frozenset({signature}),
            priority=12,
        ),
        dummy_impl("s1_test_gluon_decode"),
    )

    selected = assert_selected_amd_kernel_not_reference(
        "attention",
        "decode",
        platform=mi350_platform,
        storage_dtype=torch.bfloat16,
        dtype_roles="x",
        expected_solution="gluon",
    )

    assert selected.name == "s1_test_gluon_decode"


def test_assert_selected_amd_kernel_not_reference_rejects_reference_fallback(mi350_platform):
    signature = format_signature(x=dense_tensor_format(torch.bfloat16))
    KernelRegistry.get().register(
        KernelSpec(
            name="s1_test_reference_decode",
            family="attention",
            mode="decode",
            solution="reference",
            capability=CapabilityRequirement(vendors=frozenset({"amd"})),
            format_signatures=frozenset({signature}),
            priority=12,
        ),
        dummy_impl("s1_test_reference_decode"),
    )

    with pytest.raises(AssertionError, match="selected fallback kernel"):
        assert_selected_amd_kernel_not_reference(
            "attention",
            "decode",
            platform=mi350_platform,
            format_signature=signature,
        )
