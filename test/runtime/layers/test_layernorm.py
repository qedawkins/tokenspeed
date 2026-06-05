from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm


def _require_cdna4_gpu() -> None:
    from tokenspeed_kernel.platform import current_platform

    if not torch.cuda.is_available() or not current_platform().is_cdna4_plus:
        pytest.skip("AMD CDNA4 GPU is required for HIP RMSNorm fallback tests")


def test_fused_rmsnorm_hip_fallback_matches_individual_norms() -> None:
    _require_cdna4_gpu()

    torch.manual_seed(20260605)
    q_norm = RMSNorm(16, eps=1e-6).cuda().to(torch.bfloat16)
    kv_norm = RMSNorm(8, eps=1e-6).cuda().to(torch.bfloat16)
    fused_norm = FusedRMSNorm(q_norm, kv_norm)

    q = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    expected_q = q_norm(q.clone())
    expected_kv = kv_norm(kv.clone())

    q_out = torch.empty_like(q)
    kv_inout = kv.clone()
    actual_q, actual_kv = fused_norm(q.clone(), kv_inout, output_q_a=q_out)
    torch.cuda.synchronize()

    assert actual_q.data_ptr() == q_out.data_ptr()
    assert actual_kv.data_ptr() == kv_inout.data_ptr()
    torch.testing.assert_close(actual_q, expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_kv, expected_kv, atol=1e-2, rtol=1e-2)
