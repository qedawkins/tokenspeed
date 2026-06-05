"""Regression tests for logits processing helpers."""

from __future__ import annotations

# ruff: noqa: E402

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

import pytest
import torch

import tokenspeed.runtime.layers.logits_processor as logits_processor_module
from tokenspeed.runtime.layers.logits_processor import fused_softcap


def test_lm_head_matmul_falls_back_when_lazy_fused_library_missing(monkeypatch):
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor(
        [[1.0, 0.0], [0.5, 1.5], [2.0, -1.0]],
        dtype=torch.float32,
    )
    calls = {"fused": 0}

    def should_use_fused(hidden, lm_head_weight):
        return True

    def missing_fused_library(hidden, lm_head_weight, *, enable_pdl=False):
        calls["fused"] += 1
        raise RuntimeError("tokenspeed_kernel lm_head_gemm library not found at /tmp/x")

    monkeypatch.setattr(
        logits_processor_module,
        "_FUSED_LM_HEAD_GEMM",
        (should_use_fused, missing_fused_library),
    )

    out = logits_processor_module._lm_head_matmul(hidden_states, weight)

    torch.testing.assert_close(out, torch.matmul(hidden_states, weight.T))
    assert calls["fused"] == 1
    assert logits_processor_module._FUSED_LM_HEAD_GEMM == (None, None)


def test_lm_head_matmul_reraises_fused_kernel_runtime_errors(monkeypatch):
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.eye(2, dtype=torch.float32)

    def should_use_fused(hidden, lm_head_weight):
        return True

    def broken_fused_kernel(hidden, lm_head_weight, *, enable_pdl=False):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(
        logits_processor_module,
        "_FUSED_LM_HEAD_GEMM",
        (should_use_fused, broken_fused_kernel),
    )

    with pytest.raises(RuntimeError, match="kernel launch failed"):
        logits_processor_module._lm_head_matmul(hidden_states, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_softcap_handles_large_logits_without_nan():
    cap = 30.0
    logits = torch.tensor(
        [[5000.0, 2000.0, 1500.0, 100.0, 0.0, -100.0, -1500.0, -5000.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected = cap * torch.tanh(logits / cap)

    out = fused_softcap(logits.clone(), cap)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=2e-5)
