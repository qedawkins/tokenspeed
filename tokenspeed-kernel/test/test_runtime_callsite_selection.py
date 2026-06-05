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

"""Safeguard tests for expected kernel selection at each call site."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

# GEMM
import tokenspeed_kernel.numerics.reference.gemm
import tokenspeed_kernel.numerics.reference.moe as _moe_reference
import tokenspeed_kernel.ops.gemm as _gemm_pkg
import tokenspeed_kernel.ops.gemm.deep_gemm
import tokenspeed_kernel.ops.gemm.flashinfer as _gemm_flashinfer
import tokenspeed_kernel.ops.gemm.triton as _gemm_triton

# MoE
import tokenspeed_kernel.ops.moe as _moe_pkg
import tokenspeed_kernel.ops.moe.cuda
import tokenspeed_kernel.ops.moe.deepep
import tokenspeed_kernel.ops.moe.flashinfer
import tokenspeed_kernel.ops.moe.gluon
import tokenspeed_kernel.ops.moe.gluon.combine_gfx950
import tokenspeed_kernel.ops.moe.gluon.dispatch_gfx950
import tokenspeed_kernel.ops.moe.gluon.ep_metadata_gfx950
import tokenspeed_kernel.ops.moe.gluon.experts_fp8_gfx950
import tokenspeed_kernel.ops.moe.gluon.route_topk_gfx950
import tokenspeed_kernel.ops.moe.triton
import tokenspeed_kernel.ops.moe.triton_kernels
import torch
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import FormatSignature

# -- Pre-import so they can be reloaded into the fresh registry. --


# ---------------------------------------------------------------------------
# 1. Real kernel registration via importlib.reload
# ---------------------------------------------------------------------------

_RELOAD_MODULES = [
    # MoE
    _moe_reference,
    tokenspeed_kernel.ops.moe.cuda,
    tokenspeed_kernel.ops.moe.triton,
    tokenspeed_kernel.ops.moe.triton_kernels,
    tokenspeed_kernel.ops.moe.flashinfer,
    tokenspeed_kernel.ops.moe.deepep,
    tokenspeed_kernel.ops.moe.gluon.route_topk_gfx950,
    tokenspeed_kernel.ops.moe.gluon.dispatch_gfx950,
    tokenspeed_kernel.ops.moe.gluon.ep_metadata_gfx950,
    tokenspeed_kernel.ops.moe.gluon.experts_fp8_gfx950,
    tokenspeed_kernel.ops.moe.gluon.combine_gfx950,
    tokenspeed_kernel.ops.moe.gluon,
    _moe_pkg,  # re-registers _MoEOracle
    # GEMM
    tokenspeed_kernel.numerics.reference.gemm,
    tokenspeed_kernel.ops.gemm.deep_gemm,
    _gemm_flashinfer,
    _gemm_triton,
    _gemm_pkg,
]


@pytest.fixture(autouse=True)
def _kernel_registry(fresh_registry):
    """Reload real kernel registrations into a clean registry."""
    for mod in _RELOAD_MODULES:
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# 2. AST-based call-site scanner
# ---------------------------------------------------------------------------

# Maps ``tokenspeed_kernel.<attr>(...)`` to ``(family, mode)``.
_API_MAP: dict[str, tuple[str, str]] = {
    "moe_route": ("moe", "route"),
    "moe_dispatch": ("moe", "dispatch"),
    "moe_experts": ("moe", "experts"),
    "moe_combine": ("moe", "combine"),
    "moe_fused": ("moe", "fused"),
    "mm": ("gemm", "mm"),
}

_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def _try_literal(node: ast.AST) -> tuple[Any, bool]:
    """Return ``(value, True)`` if *node* is a compile-time literal."""
    try:
        return ast.literal_eval(node), True
    except (ValueError, TypeError):
        return None, False


def _try_torch_dtype(node: ast.AST) -> Optional[torch.dtype]:
    """Resolve ``torch.<dtype>`` attribute access."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "torch":
            return _TORCH_DTYPE_MAP.get(node.attr)
    return None


def _extract_literal_dict(node: ast.AST) -> Optional[dict[str, Any]]:
    """Extract a dict literal, keeping only keys/values resolvable at compile time."""
    if not isinstance(node, ast.Dict):
        return None
    result: dict[str, Any] = {}
    for key, value in zip(node.keys, node.values):
        k, k_ok = _try_literal(key)
        if not k_ok or not isinstance(k, str):
            continue
        v, v_ok = _try_literal(value)
        if v_ok:
            result[k] = v
    return result if result else None


def _extract_features(node: ast.AST) -> Optional[set[str]]:
    """Extract a set-literal of feature strings."""
    val, ok = _try_literal(node)
    if ok and isinstance(val, set):
        return val
    return None


# A ``CallSite`` tuple: (family, mode, dtype|None, features|None, traits, weight_format, expected_name, location)
CallSite = tuple[
    str, str, Optional[torch.dtype], Optional[set], dict, Optional[str], str, str
]


def _collect_call_sites(search_dir: Path) -> list[CallSite]:
    """Scan *search_dir* for ``tokenspeed_kernel.<api>(...)`` calls.

    Returns one entry per call whose ``expected_kernel_name`` is a string
    literal.  Calls with a variable or missing ``expected_kernel_name`` are
    silently skipped — they should be covered by ``_MANUAL_CALL_SITES``.
    """
    sites: list[CallSite] = []

    for py_path in sorted(search_dir.rglob("*.py")):
        source = py_path.read_text()
        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _API_MAP:
                continue
            if not (
                isinstance(func.value, ast.Name)
                and func.value.id == "tokenspeed_kernel"
            ):
                continue

            kwargs: dict[str, ast.AST] = {}
            for kw in node.keywords:
                if kw.arg is not None:
                    kwargs[kw.arg] = kw.value

            ekn_node = kwargs.get("expected_kernel_name")
            if ekn_node is None:
                continue
            expected, ok = _try_literal(ekn_node)
            if not ok or not isinstance(expected, str):
                continue

            family, mode = _API_MAP[func.attr]

            # -- dtype --
            dtype_node = kwargs.get("dtype")
            dtype: Optional[torch.dtype] = None
            if dtype_node is not None:
                dtype = _try_torch_dtype(dtype_node)

            # -- features --
            features: Optional[set[str]] = None
            feat_node = kwargs.get("features")
            if feat_node is not None:
                features = _extract_features(feat_node)

            # -- traits --
            traits: dict[str, Any] = {}
            traits_node = kwargs.get("traits")
            if traits_node is not None:
                parsed = _extract_literal_dict(traits_node)
                if parsed is not None:
                    traits = parsed

            weight_format: Optional[str] = None
            weight_format_node = kwargs.get("weight_format")
            if weight_format_node is not None:
                value, ok = _try_literal(weight_format_node)
                if ok and isinstance(value, str):
                    weight_format = value

            lineno = getattr(node, "lineno", 0)
            rel = py_path.relative_to(search_dir.parent.parent.parent)
            location = f"{rel}:{lineno}"

            sites.append(
                (
                    family,
                    mode,
                    dtype,
                    features,
                    traits,
                    weight_format,
                    expected,
                    location,
                )
            )

    return sites


# ---------------------------------------------------------------------------
# 3. Discover call sites
# ---------------------------------------------------------------------------

_SEARCH_DIR = (
    Path(__file__).resolve().parent.parent.parent / "python" / "tokenspeed" / "runtime"
)

_AUTO_SITES = _collect_call_sites(_SEARCH_DIR)

# Call sites that cannot be statically extracted (partial wrappers, variable
# expected_kernel_name, oracle-dependent num_tokens branching, etc.)
_MANUAL_CALL_SITES: list[CallSite] = [
    # -- MoE --
    # topk.py: biased grouped S1 route path uses platform selection.
    (
        "moe",
        "route",
        torch.bfloat16,
        None,
        {
            "output_type": "topk",
            "biased": True,
            "grouped": True,
            "ep": True,
            "num_expert_group": 8,
            "topk_group": 4,
            "topk": 8,
            "num_fused_shared_experts": 0,
        },
        None,
        "gluon_grouped_biased_topk_gfx950",
        "manual:topk/biased_grouped_s1_cdna4",
    ),
    # triton_common.py: local dispatch expected kernel is platform-dependent.
    (
        "moe",
        "dispatch",
        torch.int32,
        None,
        {"comm_strategy": "local"},
        None,
        "triton_moe_align_block_size",
        "manual:triton_common/dispatch_local",
    ),
    (
        "moe",
        "dispatch",
        torch.int32,
        None,
        {"comm_strategy": "local"},
        None,
        "gluon_local_dispatch_gfx950",
        "manual:triton_common/dispatch_local_cdna4",
    ),
    # triton_common.py: partial(tokenspeed_kernel.moe_experts, **_experts_common)
    (
        "moe",
        "experts",
        torch.bfloat16,
        {"dispatch_sorted"},
        {},
        None,
        "triton_moe_fused_experts",
        "manual:triton_common/experts",
    ),
    (
        "moe",
        "experts",
        torch.bfloat16,
        {"dispatch_sorted"},
        {},
        None,
        "gluon_fp8_local_experts_gfx950",
        "manual:triton_common/experts_cdna4",
    ),
    # triton_common.py: moe_combine(..., expected_kernel_name=expected_combine_kernel)
    (
        "moe",
        "combine",
        torch.bfloat16,
        None,
        {"num_tokens": 128, "comm_strategy": None},
        None,
        "triton_moe_sum_reduce",
        "manual:triton_common/combine_large",
    ),
    (
        "moe",
        "combine",
        torch.bfloat16,
        None,
        {"num_tokens": 8, "comm_strategy": None},
        None,
        "torch_compile_moe_sum_reduce",
        "manual:triton_common/combine_small",
    ),
    (
        "moe",
        "combine",
        torch.bfloat16,
        None,
        {"num_tokens": 8, "comm_strategy": None},
        None,
        "gluon_local_sum_reduce_gfx950",
        "manual:triton_common/combine_small_cdna4",
    ),
]

_ALL_SITES = _AUTO_SITES + _MANUAL_CALL_SITES


_DTYPE_PREFERENCE = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int32,
    torch.uint8,
    torch.float8_e4m3fn,
]


def _all_storage_dtypes(spec) -> frozenset[torch.dtype]:
    return frozenset(
        tensor_format.storage_dtype
        for signature in spec.format_signatures
        for _role, tensor_format in signature.roles
    )


def _format_signature_for_any_role(spec, dtype: torch.dtype) -> FormatSignature | None:
    matches = tuple(
        signature
        for signature in sorted(spec.format_signatures, key=str)
        if any(
            tensor_format.storage_dtype == dtype
            for _role, tensor_format in signature.roles
        )
    )
    if len(matches) > 1:
        pytest.skip(f"Kernel {spec.name!r} has multiple signatures for {dtype}")
    return matches[0] if matches else None


def _infer_dtype(expected_name: str) -> torch.dtype:
    spec = KernelRegistry.get().get_by_name(expected_name)
    if spec is not None:
        storage_dtypes = _all_storage_dtypes(spec)
        for dt in _DTYPE_PREFERENCE:
            if dt in storage_dtypes:
                return dt
        if storage_dtypes:
            return next(iter(storage_dtypes))
    return torch.bfloat16


def _infer_format_signature(
    family: str,
    mode: str,
    dtype: torch.dtype,
    weight_format: Optional[str],
    spec,
) -> FormatSignature:
    if family == "moe" and mode == "fused":
        return _moe_pkg._moe_fused_format_signature(dtype, weight_format or "bf16")
    signature = _format_signature_for_any_role(spec, dtype)
    if signature is None:
        pytest.skip(f"Kernel {spec.name!r} has no signature for {dtype}")
    return signature


def _site_id(site: CallSite) -> str:
    """Generate a readable test-id from a call-site tuple."""
    expected = site[6]
    location = site[7]
    return f"{expected}@{location}"


_CDNA4_OVERRIDDEN_MANUAL_SITES = frozenset(
    {
        "manual:triton_common/dispatch_local",
        "manual:triton_common/experts",
        "manual:triton_common/combine_small",
    }
)


def _site_applies_to_platform(site: CallSite, platform) -> bool:
    location = site[7]
    if location.endswith("_cdna4"):
        return platform.is_cdna4
    if platform.is_cdna4 and location in _CDNA4_OVERRIDDEN_MANUAL_SITES:
        return False
    return True


# ---------------------------------------------------------------------------
# 4. Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "site",
    _ALL_SITES,
    ids=[_site_id(s) for s in _ALL_SITES],
)
@pytest.mark.parametrize(
    "platform_name",
    [
        "h100_platform",
        "b200_platform",
        "mi350_platform",
    ],
)
def test_kernel_selection(site, platform_name, request):
    platform = request.getfixturevalue(platform_name)
    family, mode, raw_dtype, features, traits, weight_format, expected, location = site

    if not _site_applies_to_platform(site, platform):
        pytest.skip(f"{location} is not the expected path on {platform.device_name}")

    reg = KernelRegistry.get()
    spec = reg.get_by_name(expected)
    if spec is None:
        pytest.skip(f"Kernel {expected!r} not registered (dependency missing?)")
    if not spec.capability.satisfied_by(platform):
        pytest.skip(
            f"Kernel {expected!r} requires capability not satisfied by "
            f"{platform.device_name} ({platform.arch_version})"
        )

    dtype = raw_dtype or _infer_dtype(expected)
    signature = _infer_format_signature(family, mode, dtype, weight_format, spec)

    result = select_kernel(
        family,
        mode,
        signature,
        features=frozenset(features) if features else None,
        traits=traits,
        platform=platform,
    )
    assert result.name == expected, (
        f"Expected '{expected}' but got '{result.name}' "
        f"at {location} on {platform.device_name} "
        f"for {family}.{mode}(signature={signature}, features={features}, traits={traits})"
    )


def test_s1_fp8_tp_runtime_components_construct(monkeypatch, mi350_platform):
    import tokenspeed.runtime.layers.moe.backends.triton_common as triton_common
    import tokenspeed.runtime.layers.moe.topk as topk_mod
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    monkeypatch.setattr(triton_common, "current_platform", lambda: mi350_platform)
    monkeypatch.setattr(topk_mod, "current_platform", lambda: mi350_platform)

    topk = topk_mod.TopK(
        8,
        use_grouped_topk=True,
        topk_group=4,
        num_expert_group=8,
        correction_bias=torch.zeros(16),
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=True,
    )
    route_traits = {
        "output_type": "topk",
        "biased": True,
        "grouped": True,
        "ep": True,
        "num_expert_group": topk.topk_config.num_expert_group,
        "topk_group": topk.topk_config.topk_group,
        "topk": topk.topk_config.top_k,
        "num_fused_shared_experts": topk.topk_config.num_fused_shared_experts,
    }
    assert (
        topk_mod._expected_biased_grouped_topk_kernel(route_traits)
        == "gluon_grouped_biased_topk_gfx950"
    )

    spec = MoELayerSpec(
        top_k=8,
        num_experts=16,
        num_local_experts=16,
        hidden_size=32,
        intermediate_size=24,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )
    layer = SimpleNamespace(
        w13_weight=torch.empty((16, 48, 32), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((16, 32, 24), dtype=torch.float8_e4m3fn),
    )

    gate_up_gemm, down_gemm, get_config_func = triton_common.build_triton_gemms(
        layer,
        spec,
        use_fp8_w8a8=True,
        block_shape=[16, 16],
        dtype_tag="fp8_w8a8",
        gate_up_B_scale=torch.empty((16, 3, 2), dtype=torch.float32),
        down_B_scale=torch.empty((16, 2, 2), dtype=torch.float32),
    )

    assert callable(gate_up_gemm)
    assert callable(down_gemm)
    assert callable(get_config_func)
    assert (
        gate_up_gemm.keywords["expected_kernel_name"]
        == "gluon_fp8_local_experts_gfx950"
    )
    assert (
        down_gemm.keywords["expected_kernel_name"]
        == "gluon_fp8_local_experts_gfx950"
    )
    assert triton_common._expected_local_dispatch_kernel() == "gluon_local_dispatch_gfx950"
    assert triton_common._expected_local_combine_kernel(8) == "gluon_local_sum_reduce_gfx950"
    assert triton_common._expected_local_combine_kernel(257) == "triton_moe_sum_reduce"


def _reference_bf16_local_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate_size = w2_weight.shape[-1]
    for token in range(hidden_states.shape[0]):
        hidden = hidden_states[token].float()
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            gate_up = (hidden @ w13_weight[expert].float().T).to(torch.bfloat16)
            activated = (
                torch.nn.functional.silu(gate_up[:intermediate_size])
                * gate_up[intermediate_size:]
            ).to(torch.bfloat16)
            down = activated.float() @ w2_weight[expert].float().T
            down *= topk_weights[token, slot].float()
            output[token] += down.to(torch.bfloat16).float()
    return output.to(hidden_states.dtype)


def _make_per_channel_fp8_weight(
    dense: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.platform import current_platform

    fp8 = current_platform().fp8e4m3fn
    scale = torch.clamp(
        dense.float().abs().amax(dim=2, keepdim=True) / fp8.max,
        min=1e-6,
    )
    quantized = torch.clamp(
        dense.float() / scale,
        min=fp8.min,
        max=fp8.max,
    ).to(fp8.dtype)
    return quantized, scale


def _w8a8_per_channel_matmul(
    A: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert: int,
) -> torch.Tensor:
    from tokenspeed_kernel.ops.gemm.fp8_utils import scaled_fp8_quant

    A_fp8, A_scale = scaled_fp8_quant(
        A.contiguous(),
        None,
        use_per_token_if_dynamic=True,
    )
    A_dequantized = A_fp8.float() * A_scale.float()
    weight_dequantized = weight[expert].float() * weight_scale[expert].float()
    return A_dequantized @ weight_dequantized.T


def _reference_w8a8_local_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate_size = w2_weight.shape[-1]
    for token in range(hidden_states.shape[0]):
        hidden = hidden_states[token : token + 1]
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            gate_up = _w8a8_per_channel_matmul(
                hidden,
                w13_weight,
                w13_weight_scale,
                expert,
            )[0].to(torch.bfloat16)
            activated = (
                torch.nn.functional.silu(gate_up[:intermediate_size])
                * gate_up[intermediate_size:]
            ).reshape(1, intermediate_size).to(torch.bfloat16)
            down = _w8a8_per_channel_matmul(
                activated,
                w2_weight,
                w2_weight_scale,
                expert,
            )[0]
            down *= topk_weights[token, slot].float()
            output[token] += down.to(torch.bfloat16).float()
    return output.to(hidden_states.dtype)


def test_s2_bf16_tp_runtime_forward_uses_existing_contracts(
    device: str,
    monkeypatch,
    mi350_platform,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP GPU is required for BF16 TP runtime forward coverage")

    import tokenspeed.runtime.layers.moe.backends.triton_common as triton_common
    import tokenspeed.runtime.layers.moe.topk as topk_mod
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    monkeypatch.setattr(triton_common, "current_platform", lambda: mi350_platform)
    monkeypatch.setattr(topk_mod, "current_platform", lambda: mi350_platform)

    torch.manual_seed(7303)
    num_tokens = 8
    hidden_size = 32
    intermediate_size = 24
    num_experts = 4
    top_k = 2
    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )
    layer = SimpleNamespace(
        w13_weight=(
            torch.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device=device,
            )
            * 0.15
        ).bfloat16(),
        w2_weight=(
            torch.randn(num_experts, hidden_size, intermediate_size, device=device)
            * 0.12
        ).bfloat16(),
    )
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device) * 0.20
    ).bfloat16()
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [3, 0],
            [1, 2],
            [0, 3],
            [2, 1],
            [3, 2],
            [1, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.2,
        0.9,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)

    gate_up_gemm, down_gemm, get_config_func = triton_common.build_triton_gemms(
        layer,
        spec,
        use_fp8_w8a8=False,
        block_shape=None,
        dtype_tag="bf16",
        gate_up_B_scale=None,
        down_B_scale=None,
    )
    assert (
        gate_up_gemm.keywords["expected_kernel_name"]
        == "gluon_fp8_local_experts_gfx950"
    )
    assert (
        down_gemm.keywords["expected_kernel_name"]
        == "gluon_fp8_local_experts_gfx950"
    )
    assert triton_common._expected_local_dispatch_kernel() == "gluon_local_dispatch_gfx950"
    assert triton_common._expected_local_combine_kernel(num_tokens) == (
        "gluon_local_sum_reduce_gfx950"
    )

    actual = triton_common.triton_forward(
        gate_up_gemm,
        down_gemm,
        get_config_func,
        "silu",
        layer,
        hidden_states.contiguous(),
        SimpleNamespace(topk_ids=topk_ids, topk_weights=topk_weights),
    )
    torch.cuda.synchronize()

    expected = _reference_bf16_local_moe(
        hidden_states,
        layer.w13_weight,
        layer.w2_weight,
        topk_ids,
        topk_weights,
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.08,
        rtol=0.05,
        check_dtype=False,
    )


def test_s3_w8a8_tp_runtime_forward_uses_existing_contracts(
    device: str,
    monkeypatch,
    mi350_platform,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP GPU is required for W8A8 TP runtime forward coverage")

    import tokenspeed.runtime.layers.moe.backends.triton_common as triton_common
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec

    monkeypatch.setattr(triton_common, "current_platform", lambda: mi350_platform)

    torch.manual_seed(7304)
    num_tokens = 8
    hidden_size = 32
    intermediate_size = 24
    num_experts = 4
    top_k = 2
    spec = MoELayerSpec(
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device)
        * torch.linspace(
            0.04,
            0.90,
            steps=num_tokens,
            device=device,
            dtype=torch.float32,
        ).view(num_tokens, 1)
    ).bfloat16()
    gate_up_channel_scale = torch.linspace(
        0.05,
        1.70,
        steps=2 * intermediate_size,
        device=device,
        dtype=torch.float32,
    ).view(1, 2 * intermediate_size, 1)
    down_channel_scale = torch.linspace(
        0.03,
        1.40,
        steps=hidden_size,
        device=device,
        dtype=torch.float32,
    ).view(1, hidden_size, 1)
    w13_dense = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
        )
        * gate_up_channel_scale
        * 0.16
    )
    w2_dense = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device)
        * down_channel_scale
        * 0.14
    )
    w13_weight, w13_weight_scale = _make_per_channel_fp8_weight(w13_dense)
    w2_weight, w2_weight_scale = _make_per_channel_fp8_weight(w2_dense)
    layer = SimpleNamespace(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
    )
    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [3, 0],
            [0, 3],
            [1, 0],
            [3, 1],
            [0, 1],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.linspace(
        0.2,
        0.9,
        steps=num_tokens * top_k,
        device=device,
        dtype=torch.float32,
    ).view(num_tokens, top_k)

    gate_up_gemm, down_gemm, get_config_func = triton_common.build_triton_gemms(
        layer,
        spec,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        dtype_tag="fp8_w8a8",
        gate_up_B_scale=layer.w13_weight_scale,
        down_B_scale=layer.w2_weight_scale,
    )
    assert (
        gate_up_gemm.keywords["expected_kernel_name"]
        == "gluon_fp8_local_experts_gfx950"
    )
    assert (
        down_gemm.keywords["expected_kernel_name"]
        == "gluon_fp8_local_experts_gfx950"
    )
    assert topk_ids.ne(2).all()

    actual = triton_common.triton_forward(
        gate_up_gemm,
        down_gemm,
        get_config_func,
        "silu",
        layer,
        hidden_states.contiguous(),
        SimpleNamespace(topk_ids=topk_ids, topk_weights=topk_weights),
    )
    torch.cuda.synchronize()

    expected = _reference_w8a8_local_moe(
        hidden_states,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        topk_ids,
        topk_weights,
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.12,
        rtol=0.08,
        check_dtype=False,
    )
