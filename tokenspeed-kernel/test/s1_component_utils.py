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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    SelectedKernel,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import FormatSignature


@dataclass(frozen=True)
class S1TextModelDims:
    H: int
    E: int
    K: int
    I: int  # noqa: E741 - Kimi config shorthand for moe_intermediate_size.
    A: int
    D_nope: int
    D_rope: int
    D_v: int
    R_kv: int
    A_r: int


_DIMENSION_FIELDS = {
    "H": "hidden_size",
    "E": "n_routed_experts",
    "K": "num_experts_per_tok",
    "I": "moe_intermediate_size",
    "A": "num_attention_heads",
    "D_nope": "qk_nope_head_dim",
    "D_rope": "qk_rope_head_dim",
    "D_v": "v_head_dim",
    "R_kv": "kv_lora_rank",
}

_FALLBACK_SOLUTIONS = frozenset({"reference", "pytorch", "torch", "cpu"})
_FALLBACK_NAME_FRAGMENTS = ("reference", "pytorch", "torch", "cpu")


def _field(config: object, name: str) -> object:
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
        raise AttributeError(name)
    return getattr(config, name)


def _has_field(config: object, name: str) -> bool:
    if isinstance(config, Mapping):
        return name in config
    return hasattr(config, name)


def extract_s1_text_model_dims(config: object, *, tp_size: int) -> S1TextModelDims:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")

    text_config = (
        _field(config, "text_config") if _has_field(config, "text_config") else config
    )
    values: dict[str, int] = {}
    for output_name, source_name in _DIMENSION_FIELDS.items():
        try:
            value = _field(text_config, source_name)
        except AttributeError as exc:
            raise ValueError(
                f"missing required text config field {source_name!r}"
            ) from exc
        values[output_name] = int(value)

    attention_heads = values["A"]
    if attention_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads={attention_heads} is not divisible by tp_size={tp_size}"
        )

    return S1TextModelDims(**values, A_r=attention_heads // tp_size)


def s1_decode_token_counts() -> tuple[int, ...]:
    return (0, 1, 8, 32)


def s1_prefill_token_counts() -> tuple[int, ...]:
    return (257,)


def s1_varlen_prefill_sequences() -> tuple[tuple[int, ...], ...]:
    return ((1,), (17, 9, 12), (64, 128, 32))


def assert_exact_metadata_equal(
    actual: object,
    expected: object,
    *,
    name: str = "metadata",
) -> None:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise AssertionError(f"{name}: tensor/non-tensor mismatch")
        if actual.dtype != expected.dtype:
            raise AssertionError(
                f"{name}: dtype mismatch {actual.dtype} != {expected.dtype}"
            )
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        if not torch.equal(actual, expected):
            raise AssertionError(f"{name}: tensor values differ")
        return

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise AssertionError(f"{name}: mapping/non-mapping mismatch")
        if set(actual) != set(expected):
            raise AssertionError(
                f"{name}: keys differ {set(actual)} != {set(expected)}"
            )
        for key in actual:
            assert_exact_metadata_equal(
                actual[key],
                expected[key],
                name=f"{name}.{key}",
            )
        return

    if (
        isinstance(actual, Sequence)
        and isinstance(expected, Sequence)
        and not isinstance(actual, (str, bytes))
        and not isinstance(expected, (str, bytes))
    ):
        if len(actual) != len(expected):
            raise AssertionError(
                f"{name}: length mismatch {len(actual)} != {len(expected)}"
            )
        for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_exact_metadata_equal(
                actual_item,
                expected_item,
                name=f"{name}[{idx}]",
            )
        return

    if actual != expected:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def assert_finite_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str = "tensor",
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )

    def _comparison_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(torch.float32)
        return tensor

    actual_cmp = _comparison_tensor(actual)
    expected_cmp = _comparison_tensor(expected)
    for label, tensor in (("actual", actual_cmp), ("expected", expected_cmp)):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise AssertionError(f"{name}: {label} contains NaN or Inf")

    torch.testing.assert_close(
        actual_cmp,
        expected_cmp,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        check_dtype=False,
    )


def _unique_signature_for_dtype(
    family: str,
    mode: str,
    *,
    storage_dtype: torch.dtype,
    dtype_roles: str | Iterable[str],
    features: frozenset[str] | None,
    traits: dict[str, Any] | None,
    platform: PlatformInfo,
) -> FormatSignature:
    candidates = KernelRegistry.get().get_for_operator(
        family,
        mode,
        features=features,
        platform=platform,
    )
    if traits:
        candidates = [spec for spec in candidates if spec_matches_traits(spec, traits)]

    signatures: set[FormatSignature] = set()
    for spec in candidates:
        signatures.update(
            spec.format_signatures_for_storage_dtype(storage_dtype, dtype_roles)
        )

    if not signatures:
        raise AssertionError(
            f"no {family}.{mode} format signature found for storage dtype={storage_dtype}"
        )
    if len(signatures) > 1:
        rendered = ", ".join(
            str(signature) for signature in sorted(signatures, key=str)
        )
        raise ValueError(
            f"ambiguous {family}.{mode} format signatures for storage dtype={storage_dtype}: "
            f"{rendered}"
        )
    return next(iter(signatures))


def assert_selected_amd_kernel_not_reference(
    family: str,
    mode: str,
    *,
    platform: PlatformInfo,
    format_signature: FormatSignature | None = None,
    storage_dtype: torch.dtype | None = None,
    dtype_roles: str | Iterable[str] | None = None,
    features: frozenset[str] | None = None,
    traits: dict[str, Any] | None = None,
    expected_solution: str | None = None,
) -> SelectedKernel:
    if platform.vendor != "amd":
        raise ValueError(
            "assert_selected_amd_kernel_not_reference requires an AMD platform, "
            f"got {platform.vendor!r}"
        )

    if format_signature is None:
        if storage_dtype is None or dtype_roles is None:
            raise ValueError(
                "provide either format_signature or both storage_dtype and dtype_roles"
            )
        format_signature = _unique_signature_for_dtype(
            family,
            mode,
            storage_dtype=storage_dtype,
            dtype_roles=dtype_roles,
            features=features,
            traits=traits,
            platform=platform,
        )

    selected = select_kernel(
        family,
        mode,
        format_signature,
        features=features,
        platform=platform,
        traits=traits,
    )
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise AssertionError(
            f"selected kernel {selected.name!r} is missing from registry"
        )

    _assert_not_reference_fallback(spec, platform)
    if expected_solution is not None and spec.solution != expected_solution:
        raise AssertionError(
            f"{family}.{mode} selected solution {spec.solution!r}, expected {expected_solution!r}"
        )
    return selected


def _assert_not_reference_fallback(spec: KernelSpec, platform: PlatformInfo) -> None:
    solution = spec.solution.lower()
    name = spec.name.lower()
    if solution in _FALLBACK_SOLUTIONS or any(
        fragment in name for fragment in _FALLBACK_NAME_FRAGMENTS
    ):
        raise AssertionError(
            f"{spec.family}.{spec.mode} selected fallback kernel {spec.name!r} "
            f"(solution={spec.solution!r}) on {platform.device_name}"
        )
