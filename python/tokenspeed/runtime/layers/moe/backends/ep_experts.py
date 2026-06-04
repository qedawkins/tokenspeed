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

"""Internal owner-rank EP expert GEMM helpers.

The EP dispatch path produces owner-local token buffers arranged as ragged
expert slices. This module adapts that owner-rank layout to the existing
``moe.experts`` sorted-dispatch contract so the GFX950 FP8 Gluon expert GEMM can
run without adding a public EP backend mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.layers.moe.backends.ep_workspace import EPWorkspaceError


@dataclass(frozen=True)
class EPOwnerExpertMetadata:
    local_expert_counts: torch.Tensor
    local_expert_offsets: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def num_rows(self) -> int:
        return int(self.local_expert_offsets[-1].item())


def ep_expert_config(
    *,
    block_size: int,
    block_shape: tuple[int, int],
) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": block_shape[0],
        "BLOCK_SIZE_K": block_shape[1],
        "GROUP_SIZE_M": 1,
        "num_warps": 1,
        "num_stages": 1,
    }


def build_owner_expert_metadata(
    local_expert_counts: torch.Tensor,
    *,
    block_size: int,
    local_expert_offsets: torch.Tensor | None = None,
) -> EPOwnerExpertMetadata:
    _validate_counts(local_expert_counts)
    if block_size <= 0:
        raise EPWorkspaceError(f"block_size must be positive, got {block_size}")

    if local_expert_offsets is None:
        local_expert_offsets = _offsets_from_counts(local_expert_counts)
    else:
        _validate_offsets(local_expert_offsets, local_expert_counts)

    device = local_expert_counts.device
    num_rows = int(local_expert_offsets[-1].item())
    padded_counts = _round_counts_to_block(local_expert_counts, block_size)
    padded_offsets = _offsets_from_counts(padded_counts)
    total_padded = int(padded_offsets[-1].item())
    num_tokens_post_padded = torch.tensor(
        [total_padded],
        dtype=torch.int32,
        device=device,
    )

    if total_padded == 0:
        return EPOwnerExpertMetadata(
            local_expert_counts=local_expert_counts,
            local_expert_offsets=local_expert_offsets,
            sorted_token_ids=torch.empty((0,), dtype=torch.int32, device=device),
            expert_ids=torch.empty((0,), dtype=torch.int32, device=device),
            num_tokens_post_padded=num_tokens_post_padded,
        )

    padded_rows = torch.arange(total_padded, dtype=torch.int32, device=device)
    expert_for_row = torch.bucketize(
        padded_rows,
        padded_offsets[1:].contiguous(),
        right=True,
    )
    row_in_padded_slice = (
        padded_rows - padded_offsets.index_select(0, expert_for_row).to(torch.int32)
    )
    slice_counts = local_expert_counts.index_select(0, expert_for_row).to(torch.int32)
    slice_offsets = local_expert_offsets.index_select(0, expert_for_row).to(torch.int32)
    valid = row_in_padded_slice < slice_counts
    invalid_slot = torch.full_like(padded_rows, num_rows)
    sorted_token_ids = torch.where(
        valid,
        slice_offsets + row_in_padded_slice,
        invalid_slot,
    )

    num_blocks = total_padded // block_size
    block_starts = torch.arange(num_blocks, dtype=torch.int32, device=device) * block_size
    expert_ids = torch.bucketize(
        block_starts,
        padded_offsets[1:].contiguous(),
        right=True,
    ).to(torch.int32)

    return EPOwnerExpertMetadata(
        local_expert_counts=local_expert_counts,
        local_expert_offsets=local_expert_offsets,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


def owner_rank_fp8_expert_gemm(
    owner_tokens: torch.Tensor,
    local_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    local_expert_counts: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    block_size: int,
    local_expert_offsets: torch.Tensor | None = None,
    routed_weights: torch.Tensor | None = None,
    mul_routed_weight: bool = False,
    out: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
    expected_kernel_name: str | None = None,
) -> torch.Tensor:
    metadata = build_owner_expert_metadata(
        local_expert_counts,
        block_size=block_size,
        local_expert_offsets=local_expert_offsets,
    )
    _validate_gemm_inputs(
        owner_tokens,
        local_weight,
        local_weight_scale,
        metadata,
        block_shape=block_shape,
        routed_weights=routed_weights,
        out=out,
    )
    if out is None:
        out = torch.empty(
            (owner_tokens.shape[0], local_weight.shape[1]),
            dtype=owner_tokens.dtype,
            device=owner_tokens.device,
        )

    if owner_tokens.shape[0] == 0:
        return out

    if routed_weights is None:
        topk_weights = torch.ones(
            (owner_tokens.shape[0], 1),
            dtype=torch.float32,
            device=owner_tokens.device,
        )
    else:
        topk_weights = routed_weights.reshape(owner_tokens.shape[0], 1).contiguous()
    topk_ids = torch.zeros(
        (owner_tokens.shape[0], 1),
        dtype=torch.int32,
        device=owner_tokens.device,
    )
    if config is None:
        config = ep_expert_config(block_size=block_size, block_shape=block_shape)

    import tokenspeed_kernel
    from tokenspeed_kernel._triton import tl

    tokenspeed_kernel.moe_experts(
        owner_tokens,
        local_weight,
        None,
        out,
        None,
        local_weight_scale,
        topk_weights,
        topk_ids,
        metadata.sorted_token_ids,
        metadata.expert_ids,
        metadata.num_tokens_post_padded,
        mul_routed_weight,
        1,
        config,
        tl.bfloat16,
        True,
        False,
        False,
        False,
        block_shape=list(block_shape),
        dtype=owner_tokens.dtype,
        features={"dispatch_sorted"},
        expected_kernel_name=expected_kernel_name,
    )
    return out


def _offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        (counts.numel() + 1,),
        dtype=torch.int32,
        device=counts.device,
    )
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return offsets


def _round_counts_to_block(counts: torch.Tensor, block_size: int) -> torch.Tensor:
    return ((counts + block_size - 1) // block_size) * block_size


def _validate_counts(local_expert_counts: torch.Tensor) -> None:
    if local_expert_counts.ndim != 1:
        raise EPWorkspaceError(
            f"local_expert_counts must be rank-1, got {tuple(local_expert_counts.shape)}"
        )
    if local_expert_counts.dtype != torch.int32:
        raise EPWorkspaceError(
            f"local_expert_counts must be torch.int32, got {local_expert_counts.dtype}"
        )
    if local_expert_counts.numel() == 0:
        raise EPWorkspaceError("local_expert_counts must describe at least one expert")
    if bool(local_expert_counts.lt(0).any().item()):
        raise EPWorkspaceError("local_expert_counts must be non-negative")


def _validate_offsets(
    local_expert_offsets: torch.Tensor,
    local_expert_counts: torch.Tensor,
) -> None:
    expected_shape = (local_expert_counts.numel() + 1,)
    if local_expert_offsets.shape != expected_shape:
        raise EPWorkspaceError(
            f"local_expert_offsets shape {tuple(local_expert_offsets.shape)} != "
            f"{expected_shape}"
        )
    if local_expert_offsets.dtype != torch.int32:
        raise EPWorkspaceError(
            f"local_expert_offsets must be torch.int32, got {local_expert_offsets.dtype}"
        )
    if local_expert_offsets.device != local_expert_counts.device:
        raise EPWorkspaceError(
            "local_expert_offsets must be on the same device as local_expert_counts"
        )
    if int(local_expert_offsets[0].item()) != 0:
        raise EPWorkspaceError("local_expert_offsets must start at zero")
    diffs = local_expert_offsets[1:] - local_expert_offsets[:-1]
    if bool(diffs.ne(local_expert_counts).any().item()):
        raise EPWorkspaceError(
            "local_expert_offsets must match local_expert_counts prefix sums"
        )


def _validate_gemm_inputs(
    owner_tokens: torch.Tensor,
    local_weight: torch.Tensor,
    local_weight_scale: torch.Tensor,
    metadata: EPOwnerExpertMetadata,
    *,
    block_shape: tuple[int, int],
    routed_weights: torch.Tensor | None,
    out: torch.Tensor | None,
) -> None:
    if owner_tokens.ndim != 2:
        raise EPWorkspaceError(
            f"owner_tokens must be rank-2, got {tuple(owner_tokens.shape)}"
        )
    if local_weight.ndim != 3:
        raise EPWorkspaceError(
            f"local_weight must be rank-3, got {tuple(local_weight.shape)}"
        )
    if local_weight_scale.ndim != 3:
        raise EPWorkspaceError(
            "local_weight_scale must be rank-3, got "
            f"{tuple(local_weight_scale.shape)}"
        )
    if len(block_shape) != 2 or block_shape[0] <= 0 or block_shape[1] <= 0:
        raise EPWorkspaceError(f"invalid block_shape {block_shape!r}")
    if local_weight.dtype not in _fp8_e4m3_dtypes():
        raise EPWorkspaceError(f"local_weight must be FP8 E4M3, got {local_weight.dtype}")
    if local_weight_scale.dtype != torch.float32:
        raise EPWorkspaceError(
            f"local_weight_scale must be torch.float32, got {local_weight_scale.dtype}"
        )
    expected_scale_shape = (
        local_weight.shape[0],
        _ceil_div(local_weight.shape[1], block_shape[0]),
        _ceil_div(local_weight.shape[2], block_shape[1]),
    )
    if local_weight_scale.shape != expected_scale_shape:
        raise EPWorkspaceError(
            f"local_weight_scale shape {tuple(local_weight_scale.shape)} != "
            f"{expected_scale_shape}"
        )
    if local_weight.shape[0] != metadata.local_expert_counts.numel():
        raise EPWorkspaceError(
            f"local_weight experts {local_weight.shape[0]} != "
            f"{metadata.local_expert_counts.numel()}"
        )
    if local_weight_scale.shape[0] != local_weight.shape[0]:
        raise EPWorkspaceError(
            f"local_weight_scale experts {local_weight_scale.shape[0]} != "
            f"{local_weight.shape[0]}"
        )
    if local_weight.shape[2] != owner_tokens.shape[1]:
        raise EPWorkspaceError(
            f"local_weight K {local_weight.shape[2]} != owner token hidden "
            f"{owner_tokens.shape[1]}"
        )
    if metadata.num_rows != owner_tokens.shape[0]:
        raise EPWorkspaceError(
            f"metadata rows {metadata.num_rows} != owner token rows "
            f"{owner_tokens.shape[0]}"
        )
    for name, tensor in (
        ("local_weight", local_weight),
        ("local_weight_scale", local_weight_scale),
        ("local_expert_counts", metadata.local_expert_counts),
        ("local_expert_offsets", metadata.local_expert_offsets),
    ):
        if tensor.device != owner_tokens.device:
            raise EPWorkspaceError(
                f"{name} device {tensor.device} != owner_tokens device "
                f"{owner_tokens.device}"
            )
    if routed_weights is not None:
        if routed_weights.numel() != owner_tokens.shape[0]:
            raise EPWorkspaceError(
                f"routed_weights elements {routed_weights.numel()} != "
                f"owner token rows {owner_tokens.shape[0]}"
            )
        if routed_weights.device != owner_tokens.device:
            raise EPWorkspaceError(
                f"routed_weights device {routed_weights.device} != owner_tokens "
                f"device {owner_tokens.device}"
            )
    if out is not None:
        expected_shape = (owner_tokens.shape[0], local_weight.shape[1])
        if out.shape != expected_shape:
            raise EPWorkspaceError(f"out shape {tuple(out.shape)} != {expected_shape}")
        if out.dtype != owner_tokens.dtype:
            raise EPWorkspaceError(f"out dtype {out.dtype} != {owner_tokens.dtype}")
        if out.device != owner_tokens.device:
            raise EPWorkspaceError(
                f"out device {out.device} != owner_tokens device {owner_tokens.device}"
            )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _fp8_e4m3_dtypes() -> tuple[torch.dtype, ...]:
    return tuple(
        dtype
        for name in ("float8_e4m3fn", "float8_e4m3fnuz")
        if (dtype := getattr(torch, name, None)) is not None
    )


__all__ = [
    "EPOwnerExpertMetadata",
    "build_owner_expert_metadata",
    "ep_expert_config",
    "owner_rank_fp8_expert_gemm",
]
