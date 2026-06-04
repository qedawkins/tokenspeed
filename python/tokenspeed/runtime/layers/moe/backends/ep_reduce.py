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

"""Internal EP returned-slot weighted reduce."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.moe.backends.ep_workspace import EPWorkspaceError


def ep_weighted_reduce(
    returned_slots: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
    out: torch.Tensor | None = None,
    use_kernel: bool | None = None,
    expected_kernel_name: str | None = None,
    returned_slots_are_weighted: bool = False,
) -> torch.Tensor:
    """Apply top-k weights to EP returned slots, then reuse ``moe.combine``.

    ``returned_slots`` must be unweighted expert outputs in source
    ``[token, top_k, hidden]`` order. This mirrors the existing combine contract:
    the tensor passed to ``moe.combine`` is already weighted, and combine only
    sums slots plus the scalar routed scaling factor.
    """
    if returned_slots_are_weighted:
        raise EPWorkspaceError(
            "ep_weighted_reduce expects unweighted returned_slots; call "
            "moe_combine directly for already-weighted slots"
        )
    _validate_reduce_inputs(returned_slots, topk_weights, out)
    num_tokens, _top_k, hidden_size = returned_slots.shape
    if out is None:
        out = torch.empty(
            (num_tokens, hidden_size),
            dtype=returned_slots.dtype,
            device=returned_slots.device,
        )

    if use_kernel is None:
        use_kernel = returned_slots.device.type == "cuda"

    if num_tokens == 0:
        return out

    weighted_slots = (
        returned_slots.float() * topk_weights.float().unsqueeze(-1)
    ).to(returned_slots.dtype)

    if use_kernel:
        import tokenspeed_kernel

        tokenspeed_kernel.moe_combine(
            weighted_slots,
            out,
            float(routed_scaling_factor),
            dtype=returned_slots.dtype,
            traits={"num_tokens": num_tokens, "comm_strategy": None},
            expected_kernel_name=expected_kernel_name,
        )
    else:
        reduced = weighted_slots.float().sum(dim=1) * float(routed_scaling_factor)
        out.copy_(reduced.to(out.dtype))
    return out


def _validate_reduce_inputs(
    returned_slots: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if returned_slots.ndim != 3:
        raise EPWorkspaceError(
            f"returned_slots must be rank-3, got {tuple(returned_slots.shape)}"
        )
    if topk_weights.shape != returned_slots.shape[:2]:
        raise EPWorkspaceError(
            f"topk_weights shape {tuple(topk_weights.shape)} != returned slot shape "
            f"{tuple(returned_slots.shape[:2])}"
        )
    if topk_weights.device != returned_slots.device:
        raise EPWorkspaceError(
            f"topk_weights device {topk_weights.device} != returned_slots device "
            f"{returned_slots.device}"
        )
    if topk_weights.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise EPWorkspaceError(
            f"topk_weights must be floating point, got {topk_weights.dtype}"
        )
    if out is not None:
        expected_shape = (returned_slots.shape[0], returned_slots.shape[2])
        if out.shape != expected_shape:
            raise EPWorkspaceError(f"out shape {tuple(out.shape)} != {expected_shape}")
        if out.dtype != returned_slots.dtype:
            raise EPWorkspaceError(f"out dtype {out.dtype} != {returned_slots.dtype}")
        if out.device != returned_slots.device:
            raise EPWorkspaceError(
                f"out device {out.device} != returned_slots device "
                f"{returned_slots.device}"
            )


__all__ = ["ep_weighted_reduce"]
