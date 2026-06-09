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

import math

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.cuda  # noqa: F401
import tokenspeed_kernel.ops.attention.flash_attn  # noqa: F401
import tokenspeed_kernel.ops.attention.flash_mla  # noqa: F401
import tokenspeed_kernel.ops.attention.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.attention.gluon  # noqa: F401
import tokenspeed_kernel.ops.attention.triton  # noqa: F401
import torch
from tokenspeed_kernel.ops.attention.flash_attn import mha_decode_scheduler_metadata
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


__all__ = [
    "mha_prefill",
    "mha_extend_with_kvcache",
    "mha_decode_with_kvcache",
    "mha_merge_state",
    "mha_decode_scheduler_metadata",
    "mla_prefill",
    "mla_prefill_with_kvcache",
    "mla_decode_with_kvcache",
]

LSE_LN = math.log2(math.e)


def _requires_logit_cap(logit_cap: float) -> bool:
    return logit_cap != 0.0


def _mla_page_size(kv_cache: torch.Tensor) -> int:
    if kv_cache.dim() == 3:
        return kv_cache.shape[1]
    if kv_cache.shape[1] == 1:
        return kv_cache.shape[2]
    return kv_cache.shape[1]


def mha_prefill(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    # attention options
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MHA prefill from uncached KV.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        k: Key tensor with shape [total_kv, num_kv_heads, head_dim].
        v: Value tensor with shape [total_kv, num_kv_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths with shape [batch + 1].
            KV cumulative sequence lengths are assumed to be identical.
        cu_seqlens_cpu: Host-side cumulative sequence lengths as a strict
            list[int]. Used for host-side launch metadata; must match cu_seqlens.
        max_seqlen: Maximum sequence length.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Standard full-sequence prefill assumes query and KV sequence boundaries match.
    """
    batch_size = cu_seqlens.shape[0] - 1

    # Select kernel
    traits = {
        "head_dim": q.shape[-1],
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "mha_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": batch_size,
        "total_q": q.shape[0],
        "total_kv": k.shape[0],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "max_seqlen": max_seqlen,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=max_seqlen,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
        )


def mha_extend_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # attention options
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MHA extend with paged KV cache.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Visible KV lengths in the cache, shape [batch]. Query
            lengths are independent and may be smaller than KV lengths.
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        is_causal: Whether query tokens are a causal suffix of cached KV.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Each request's query tokens attend all visible cached KV tokens.
    """
    # Select kernel
    traits = {
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    kernel = select_kernel(
        "attention",
        "mha_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2],
        "head_dim": q.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_extend_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )


def mha_decode_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    # attention options
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    scheduler_metadata: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MHA decode with paged KV cache.

    Args:
        q: Query tensor with shape [batch, num_q_heads, head_dim].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths after appending current decode tokens, shape [batch].
        max_seqlen_k: Maximum KV length.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.
    """
    if q.shape[0] != cache_seqlens.shape[0]:
        raise ValueError(
            "mha_decode_with_kvcache assumes query length 1; "
            f"got q.shape[0]={q.shape[0]} and batch={cache_seqlens.shape[0]}"
        )

    # Select kernel
    traits = {
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    kernel = select_kernel(
        "attention",
        "mha_decode_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2],
        "head_dim": q.shape[-1],
        "max_seqlen_q": 1,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_decode_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_decode_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel_kwargs = dict(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
            max_seqlen_k=max_seqlen_k,
        )
        # Only the FA3 path accepts pre-computed scheduler metadata; other
        # backends would reject the unknown kwarg.
        if scheduler_metadata is not None:
            kernel_kwargs["scheduler_metadata"] = scheduler_metadata
        return kernel(**kernel_kwargs)


def mha_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    lse_scale_log2: float = LSE_LN,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two MHA partial attention states.

    Args:
        out_a: First partial output with shape [total_q, num_heads, head_dim].
        lse_a: First partial log-sum-exp with shape [total_q, num_heads].
        out_b: Second partial output with shape [total_q, num_heads, head_dim].
        lse_b: Second partial log-sum-exp with shape [total_q, num_heads].
        lse_scale_log2: Multiplier that converts input LSE to log2 domain.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.
    """
    traits = {
        "head_dim": out_a.shape[-1],
    }
    signature = _attention_format_signature(out_a=out_a, out_b=out_b)
    kernel = select_kernel(
        "attention",
        "mha_merge_state",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "total_q": out_a.shape[0],
        "num_heads": out_a.shape[1],
        "head_dim": out_a.shape[2],
    }
    ShapeCapture.get().record(
        "attention",
        "mha_merge_state",
        kernel.name,
        out_a.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mha_merge_state",
        out_a.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            out_a=out_a,
            lse_a=lse_a,
            out_b=out_b,
            lse_b=lse_b,
            lse_scale_log2=lse_scale_log2,
        )


def mla_prefill(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # attention options
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Ragged MLA prefill without reading a paged KV cache.

    Args:
        q_nope: Query no-PE component with shape
            [total_q, num_q_heads, qk_nope_head_dim].
        q_pe: Query RoPE component with shape
            [total_q, num_q_heads, qk_rope_head_dim].
        k_nope: Key no-PE component with shape
            [total_kv, num_kv_heads, qk_nope_head_dim].
        k_pe: Key RoPE component with shape
            [total_kv, num_kv_heads, qk_rope_head_dim].
        v: Value tensor with shape [total_kv, num_kv_heads, v_head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        cu_seqlens_kv: KV cumulative sequence lengths with shape [batch + 1].
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        softmax_scale: Optional scale factor applied before softmax.
        is_causal: Whether to apply causal masking.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        return_lse: Whether to also return log-sum-exp values.
        override: Optional kernel override name.

    Returns:
        Attention output with shape [total_q, num_q_heads, v_head_dim], or
        ``(output, lse)`` when ``return_lse`` is true.
    """

    traits = {
        "num_q_heads": q_nope.shape[1],
        "num_kv_heads": k_nope.shape[1],
        "kv_lora_rank": q_nope.shape[-1],
        "qk_rope_head_dim": q_pe.shape[-1],
        "v_head_dim": v.shape[-1],
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": _requires_logit_cap(logit_cap),
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(
        q_nope=q_nope,
        q_pe=q_pe,
        k_nope=k_nope,
        k_pe=k_pe,
        v=v,
    )
    kernel = select_kernel(
        "attention",
        "mla_prefill",
        signature,
        features=frozenset({"mla"}),
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cu_seqlens_q.shape[0] - 1,
        "total_q": q_nope.shape[0],
        "total_kv": k_nope.shape[0],
        "num_q_heads": q_nope.shape[1],
        "num_kv_heads": k_nope.shape[1],
        "kv_lora_rank": q_nope.shape[-1],
        "qk_rope_head_dim": q_pe.shape[-1],
        "v_head_dim": v.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_prefill",
        kernel.name,
        q_nope.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_prefill",
        q_nope.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            return_lse=return_lse,
        )


def mla_prefill_with_kvcache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # attention options
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    qk_nope_head_dim: int | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Ragged MLA prefill using a paged compressed MLA KV cache.

    The cache is not stored as separated full K/V tensors. It stores the
    compressed latent KV slice followed by the RoPE key slice in one tensor:
    ``[ckv | k_pe]``.

    Args:
        q_nope: Query no-PE component in the cached attention space with shape
            [total_q, num_q_heads, kv_lora_rank].
        q_pe: Query RoPE component with shape
            [total_q, num_q_heads, qk_rope_head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        kv_cache: Paged compressed MLA KV cache with shape
            [num_pages, page_size, kv_lora_rank + qk_rope_head_dim] or
            [num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths, shape [batch].
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        softmax_scale: Optional scale factor applied before softmax.
        is_causal: Whether to apply causal masking.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        return_lse: Whether to also return log-sum-exp values.
        qk_nope_head_dim: Optional original no-PE key dimension for kernels that
            need it when ``q_nope`` has already been absorbed to ``kv_lora_rank``.
        override: Optional kernel override name.

    Returns:
        Latent attention output with shape [total_q, num_q_heads, kv_lora_rank],
        or ``(output, lse)`` when ``return_lse`` is true.
    """

    page_size = _mla_page_size(kv_cache)
    traits = {
        "num_q_heads": q_nope.shape[1],
        "num_kv_heads": 1,
        "kv_lora_rank": q_nope.shape[-1],
        "qk_rope_head_dim": q_pe.shape[-1],
        "v_head_dim": q_nope.shape[-1],
        "page_size": page_size,
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": _requires_logit_cap(logit_cap),
        "kv_cache_mode": "cached",
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q_nope=q_nope, q_pe=q_pe, kv_cache=kv_cache)
    kernel = select_kernel(
        "attention",
        "mla_prefill_with_kvcache",
        signature,
        features=frozenset({"mla", "paged"}),
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q_nope.shape[0],
        "num_pages": kv_cache.shape[0],
        "page_size": page_size,
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q_nope.shape[1],
        "kv_lora_rank": q_nope.shape[-1],
        "qk_rope_head_dim": q_pe.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_prefill_with_kvcache",
        kernel.name,
        q_nope.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_prefill_with_kvcache",
        q_nope.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q_nope=q_nope,
            q_pe=q_pe,
            cu_seqlens_q=cu_seqlens_q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            return_lse=return_lse,
            qk_nope_head_dim=qk_nope_head_dim,
        )


def mla_decode_with_kvcache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    # attention options
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    qk_nope_head_dim: int | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Single-token MLA decode using a paged compressed MLA KV cache.

    The cache is not stored as separated full K/V tensors. It stores the
    compressed latent KV slice followed by the RoPE key slice in one tensor:
    ``[ckv | k_pe]``.

    Args:
        q_nope: Query no-PE component in the cached attention space with shape
            [batch, num_q_heads, kv_lora_rank].
        q_pe: Query RoPE component with shape
            [batch, num_q_heads, qk_rope_head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        kv_cache: Paged compressed MLA KV cache with shape
            [num_pages, page_size, kv_lora_rank + qk_rope_head_dim] or
            [num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths, shape [batch].
        max_seqlen_k: Maximum KV length.
        softmax_scale: Optional scale factor applied before softmax.
        is_causal: Whether to apply causal masking.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        return_lse: Whether to also return log-sum-exp values.
        qk_nope_head_dim: Optional original no-PE key dimension for kernels that
            need it when ``q_nope`` has already been absorbed to ``kv_lora_rank``.
        override: Optional kernel override name.

    Returns:
        Latent attention output with shape [batch, num_q_heads, kv_lora_rank],
        or ``(output, lse)`` when ``return_lse`` is true.
    """

    page_size = _mla_page_size(kv_cache)
    traits = {
        "num_q_heads": q_nope.shape[1],
        "num_kv_heads": 1,
        "kv_lora_rank": q_nope.shape[-1],
        "qk_rope_head_dim": q_pe.shape[-1],
        "v_head_dim": q_nope.shape[-1],
        "page_size": page_size,
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": _requires_logit_cap(logit_cap),
        "kv_cache_mode": "cached",
        "return_lse": return_lse,
        "uniform_query_len": True,
        "query_len": 1,
    }
    signature = _attention_format_signature(q_nope=q_nope, q_pe=q_pe, kv_cache=kv_cache)
    kernel = select_kernel(
        "attention",
        "mla_decode_with_kvcache",
        signature,
        features=frozenset({"mla", "paged"}),
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q_nope.shape[0],
        "num_pages": kv_cache.shape[0],
        "page_size": page_size,
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q_nope.shape[1],
        "kv_lora_rank": q_nope.shape[-1],
        "qk_rope_head_dim": q_pe.shape[-1],
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_decode_with_kvcache",
        kernel.name,
        q_nope.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_decode_with_kvcache",
        q_nope.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q_nope=q_nope,
            q_pe=q_pe,
            cu_seqlens_q=cu_seqlens_q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            return_lse=return_lse,
            qk_nope_head_dim=qk_nope_head_dim,
        )
