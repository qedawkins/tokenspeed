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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    mla_decode_with_kvcache,
    mla_prefill,
    mla_prefill_with_kvcache,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class MLAMetadata:
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    max_seq_len_q: int | None = None
    max_seq_len_k: int | None = None


@dataclass
class MLAChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    max_extend_seq_len: int
    chunked_loop_num: int
    chunk_kv_indices_list: list
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list


class MLAAttnBackend(AttentionBackend):
    """Unified MLA backend that routes through tokenspeed_kernel attention APIs."""

    @property
    def support_kv_cache_prewrite(self) -> bool:
        return True

    def __init__(self, config: MLAConfig):
        super().__init__(config)
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.forward_decode_metadata: MLAMetadata | None = None
        self.forward_prefill_metadata: MLAMetadata | None = None
        self.chunked_prefill_metadata: MLAChunkedPrefillMetadata | None = None

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int | None = None,
        req_pool_indices: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        forward_mode: ForwardMode | None = None,
        req_to_page: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        **kwargs,
    ):
        if (
            req_pool_indices is None
            or seq_lens is None
            or forward_mode is None
            or req_to_page is None
        ):
            raise ValueError("mla metadata initialization is missing required inputs")

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        seq_lens = seq_lens[:bs]
        page_table = build_page_table(
            req_pool_indices[:bs],
            req_to_page,
            self.page_size,
            self.max_context_len,
        )

        if forward_mode.is_decode_or_idle():
            tokens_per_req = self._tokens_per_request(bs, num_tokens)
            if tokens_per_req > 1:
                self.forward_prefill_metadata = self._make_multi_token_metadata(
                    bs,
                    tokens_per_req,
                    seq_lens,
                    page_table,
                )

            self.forward_decode_metadata = self._make_decode_metadata(
                bs,
                seq_lens,
                page_table,
            )
            return

        if forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            tokens_per_req = self._tokens_per_request(bs, num_tokens)
            self.forward_prefill_metadata = self._make_multi_token_metadata(
                bs,
                tokens_per_req,
                seq_lens,
                page_table,
            )
            self.forward_decode_metadata = self._make_decode_metadata(
                bs,
                seq_lens.clone(),
                page_table,
            )
            return

        extend_seq_lens = kwargs.get("extend_seq_lens")
        extend_prefix_lens_cpu = kwargs.get("extend_prefix_lens_cpu")
        extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")

        if extend_prefix_lens is None:
            extend_seq_lens = seq_lens
        elif extend_seq_lens is None:
            assert (
                extend_prefix_lens.dtype == torch.int32
            ), f"extend_prefix_lens must be int32, got {extend_prefix_lens.dtype}"
            extend_seq_lens = seq_lens - extend_prefix_lens[:bs]

        if extend_seq_lens_cpu is not None:
            max_seq_len_q = int(extend_seq_lens_cpu.max().item())
        elif num_tokens is not None:
            max_seq_len_q = num_tokens
        else:
            max_seq_len_q = int(extend_seq_lens.max().item())
        self.forward_prefill_metadata = MLAMetadata(
            cache_seqlens_int32=seq_lens,
            cu_seqlens_q=self._make_cu_seqlens(extend_seq_lens),
            page_table=page_table,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=self.max_context_len,
        )

        if extend_prefix_lens is not None and extend_seq_lens_cpu is not None:
            cum_extend_seq_lens = self._make_cu_seqlens(extend_seq_lens)
            (
                chunked_loop_num,
                chunk_kv_indices_list,
                chunked_seq_len,
                cu_chunked_seq_len,
                max_chunk_len_per_loop,
            ) = build_chunked_prefill_metadata_arrays(
                extend_prefix_lens,
                extend_prefix_lens_cpu,
                req_to_page,
                req_pool_indices,
                self.page_size,
            )
            self.chunked_prefill_metadata = MLAChunkedPrefillMetadata(
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                req_pool_indices=req_pool_indices,
                cum_extend_seq_lens=cum_extend_seq_lens,
                max_extend_seq_len=int(extend_seq_lens_cpu.max().item()),
                chunked_loop_num=chunked_loop_num,
                chunk_kv_indices_list=chunk_kv_indices_list,
                chunked_seq_len=chunked_seq_len,
                cu_chunked_seq_len=cu_chunked_seq_len,
                max_chunk_len_per_loop=max_chunk_len_per_loop,
            )

    def _make_decode_metadata(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
    ) -> MLAMetadata:
        return MLAMetadata(
            cache_seqlens_int32=seq_lens,
            cu_seqlens_q=self._make_uniform_cu_seqlens(bs, 1, seq_lens.device),
            page_table=page_table,
            max_seq_len_q=1,
            max_seq_len_k=self.max_context_len,
        )

    def _make_multi_token_metadata(
        self,
        bs: int,
        tokens_per_req: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
    ) -> MLAMetadata:
        return MLAMetadata(
            cache_seqlens_int32=seq_lens,
            cu_seqlens_q=self._make_uniform_cu_seqlens(
                bs,
                tokens_per_req,
                seq_lens.device,
            ),
            page_table=page_table,
            max_seq_len_q=tokens_per_req,
            max_seq_len_k=self.max_context_len,
        )

    def _tokens_per_request(self, bs: int, num_tokens: int | None) -> int:
        if bs <= 0:
            return 1
        if num_tokens is not None:
            return max(int(num_tokens) // bs, 1)
        return max(int(self.spec_num_tokens or 1), 1)

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        self.cuda_graph_prefill_metadata = {}
        self.cuda_graph_decode_metadata = {}
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cache_seqlens = seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        if forward_mode.is_decode_or_idle():
            spec_num_tokens = self.spec_num_tokens or 1
            if spec_num_tokens > 1:
                self._init_target_verify_metadata_capture(bs, spec_num_tokens)
            self._init_decode_metadata_capture(bs)
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            spec_num_tokens = self.spec_num_tokens or 1
            self._init_target_verify_metadata_capture(bs, spec_num_tokens)
            if forward_mode.is_draft_extend():
                self._init_decode_metadata_capture(bs)
        else:
            raise NotImplementedError(
                f"mla CUDA graph capture not supported for {forward_mode}"
            )

    def _init_decode_metadata_capture(self, bs: int):
        metadata = MLAMetadata(
            cache_seqlens_int32=self.cuda_graph_cache_seqlens[:bs],
            cu_seqlens_q=self._make_uniform_cu_seqlens(bs, 1, self.device),
            page_table=self.cuda_graph_page_table[:bs, :],
            max_seq_len_q=1,
            max_seq_len_k=self.max_context_len,
        )
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _init_target_verify_metadata_capture(self, bs: int, spec_num_tokens: int):
        metadata = MLAMetadata(
            cache_seqlens_int32=self.cuda_graph_cache_seqlens[:bs],
            cu_seqlens_q=self._make_uniform_cu_seqlens(
                bs,
                spec_num_tokens,
                self.device,
            ),
            page_table=self.cuda_graph_page_table[:bs, :],
            max_seq_len_q=spec_num_tokens,
            max_seq_len_k=self.max_context_len,
        )
        self.cuda_graph_prefill_metadata[bs] = metadata
        self.forward_prefill_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        if req_to_page is not None:
            self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
                req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            )

        if forward_mode.is_decode_or_idle():
            spec_num_tokens = self.spec_num_tokens or 1
            if spec_num_tokens > 1:
                self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]
        elif forward_mode.is_target_verify():
            self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
        elif forward_mode.is_draft_extend():
            self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]
        else:
            raise NotImplementedError(
                f"mla CUDA graph replay not supported for {forward_mode}"
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                forward_mode=ForwardMode.DECODE,
                **kwargs,
            )

        if save_kv_cache and k is not None:
            self._set_kv_cache(layer, out_cache_loc, token_to_kv_pool, k)

        metadata = self.forward_decode_metadata
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        q_nope = q[..., : self.kv_lora_rank]
        q_pe = q[..., self.kv_lora_rank :]
        kv_cache = self._get_kv_cache(layer, token_to_kv_pool)

        result = mla_decode_with_kvcache(
            q_nope=q_nope,
            q_pe=q_pe,
            cu_seqlens_q=metadata.cu_seqlens_q,
            kv_cache=kv_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            max_seqlen_k=metadata.max_seq_len_k,
            softmax_scale=layer.scaling,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            return_lse=False,
            qk_nope_head_dim=self.qk_nope_head_dim,
        )
        return self._unwrap_output(result).reshape(
            -1, layer.tp_q_head_num * layer.v_head_dim
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        forward_mode: ForwardMode | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del bs, forward_mode

        metadata = self.forward_prefill_metadata
        cu_seqlens_q = metadata.cu_seqlens_q
        assert cu_seqlens_q is not None
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        q_nope = q[..., : self.kv_lora_rank]
        q_pe = q[..., self.kv_lora_rank :]
        k = None if k is None else k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = None if v is None else v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        if save_kv_cache and k is not None:
            self._set_kv_cache(layer, out_cache_loc, token_to_kv_pool, k)

        no_kv_cache = False
        if k is not None and v is not None and not save_kv_cache:
            query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            no_kv_cache = bool(torch.equal(metadata.cache_seqlens_int32, query_lens))

        if no_kv_cache:
            result = mla_prefill(
                q_nope=q_nope,
                q_pe=q_pe,
                k_nope=k[..., : self.kv_lora_rank],
                k_pe=k[..., self.kv_lora_rank :],
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                max_seqlen_q=metadata.max_seq_len_q,
                max_seqlen_k=metadata.max_seq_len_q,
                softmax_scale=layer.scaling,
                is_causal=True,
                window_left=layer.sliding_window_size,
                logit_cap=layer.logit_cap,
                return_lse=False,
            )
            return self._unwrap_output(result).reshape(
                -1, layer.tp_q_head_num * layer.v_head_dim
            )
        else:
            kv_cache = self._get_kv_cache(layer, token_to_kv_pool)
            result = mla_prefill_with_kvcache(
                q_nope=q_nope,
                q_pe=q_pe,
                cu_seqlens_q=cu_seqlens_q,
                kv_cache=kv_cache,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                max_seqlen_q=metadata.max_seq_len_q,
                max_seqlen_k=metadata.max_seq_len_k,
                softmax_scale=layer.scaling,
                is_causal=True,
                window_left=layer.sliding_window_size,
                logit_cap=layer.logit_cap,
                return_lse=False,
                qk_nope_head_dim=self.qk_nope_head_dim,
            )
            return self._unwrap_output(result).reshape(
                -1, layer.tp_q_head_num * layer.v_head_dim
            )

    def forward_extend_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scaling,
        logits_soft_cap,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
        out: torch.Tensor | None = None,
    ):
        if causal:
            step_counter = getattr(self, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)

        result = mla_prefill(
            q_nope=q[..., : self.qk_nope_head_dim],
            q_pe=q[..., self.qk_nope_head_dim :],
            k_nope=k[..., : self.qk_nope_head_dim],
            k_pe=k[..., self.qk_nope_head_dim :],
            v=v,
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_kv=cum_seq_lens_kv,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scaling,
            is_causal=causal,
            logit_cap=logits_soft_cap,
            return_lse=True,
        )
        if isinstance(result, tuple):
            output, lse = result[0], result[1]
        else:
            output, lse = result, None
        if out is not None:
            out.copy_(output.view(out.shape))
            output = out
        return output, lse

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if isinstance(k_cache, tuple):
            raise NotImplementedError(
                "unified mla backend does not support per-token FP8 cache"
            )
        return k_cache.view(-1, self.page_size, self.kv_cache_dim)

    def _set_kv_cache(self, layer, out_cache_loc, token_to_kv_pool, k):
        token_to_kv_pool.set_mla_kv_buffer(
            layer,
            out_cache_loc,
            k[..., : self.kv_lora_rank],
            k[..., self.kv_lora_rank :],
        )

    @staticmethod
    def _make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
            (1, 0),
        )

    @staticmethod
    def _make_uniform_cu_seqlens(
        batch_size: int,
        tokens_per_req: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(
            0,
            batch_size * tokens_per_req + 1,
            tokens_per_req,
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _unwrap_output(result):
        if isinstance(result, tuple):
            return result[0]
        return result


register_backend("mla", {AttentionArch.MLA}, MLAAttnBackend)
