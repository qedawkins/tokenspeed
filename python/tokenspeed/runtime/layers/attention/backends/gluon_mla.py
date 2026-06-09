# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AMD Gluon MLA attention backend for TokenSpeed scheduling."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention import (
    mla_decode_absorbed_with_kvcache as mla_decode_with_kvcache,
    mla_prefill_absorbed as mla_prefill,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)


@dataclass
class GluonMLAPrefillMetadata:
    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class GluonMLAChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    max_extend_seq_len: int
    chunked_loop_num: int
    chunk_kv_indices_list: list[torch.Tensor]
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list[int]


@dataclass
class GluonMLADecodeMetadata:
    num_extends: int = 0
    block_kv_indices: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None


class GluonMLABackend(AttentionBackend):
    """MLA backend that dispatches through CDNA4 Gluon kernel wrappers."""

    supports_fused_fp8_mla = False
    _logged_decode = False
    _logged_prefill = False

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype

        if self.page_size not in (32, 64):
            raise ValueError(
                f"tokenspeed_mla AMD backend requires page_size 32 or 64, got {self.page_size}"
            )

        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.forward_decode_metadata: GluonMLADecodeMetadata | None = None
        self.forward_prefill_metadata: GluonMLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, GluonMLADecodeMetadata] = {}
        self.decode_cuda_graph_kv_indices = None
        self.chunked_prefill_metadata: GluonMLAChunkedPrefillMetadata | None = None

    def _calc_num_blocks(self, max_seq_len: int) -> int:
        return (max_seq_len + self.page_size - 1) // self.page_size

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        req_to_page: torch.Tensor,
        block_kv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_kv_indices is None:
            block_kv_indices = torch.zeros(
                (batch_size, max_blocks), dtype=torch.int32, device=self.device
            )

        copy_len = min(max_blocks, req_to_page.shape[1])
        block_kv_indices[:batch_size, :copy_len] = req_to_page[
            req_pool_indices[:batch_size], :copy_len
        ]
        return block_kv_indices

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        spec_info=None,
        **kwargs,
    ):
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens[:num_extends],
                req_pool_indices=req_pool_indices[:num_extends],
                req_to_page=req_to_page,
                extend_prefix_lens=kwargs.pop("extend_prefix_lens"),
                extend_prefix_lens_cpu=kwargs.pop("extend_prefix_lens_cpu"),
                extend_seq_lens=kwargs.pop("extend_seq_lens"),
                extend_seq_lens_cpu=kwargs.pop("extend_seq_lens_cpu"),
            )
        if (
            forward_mode.is_decode()
            or forward_mode.is_mixed()
            or (forward_mode.is_extend() and self.is_draft)
        ):
            self._init_decode_metadata(
                bs,
                num_extends,
                req_pool_indices,
                seq_lens,
                req_to_page,
            )

    @contextmanager
    def override_num_extends(self, num_extends: int):
        assert self.forward_decode_metadata is not None
        prev = self.forward_decode_metadata.num_extends
        self.forward_decode_metadata.num_extends = num_extends
        try:
            yield
        finally:
            self.forward_decode_metadata.num_extends = prev

    def _init_decode_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
    ):
        max_blocks = self._calc_num_blocks(self.max_context_len)
        block_kv_indices = self._create_block_kv_indices(
            bs, max_blocks, req_pool_indices, req_to_page
        )

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_decode_metadata = GluonMLADecodeMetadata(
            num_extends=num_extends,
            block_kv_indices=block_kv_indices,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens,
        )

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        req_to_page: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
    ):
        max_seq_len = self.max_context_len
        cum_seq_lens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_prefill_metadata = GluonMLAPrefillMetadata(
            max_seq_len=max_seq_len,
            cum_seq_lens=cum_seq_lens,
            seq_lens=seq_lens,
        )
        num_extends = extend_seq_lens.shape[0]
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
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
        self.chunked_prefill_metadata = GluonMLAChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            cum_extend_seq_lens=cum_extend_seq_lens,
            max_extend_seq_len=max_extend_seq_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
        )

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        self.cuda_graph_seq_lens_buf = seq_lens_buf
        max_blocks = self._calc_num_blocks(self.max_context_len)
        self.decode_cuda_graph_kv_indices = torch.zeros(
            (max_bs, max_blocks), dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"tokenspeed_mla AMD CUDA graph capture not supported for {forward_mode}"
            )

        max_blocks = self._calc_num_blocks(self.max_context_len)
        metadata = GluonMLADecodeMetadata(
            num_extends=0,
            block_kv_indices=self.decode_cuda_graph_kv_indices[:bs, :max_blocks],
            max_seq_len_k=self.max_context_len,
            seq_lens_k=self.cuda_graph_seq_lens_buf[:bs],
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"tokenspeed_mla AMD CUDA graph replay not supported for {forward_mode}"
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        if req_to_page is not None and not getattr(self, "_block_table_aliased", False):
            self._create_block_kv_indices(
                bs,
                metadata.block_kv_indices.shape[1],
                req_pool_indices[:bs],
                req_to_page,
                metadata.block_kv_indices,
            )
        self.forward_decode_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if save_kv_cache:
            assert k is not None
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        if q.numel() == 0 or bs == 0:
            return torch.empty(
                (0, layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.q_data_type,
                device=q.device,
            )

        metadata = self.forward_decode_metadata
        assert metadata is not None
        num_extends = metadata.num_extends
        q_len_per_req = q.shape[0] // bs
        query = q.view(bs, q_len_per_req, layer.tp_q_head_num, layer.head_dim)
        query = query[num_extends:].reshape(-1, layer.tp_q_head_num, layer.head_dim)

        block_tables = metadata.block_kv_indices[num_extends:].repeat_interleave(
            q_len_per_req, dim=0
        )
        seq_lens = metadata.seq_lens_k[num_extends:].repeat_interleave(q_len_per_req)
        if q_len_per_req > 1:
            offsets = torch.arange(
                q_len_per_req, device=seq_lens.device, dtype=seq_lens.dtype
            ).repeat(metadata.seq_lens_k.shape[0] - num_extends)
            seq_lens = seq_lens + offsets

        softmax_scale = layer.scaling
        if self.data_type == torch.float8_e4m3fn:
            query = query.to(self.data_type)
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            softmax_scale = k_scale * layer.scaling

        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim)

        if not GluonMLABackend._logged_decode:
            logger.info(
                "AMD Gluon MLA decode kernel invoked (query_dtype=%s, kv_dtype=%s)",
                query.dtype,
                kv_cache.dtype,
            )
            GluonMLABackend._logged_decode = True

        raw_out = mla_decode_with_kvcache(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            metadata.max_seq_len_k,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            value_head_dim=layer.v_head_dim,
            softmax_scale=softmax_scale,
        )
        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

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

        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        if not GluonMLABackend._logged_prefill:
            logger.info("AMD Gluon MLA prefill kernel invoked")
            GluonMLABackend._logged_prefill = True

        result = mla_prefill(
            q,
            k,
            v,
            seq_lens,
            cum_seq_lens_kv,
            max_kv_len,
            batch_size,
            scaling,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max_q_len,
            out=out,
        )

        if isinstance(result, tuple):
            out, lse = result
        else:
            out, lse = result, None

        if out.dtype != self.q_data_type:
            out = out.to(self.q_data_type)
        return out, lse


register_backend("tokenspeed_mla", {AttentionArch.MLA}, GluonMLABackend)
