"""Tests for comm_ops and comm_backend.

Spawns real distributed workers to test all_reduce, all_gather, reduce_scatter,
token_all_gather, token_reduce_scatter, fused ops, and backend registry.

Usage:
    python -m pytest test/runtime/distributed/test_comm_ops.py -v
"""

import socket
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single


class TestAutoBackendTopology:
    @pytest.fixture
    def backend(self, monkeypatch):
        from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
        from tokenspeed.runtime.utils.env import global_server_args_dict

        monkeypatch.setitem(
            global_server_args_dict,
            "mapping",
            SimpleNamespace(nprocs_per_node=4),
        )
        backend = AutoBackend.__new__(AutoBackend)
        backend._nccl = Mock()
        backend._rsag = Mock()
        backend._triton_ar = Mock()
        backend._trtllm_ar = Mock()
        backend._trtllm_ar.has_trtllm_ar.return_value = False
        return backend

    def test_group_spans_nodes(self, backend):
        assert not backend._group_spans_nodes((0, 1, 2, 3))
        assert not backend._group_spans_nodes((4, 5, 6, 7))
        assert backend._group_spans_nodes((0, 1, 4, 5))

    @pytest.mark.parametrize("method", ["token_all_gather", "token_reduce_scatter"])
    def test_cross_node_token_ops_fall_back_to_nccl(self, backend, method):
        tensor = Mock()
        scattered = [1] * 8
        getattr(backend._nccl, method).return_value = "nccl-result"

        result = getattr(backend, method)(tensor, tuple(range(8)), scattered)

        assert result == "nccl-result"
        getattr(backend._nccl, method).assert_called_once_with(
            tensor, tuple(range(8)), scattered
        )
        getattr(backend._rsag, method).assert_not_called()

    @pytest.mark.parametrize("method", ["token_all_gather", "token_reduce_scatter"])
    def test_node_local_token_ops_use_rsag(self, backend, method):
        tensor = Mock()
        scattered = [1] * 4
        getattr(backend._rsag, method).return_value = "rsag-result"

        result = getattr(backend, method)(tensor, (0, 1, 2, 3), scattered)

        assert result == "rsag-result"
        getattr(backend._rsag, method).assert_called_once_with(
            tensor, (0, 1, 2, 3), scattered
        )
        getattr(backend._nccl, method).assert_not_called()

    def test_cross_node_all_reduce_falls_back_to_nccl(self, backend):
        # trtllm_ar is still consulted for a cross-node group: its mnnvl
        # workspace spans nodes, and it is only armed when that succeeded.
        # NCCL is the fallback for when it is not armed.
        backend._trtllm_ar.has_trtllm_ar.return_value = False
        tensor = Mock()
        backend._nccl.all_reduce.return_value = "nccl-result"

        result = backend.all_reduce(tensor, tuple(range(8)))

        assert result == "nccl-result"
        backend._nccl.all_reduce.assert_called_once_with(
            tensor, tuple(range(8)), op=None
        )
        backend._trtllm_ar.has_trtllm_ar.assert_called_once_with(tuple(range(8)))
        backend._triton_ar.can_run.assert_not_called()

    def test_cross_node_all_reduce_uses_trtllm_when_armed(self, backend):
        """An armed mnnvl workspace serves cross-node groups directly."""
        backend._trtllm_ar.has_trtllm_ar.return_value = True
        backend._trtllm_ar.all_reduce.return_value = "trtllm-result"
        tensor = Mock()

        result = backend.all_reduce(tensor, tuple(range(8)))

        assert result == "trtllm-result"
        backend._nccl.all_reduce.assert_not_called()

    def test_cross_node_last_dim_all_gather_falls_back_to_nccl(self, backend):
        tensor = Mock()
        tensor.dim.return_value = 2
        backend._nccl.all_gather.return_value = "nccl-result"

        result = backend.all_gather(tensor, tuple(range(8)), dim=-1)

        assert result == "nccl-result"
        backend._nccl.all_gather.assert_called_once_with(tensor, tuple(range(8)), -1)
        backend._rsag.all_gather.assert_not_called()


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_fn(rank, world_size, port, test_fn, error_dict):
    try:
        _worker_main(rank, world_size, port, test_fn)
    except Exception:
        import traceback

        error_dict[rank] = traceback.format_exc()


def _worker_main(rank, world_size, port, test_fn):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    group = tuple(range(world_size))
    pg_manager.init_process_group(group)
    ref_group = pg_manager.get_process_group("nccl", group)

    _setup_runtime_globals(rank, world_size)

    test_fn(
        rank=rank,
        world_size=world_size,
        device=device,
        group=group,
        ref_group=ref_group,
    )

    dist.destroy_process_group()


def _setup_runtime_globals(rank, world_size):
    """Match the runtime's setup of global_server_args_dict.

    AutoBackend's 2-D last-dim all_gather and all token-aware ops route through
    TritonRSAGBackend, which sizes its persistent buffers from these globals.
    """
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["chunked_prefill_size"] = 8192
    global_server_args_dict["max_prefill_tokens"] = 8192
    global_server_args_dict["max_model_len"] = 4096
    global_server_args_dict["force_deterministic_rsag"] = True


def _run(world_size, test_fn):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = get_open_port()
    error_dict = mp.Manager().dict()

    mp.spawn(
        worker_fn,
        args=(world_size, port, test_fn, error_dict),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


# ---------------------------------------------------------------------------
# Test functions (run inside each worker)
# ---------------------------------------------------------------------------

TEST_SIZES = [512, 4096, 32768]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _test_all_reduce(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import all_reduce, all_reduce_two

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            expected = inp.clone()
            dist.all_reduce(expected, group=ref_group)
            result = all_reduce(inp.clone(), group)
            torch.testing.assert_close(result, expected)

    # 2D
    for dtype in DTYPES:
        inp = torch.randint(1, 16, (8, 512), dtype=dtype, device=device)
        expected = inp.clone()
        dist.all_reduce(expected, group=ref_group)
        result = all_reduce(inp.clone(), group)
        torch.testing.assert_close(result, expected)

    # Independent production-sized Kimi shared and routed segments. AMD Iris
    # handles this with one kernel; other backends use the two-call fallback.
    first = torch.randint(1, 16, (1, 7168), dtype=torch.bfloat16, device=device)
    second = torch.randint(1, 16, (1, 3584), dtype=torch.bfloat16, device=device)
    expected_first = first.clone()
    expected_second = second.clone()
    dist.all_reduce(expected_first, group=ref_group)
    dist.all_reduce(expected_second, group=ref_group)
    result_first, result_second = all_reduce_two(
        first.clone(),
        second.clone(),
        group,
    )
    torch.testing.assert_close(result_first, expected_first)
    torch.testing.assert_close(result_second, expected_second)


def _test_all_gather(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import all_gather

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            output_list = [torch.empty_like(inp) for _ in range(world_size)]
            dist.all_gather(output_list, inp, group=ref_group)
            expected = torch.cat(output_list, dim=0)
            result = all_gather(inp, group, dim=0)
            torch.testing.assert_close(result, expected)

    # last dim
    for dtype in DTYPES:
        inp = torch.randint(1, 16, (4, 128), dtype=dtype, device=device)
        output_list = [torch.empty_like(inp) for _ in range(world_size)]
        dist.all_gather(output_list, inp, group=ref_group)
        expected = torch.cat(output_list, dim=-1)
        result = all_gather(inp, group, dim=-1)
        torch.testing.assert_close(result, expected)


def _test_all_gather_into_tensor(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            output = torch.empty(sz * world_size, dtype=dtype, device=device)
            expected = torch.empty_like(output)
            dist.all_gather_into_tensor(expected, inp, group=ref_group)
            all_gather_into_tensor(output, inp, group)
            torch.testing.assert_close(output, expected)

    # 2D
    inp = torch.randint(1, 16, (4, 128), dtype=torch.float32, device=device)
    output = torch.empty(4 * world_size, 128, dtype=torch.float32, device=device)
    expected = torch.empty_like(output)
    dist.all_gather_into_tensor(expected, inp, group=ref_group)
    all_gather_into_tensor(output, inp, group)
    torch.testing.assert_close(output, expected)


def _test_all_to_all_single(rank, world_size, device, group, ref_group):
    for sz in TEST_SIZES:
        for dtype in DTYPES:
            total = sz * world_size
            inp = torch.randint(1, 16, (total,), dtype=dtype, device=device)
            expected = torch.empty_like(inp)
            dist.all_to_all_single(expected, inp, group=ref_group)
            output = torch.empty_like(inp)
            all_to_all_single(output, inp, group)
            torch.testing.assert_close(output, expected)

    for dtype in DTYPES:
        rows_per_rank = 4
        total_rows = rows_per_rank * world_size
        inp = torch.randint(1, 16, (total_rows, 128), dtype=dtype, device=device)
        expected = torch.empty_like(inp)
        dist.all_to_all_single(expected, inp, group=ref_group)
        output = torch.empty_like(inp)
        all_to_all_single(output, inp, group)
        torch.testing.assert_close(output, expected)


def _test_reduce_scatter(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import reduce_scatter

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            total_sz = sz * world_size
            inp = torch.randint(1, 16, (total_sz,), dtype=dtype, device=device)
            expected = torch.empty(sz, dtype=dtype, device=device)
            dist.reduce_scatter_tensor(expected, inp, group=ref_group)
            result = reduce_scatter(inp.clone(), group)
            torch.testing.assert_close(result, expected)

    # 2D
    for dtype in DTYPES:
        total_rows = 16 * world_size
        inp = torch.randint(1, 16, (total_rows, 128), dtype=dtype, device=device)
        expected = torch.empty(16, 128, dtype=dtype, device=device)
        dist.reduce_scatter_tensor(expected, inp, group=ref_group)
        result = reduce_scatter(inp.clone(), group)
        torch.testing.assert_close(result, expected)


def _test_token_ops(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import (
        token_all_gather,
        token_reduce_scatter,
    )

    hidden_size = 256

    # Even all_gather
    tokens_per_rank = 64
    scattered = [tokens_per_rank] * world_size
    inp = torch.randn(tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device)
    result = token_all_gather(inp, group, scattered_num_tokens=scattered)
    assert result.shape[0] == tokens_per_rank * world_size

    # Even reduce_scatter
    total_tokens = tokens_per_rank * world_size
    inp = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    result = token_reduce_scatter(inp, group, scattered_num_tokens=scattered)
    assert result.shape[0] == tokens_per_rank

    # Roundtrip: all_gather(reduce_scatter(x) / world_size) == x
    tokens_per_rank = 32
    total_tokens = tokens_per_rank * world_size
    scattered = [tokens_per_rank] * world_size
    torch.manual_seed(42)
    full = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    scattered_out = token_reduce_scatter(full, group, scattered_num_tokens=scattered)
    scattered_out = scattered_out / world_size
    gathered = token_all_gather(scattered_out, group, scattered_num_tokens=scattered)
    torch.testing.assert_close(gathered, full, atol=0.02, rtol=0.02)

    # Uneven distribution
    scattered = [1] * world_size
    scattered[0] = 100
    total_tokens = sum(scattered)
    my_tokens = scattered[rank]
    full = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    scattered_out = token_reduce_scatter(full, group, scattered_num_tokens=scattered)
    assert scattered_out.shape[0] == my_tokens
    gathered = token_all_gather(scattered_out, group, scattered_num_tokens=scattered)
    assert gathered.shape[0] == total_tokens


def _test_fused_ops(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import (
        FusionOp,
        FusionParams,
        fused_all_gather,
        fused_all_reduce,
        fused_reduce_scatter,
    )

    # fused_all_reduce with NONE
    inp = torch.randint(1, 16, (1024,), dtype=torch.float32, device=device)
    expected = inp.clone()
    dist.all_reduce(expected, group=ref_group)
    result = fused_all_reduce(inp.clone(), rank, group)
    torch.testing.assert_close(result, expected)
    result2 = fused_all_reduce(
        inp.clone(), rank, group, fusion_params=FusionParams(fusion_op=FusionOp.NONE)
    )
    torch.testing.assert_close(result2, expected)

    from tokenspeed.runtime.distributed.comm_ops import (
        ResidualRMSNormEpilogue,
        all_reduce_with_epilogue,
    )

    hidden = 128
    inp = torch.full((2, hidden), rank + 1, dtype=torch.bfloat16, device=device)
    residual = torch.linspace(
        -0.5,
        0.5,
        inp.numel(),
        dtype=torch.float32,
        device=device,
    ).reshape_as(inp)
    weight = torch.linspace(
        0.5,
        1.5,
        hidden,
        dtype=torch.float32,
        device=device,
    )
    norm_out, residual_out, scale, partial = all_reduce_with_epilogue(
        inp.clone(),
        group,
        ResidualRMSNormEpilogue(
            residual=residual.to(torch.bfloat16),
            weight=weight,
            eps=1e-6,
            max_token_num=8,
        ),
    )
    reduced = inp.float()
    dist.all_reduce(reduced, group=ref_group)
    expected_residual = reduced + residual.to(torch.bfloat16).float()
    expected_norm = expected_residual * torch.rsqrt(
        expected_residual.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected_norm *= weight
    torch.testing.assert_close(
        residual_out.float(), expected_residual, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(norm_out.float(), expected_norm, atol=2e-2, rtol=2e-2)
    assert scale is None
    assert partial is None

    # fused_reduce_scatter with NONE
    total_sz = 512 * world_size
    inp = torch.randint(1, 16, (total_sz,), dtype=torch.float32, device=device)
    expected = torch.empty(512, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(expected, inp, group=ref_group)
    result = fused_reduce_scatter(inp.clone(), rank, group)
    torch.testing.assert_close(result, expected)

    # fused_all_gather with NONE
    inp = torch.randint(1, 16, (256,), dtype=torch.float32, device=device)
    output_list = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(output_list, inp, group=ref_group)
    expected = torch.cat(output_list, dim=0)
    result = fused_all_gather(inp, rank, group, dim=0)
    torch.testing.assert_close(result, expected)


def _test_backend_registry(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_backend import get_global_backend

    backend = get_global_backend()
    assert backend is not None

    # Singleton
    b2 = get_global_backend()
    assert backend is b2

    # Auto-create resources on first use
    inp = torch.ones(4, device=device)
    result = backend.all_reduce(inp, group)
    assert result.shape == inp.shape


# ---------------------------------------------------------------------------
# FusionParams (no GPU needed)
# ---------------------------------------------------------------------------


class TestFusionParams:
    def test_default_params(self):
        from tokenspeed.runtime.distributed.comm_ops import FusionOp, FusionParams

        params = FusionParams()
        assert params.fusion_op == FusionOp.NONE
        assert params.residual is None
        assert params.norm_weight is None

    def test_residual_rmsnorm_params(self):
        from tokenspeed.runtime.distributed.comm_ops import FusionOp, FusionParams

        weight = torch.ones(128)
        residual = torch.zeros(4, 128)
        params = FusionParams(
            fusion_op=FusionOp.RESIDUAL_RMS_NORM,
            norm_weight=weight,
            residual=residual,
            eps=1e-5,
        )
        assert params.fusion_op == FusionOp.RESIDUAL_RMS_NORM
        assert params.norm_weight is weight

    def test_prepare_all_reduce_lane_uses_backend_capability(self):
        from tokenspeed.runtime.distributed.comm_ops import prepare_all_reduce_lane

        calls = []

        class Backend:
            def prepare_all_reduce_lane(self, group, hidden_dim):
                calls.append((group, hidden_dim))
                return True

        group = (0, 1)
        assert prepare_all_reduce_lane(group, 10752, backend=Backend())
        assert calls == [(group, 10752)]

    def test_prepare_all_reduce_fusion_hides_kernel_backend(self, monkeypatch):
        from tokenspeed.runtime.distributed import comm_ops

        process_group = type("ProcessGroup", (), {"rank": lambda self: 3})()
        calls = []
        monkeypatch.setattr(
            comm_ops,
            "_get_process_group",
            lambda group: process_group,
        )
        monkeypatch.setattr(
            comm_ops,
            "kernel_prepare_allreduce_fusion",
            lambda **kwargs: calls.append(kwargs) or True,
        )

        assert comm_ops.prepare_all_reduce_fusion((0, 1), 10752, 8)
        assert calls == [
            {
                "rank": 3,
                "group": process_group,
                "max_token_num": 8,
                "hidden_dim": 10752,
            }
        ]


class TestAllReduceEpilogues:
    @pytest.fixture(autouse=True)
    def process_group(self, monkeypatch):
        from tokenspeed.runtime.distributed import comm_ops

        process_group = SimpleNamespace(rank=lambda: 0)
        monkeypatch.setattr(
            comm_ops, "_get_process_group", lambda _group: process_group
        )
        return process_group

    def test_residual_rmsnorm_falls_back_as_one_typed_operation(self, monkeypatch):
        from tokenspeed.runtime.distributed import comm_ops

        monkeypatch.setattr(
            comm_ops,
            "allreduce_residual_rmsnorm",
            lambda **_kwargs: (None, None, None, None),
        )
        backend = Mock()
        reduced = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
        backend.all_reduce.return_value = reduced
        residual = torch.tensor([[0.51, -0.49, 0.53, -0.47]], dtype=torch.bfloat16)
        weight = torch.tensor([1.0, 1.5, 0.5, 2.0], dtype=torch.bfloat16)

        normalized, residual_out, scale, partial = comm_ops.all_reduce_with_epilogue(
            torch.zeros_like(reduced),
            (0, 1),
            comm_ops.ResidualRMSNormEpilogue(
                residual=residual,
                weight=weight,
                eps=1e-5,
            ),
            backend=backend,
        )

        expected_fp32 = reduced.float() + residual.float()
        expected_residual = expected_fp32.to(torch.bfloat16)
        expected_normalized = expected_fp32 * torch.rsqrt(
            expected_fp32.square().mean(dim=-1, keepdim=True) + 1e-5
        )
        expected_normalized = (expected_normalized * weight.float()).to(torch.bfloat16)
        torch.testing.assert_close(residual_out, expected_residual)
        torch.testing.assert_close(normalized, expected_normalized)
        assert scale is None
        assert partial is None

    def test_latent_norm_fallback_only_normalizes_routed_prefix(self):
        from tokenspeed.runtime.distributed import comm_ops

        backend = Mock()
        reduced = torch.tensor([[3.0, 4.0, 7.0, 8.0]])
        backend.all_reduce.return_value = reduced
        weight = torch.tensor([2.0, 0.5])

        result = comm_ops.all_reduce_with_epilogue(
            torch.zeros_like(reduced),
            (0, 1),
            comm_ops.LatentRMSNormEpilogue(
                weight=weight,
                latent_width=2,
                eps=1e-6,
                max_token_num=8,
            ),
            backend=backend,
        )

        expected_routed = comm_ops._rmsnorm(reduced[:, :2], weight, 1e-6)
        torch.testing.assert_close(result[:, :2], expected_routed)
        torch.testing.assert_close(result[:, 2:], reduced[:, 2:])

    def test_attnres_uses_specialized_kernel_when_supported(self, monkeypatch):
        from tokenspeed.runtime.distributed import comm_ops

        expected = (Mock(), Mock())
        monkeypatch.setattr(
            comm_ops,
            "allreduce_residual_attnres_combine_supported",
            lambda *_args, **_kwargs: True,
        )
        fused = Mock(return_value=expected)
        monkeypatch.setattr(comm_ops, "allreduce_residual_attnres_combine", fused)
        tensor = torch.empty(1, 4)
        scratch = (torch.empty(1), torch.empty(1), torch.empty(1, 4))
        epilogue = comm_ops.AttnResEpilogue(
            residual=torch.empty_like(tensor),
            res_weight=torch.empty(4),
            rms_weight=torch.empty(4),
            combined_score_weight=torch.empty(4),
            output_weight=torch.empty(4),
            scratch=scratch,
            eps=1e-5,
            max_token_num=8,
            local_world_size=8,
            prepared=True,
        )

        result = comm_ops.all_reduce_with_epilogue(
            tensor,
            tuple(range(8)),
            epilogue,
            backend=Mock(),
        )

        assert result is expected
        fused.assert_called_once()

    def test_attnres_fallback_reduces_before_combine(self, monkeypatch):
        from tokenspeed_kernel.ops.activation import triton as activation_triton

        from tokenspeed.runtime.distributed import comm_ops

        monkeypatch.setattr(
            comm_ops,
            "allreduce_residual_attnres_combine_supported",
            lambda *_args, **_kwargs: False,
        )
        combine = Mock(side_effect=lambda prefix, *_args: prefix + 1)
        monkeypatch.setattr(activation_triton, "attnres_combine", combine)
        backend = Mock()
        reduced = torch.tensor([[2.0, 3.0]])
        backend.all_reduce.return_value = reduced
        residual = torch.tensor([[0.5, 1.5]])
        weight = torch.empty(2)
        scratch = (torch.empty(1), torch.empty(1), torch.empty(1, 2))

        hidden, residual_out = comm_ops.all_reduce_with_epilogue(
            torch.zeros_like(reduced),
            (0, 1),
            comm_ops.AttnResEpilogue(
                residual=residual,
                res_weight=weight,
                rms_weight=weight,
                combined_score_weight=weight,
                output_weight=weight,
                scratch=scratch,
                eps=1e-5,
                max_token_num=8,
                local_world_size=8,
            ),
            backend=backend,
        )

        torch.testing.assert_close(residual_out, residual + reduced)
        torch.testing.assert_close(hidden, residual_out + 1)
        assert combine.call_args.args[0] is residual_out


# ---------------------------------------------------------------------------
# Multi-GPU test classes
# ---------------------------------------------------------------------------

WORLD_SIZES = [
    pytest.param(2, id="ws2"),
    pytest.param(4, id="ws4"),
]


class TestCommOps:

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_reduce(self, world_size):
        _run(world_size, _test_all_reduce)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_gather(self, world_size):
        _run(world_size, _test_all_gather)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_gather_into_tensor(self, world_size):
        _run(world_size, _test_all_gather_into_tensor)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_to_all_single(self, world_size):
        _run(world_size, _test_all_to_all_single)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_reduce_scatter(self, world_size):
        _run(world_size, _test_reduce_scatter)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_token_ops(self, world_size):
        _run(world_size, _test_token_ops)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_fused_ops(self, world_size):
        _run(world_size, _test_fused_ops)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_backend_registry(self, world_size):
        _run(world_size, _test_backend_registry)
