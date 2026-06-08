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

import asyncio
import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli import _smg_grpc_engine  # noqa: E402
from tokenspeed.cli._proc import spawn_engine  # noqa: E402


def _as_dict(options):
    return dict(options)


def test_merge_grpc_server_options_overrides_keepalive_defaults(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_GRPC_MIN_RECV_PING_INTERVAL_MS", raising=False)
    monkeypatch.delenv("TOKENSPEED_GRPC_MAX_PINGS_WITHOUT_DATA", raising=False)
    monkeypatch.delenv("TOKENSPEED_GRPC_MAX_PING_STRIKES", raising=False)

    merged = _smg_grpc_engine._merge_grpc_server_options(
        [
            ("grpc.max_send_message_length", 1024),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
    )

    assert _as_dict(merged) == {
        "grpc.max_send_message_length": 1024,
        "grpc.keepalive_permit_without_calls": True,
        "grpc.http2.min_recv_ping_interval_without_data_ms": 1000,
        "grpc.http2.max_pings_without_data": 0,
        "grpc.http2.max_ping_strikes": 0,
    }
    assert [key for key, _ in merged].count(
        "grpc.http2.min_recv_ping_interval_without_data_ms"
    ) == 1


def test_merge_grpc_server_options_accepts_env_overrides(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_GRPC_MIN_RECV_PING_INTERVAL_MS", "250")
    monkeypatch.setenv("TOKENSPEED_GRPC_MAX_PINGS_WITHOUT_DATA", "7")
    monkeypatch.setenv("TOKENSPEED_GRPC_MAX_PING_STRIKES", "9")

    merged = _as_dict(_smg_grpc_engine._merge_grpc_server_options(None))

    assert merged["grpc.http2.min_recv_ping_interval_without_data_ms"] == 250
    assert merged["grpc.http2.max_pings_without_data"] == 7
    assert merged["grpc.http2.max_ping_strikes"] == 9


def test_patch_upstream_grpc_server_options(monkeypatch):
    calls = {}

    def fake_server(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "server"

    fake_upstream_server = types.ModuleType("smg_grpc_servicer.tokenspeed.server")
    fake_upstream_server.grpc = types.SimpleNamespace(
        aio=types.SimpleNamespace(server=fake_server)
    )
    fake_tokenspeed_pkg = types.ModuleType("smg_grpc_servicer.tokenspeed")
    fake_tokenspeed_pkg.server = fake_upstream_server
    fake_root_pkg = types.ModuleType("smg_grpc_servicer")
    fake_root_pkg.tokenspeed = fake_tokenspeed_pkg
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer", fake_root_pkg)
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer.tokenspeed", fake_tokenspeed_pkg)
    monkeypatch.setitem(
        sys.modules,
        "smg_grpc_servicer.tokenspeed.server",
        fake_upstream_server,
    )

    with _smg_grpc_engine._patch_upstream_grpc_server_options():
        result = fake_upstream_server.grpc.aio.server(
            "pool",
            options=[
                ("grpc.max_receive_message_length", 2048),
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ],
        )

    assert result == "server"
    assert _as_dict(calls["kwargs"]["options"]) == {
        "grpc.max_receive_message_length": 2048,
        "grpc.http2.min_recv_ping_interval_without_data_ms": 1000,
        "grpc.http2.max_pings_without_data": 0,
        "grpc.http2.max_ping_strikes": 0,
    }
    assert fake_upstream_server.grpc.aio.server is fake_server


@pytest.mark.asyncio
async def test_spawn_engine_uses_tokenspeed_grpc_wrapper_by_default(monkeypatch):
    monkeypatch.delenv("TS_SERVE_ENGINE_MODULE", raising=False)
    captured = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    proc = await spawn_engine(["--model", "/fake"], host="127.0.0.1", port=12345)

    assert proc is not None
    assert captured["cmd"][:3] == (
        sys.executable,
        "-m",
        "tokenspeed.cli._smg_grpc_engine",
    )
    assert captured["cmd"][3:] == (
        "--host",
        "127.0.0.1",
        "--port",
        "12345",
        "--model",
        "/fake",
    )
    assert captured["kwargs"]["stdout"] == asyncio.subprocess.PIPE
    assert captured["kwargs"]["stderr"] == asyncio.subprocess.PIPE
