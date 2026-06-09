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

"""TokenSpeed-owned entrypoint for the bundled SMG gRPC engine.

The upstream ``tokenspeed-smg-grpc-servicer`` package owns the gRPC service
implementation.  ``ts serve`` still needs a TokenSpeed-controlled launch point
because long first-token work can keep the backend silent long enough for SMG's
HTTP/2 keepalive pings to trip Python gRPC server enforcement.  This wrapper
keeps the implementation delegated upstream while applying the server options
that TokenSpeed serving requires.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any

try:
    import uvloop
except ImportError:  # uvloop is optional; fall back to the default loop.
    uvloop = None

logger = logging.getLogger(__name__)

_GRPC_MIN_RECV_PING_INTERVAL_ENV = "TOKENSPEED_GRPC_MIN_RECV_PING_INTERVAL_MS"
_GRPC_MAX_PINGS_WITHOUT_DATA_ENV = "TOKENSPEED_GRPC_MAX_PINGS_WITHOUT_DATA"
_GRPC_MAX_PING_STRIKES_ENV = "TOKENSPEED_GRPC_MAX_PING_STRIKES"

_DEFAULT_MIN_RECV_PING_INTERVAL_MS = 1000
_DEFAULT_MAX_PINGS_WITHOUT_DATA = 0
_DEFAULT_MAX_PING_STRIKES = 0

_TOKENSPEED_KEEPALIVE_OPTION_KEYS = {
    "grpc.http2.min_recv_ping_interval_without_data_ms",
    "grpc.http2.max_pings_without_data",
    "grpc.http2.max_ping_strikes",
}


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an int; falling back to %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%d must be non-negative; falling back to %d", name, value, default)
        return default
    return value


def _tokenspeed_keepalive_options() -> tuple[tuple[str, int], ...]:
    return (
        (
            "grpc.http2.min_recv_ping_interval_without_data_ms",
            _env_nonnegative_int(
                _GRPC_MIN_RECV_PING_INTERVAL_ENV,
                _DEFAULT_MIN_RECV_PING_INTERVAL_MS,
            ),
        ),
        (
            "grpc.http2.max_pings_without_data",
            _env_nonnegative_int(
                _GRPC_MAX_PINGS_WITHOUT_DATA_ENV,
                _DEFAULT_MAX_PINGS_WITHOUT_DATA,
            ),
        ),
        (
            "grpc.http2.max_ping_strikes",
            _env_nonnegative_int(
                _GRPC_MAX_PING_STRIKES_ENV,
                _DEFAULT_MAX_PING_STRIKES,
            ),
        ),
    )


def _merge_grpc_server_options(
    options: Iterable[tuple[str, Any]] | None,
) -> list[tuple[str, Any]]:
    """Return gRPC server options with TokenSpeed keepalive policy last."""

    merged = [
        option
        for option in (options or ())
        if option[0] not in _TOKENSPEED_KEEPALIVE_OPTION_KEYS
    ]
    merged.extend(_tokenspeed_keepalive_options())
    return merged


@contextlib.contextmanager
def _patch_upstream_grpc_server_options() -> Iterator[None]:
    """Patch the upstream servicer's ``grpc.aio.server`` call during launch."""

    from smg_grpc_servicer.tokenspeed import server as upstream_server

    original_server = upstream_server.grpc.aio.server

    def server_with_tokenspeed_options(*args: Any, **kwargs: Any) -> Any:
        if "options" in kwargs:
            kwargs["options"] = _merge_grpc_server_options(kwargs["options"])
            return original_server(*args, **kwargs)

        if len(args) >= 4:
            patched_args = list(args)
            patched_args[3] = _merge_grpc_server_options(patched_args[3])
            return original_server(*patched_args, **kwargs)

        kwargs["options"] = _merge_grpc_server_options(None)
        return original_server(*args, **kwargs)

    upstream_server.grpc.aio.server = server_with_tokenspeed_options
    try:
        yield
    finally:
        upstream_server.grpc.aio.server = original_server


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    from tokenspeed.runtime.utils.server_args import prepare_server_args
    from smg_grpc_servicer.tokenspeed import server as upstream_server

    server_args = prepare_server_args(argv)
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    with _patch_upstream_grpc_server_options():
        asyncio.run(upstream_server.serve_grpc(server_args))


if __name__ == "__main__":
    main()
