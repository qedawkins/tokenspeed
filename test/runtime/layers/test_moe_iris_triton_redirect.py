from __future__ import annotations

import os
import subprocess
import sys


def test_ep_iris_imports_share_tokenspeed_triton_jit() -> None:
    script = """
from tokenspeed.runtime.layers.moe.backends import ep_dispatch
from tokenspeed_kernel.ops.communication import iris as communication_iris
import tokenspeed_triton.runtime.jit as ts_jit

assert ep_dispatch.gluon is not None
assert isinstance(communication_iris.iris.load, ts_jit.JITCallable), (
    type(communication_iris.iris.load),
    type(communication_iris.iris.load).__module__,
)
communication_iris.iris_allreduce_residual_rmsnorm_kernel.cache_key
print("ok")
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
