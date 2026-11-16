"""Process configuration is applied before JAX or its backend loads."""

import subprocess
import sys


def test_runtime_import_does_not_import_jax() -> None:
    code = "import sys; import astrogwb.runtime; assert 'jax' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_runtime_settings_apply_before_initialization() -> None:
    code = """
import os
import sys
from astrogwb.runtime import CPU_THREAD_ENV_VARS, configure_runtime
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "cpu"
configure_runtime(cpu_threads=1)
import jax
assert jax.config.jax_enable_x64
assert os.environ["JAX_PLATFORMS"] == "cpu"
assert all(os.environ[var] == "1" for var in CPU_THREAD_ENV_VARS)
assert "intra_op_parallelism_threads=1" in os.environ["XLA_FLAGS"]
import numpyro
numpyro.set_host_device_count(2)
assert jax.device_count() == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
