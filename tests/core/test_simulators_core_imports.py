"""``astrogwb.simulators.core`` knows nothing about the physics it caches."""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = ("populations", "waveform", "gwb", "inference")


def test_core_imports_no_physics_package() -> None:
    code = f"""
import sys

import astrogwb.simulators.core

loaded = [
    name
    for name in sys.modules
    for pkg in {FORBIDDEN!r}
    if name == f"astrogwb.{{pkg}}" or name.startswith(f"astrogwb.{{pkg}}.")
]
assert not loaded, f"astrogwb.simulators.core imported {{sorted(loaded)}}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
