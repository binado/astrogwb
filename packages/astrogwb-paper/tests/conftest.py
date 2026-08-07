"""Shared pytest configuration for the paper package.

Enables JAX double precision for the whole suite. Production runs always go
through :func:`astrogwb_paper.runtime.configure_runtime`, which turns x64 on
before any array is created, so tests that assert on numerical results are only
meaningful at the same precision.

Without this, x64 was enabled as a side effect of whichever test happened to
call ``configure_runtime`` first, making precision depend on collection order:
``pytest packages/astrogwb-paper/tests`` passed while ``pytest
packages/astrogwb-paper/tests/test_amplitude_config.py`` failed on float32
rounding (``72.99999237`` for a grid node of ``73.0``).
"""

from __future__ import annotations

import jax

# Must precede any array creation, hence module scope rather than a fixture.
jax.config.update("jax_enable_x64", True)
