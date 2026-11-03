"""Shared pytest configuration for the paper package.

Enables JAX double precision for the whole suite, as
:func:`astrogwb.runtime.configure_runtime` does, so precision does not
depend on test collection order.
"""

from __future__ import annotations

import jax

# Must precede any array creation, hence module scope rather than a fixture.
jax.config.update("jax_enable_x64", True)
