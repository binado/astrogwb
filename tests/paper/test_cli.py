from __future__ import annotations

import subprocess
import sys

from repo import REPO_ROOT


def test_pure_config_imports_do_not_import_jax() -> None:
    """Pure-config and runtime modules must not pull jax into sys.modules.

    ``astrogwb.paper.config.mcmc`` and ``astrogwb.paper.runtime`` parse flags
    and configure the process without touching jax: importing jax is slow, and
    ``runtime`` deliberately imports it only inside ``configure_runtime``.

    ``config.runs`` is the one this genuinely protects: the ``Snakefile``
    imports it to build the DAG, so every ``--dry-run`` would pay for a jax
    import. It reaches no further than stdlib.

    ``config.catalogs`` is here as a cheap habit rather than a constraint. The
    ``Snakefile`` no longer imports it -- populations are registered model
    names, not graph files it has to resolve into rule inputs -- so its import
    cost stops mattering for ``--dry-run``. It reaches the population registry
    inside the one function that needs it, which keeps this true for free. The
    constraint that *is* load-bearing is the one below.
    """
    code = """
import sys
import astrogwb.paper
import astrogwb.paper.config.mcmc
import astrogwb.paper.config.runs
import astrogwb.paper.config.catalogs
import astrogwb.paper.runtime
assert 'jax' not in sys.modules
assert 'pydantic' not in sys.modules or 'astrogwb.paper.config.mcmc' in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT / "notebooks",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_catalog_inference_snr_imports_leave_the_xla_backend_uninitialized() -> None:
    """Importing jax-touching paper modules must not initialize the XLA backend.

    ``import jax`` / ``import jax.numpy`` only load the package. Backend init
    (``jax.devices()``, array creation) is what freezes ``JAX_PLATFORMS`` /
    ``set_host_device_count``. A late ``set_host_device_count(2)`` still yielding
    two devices proves catalogs / inference / snr did not consume that config.
    This is the real requirement, and it is narrower than "no jax at import":
    ``scripts/run_mcmc.py`` loads and validates both catalogs -- which executes
    their recorded population models -- *before* ``configure_runtime``, so
    those imports must leave the backend free even though they pull in jax.
    """
    code = """
import astrogwb.paper.catalogs
import astrogwb.paper.config.catalogs
import astrogwb.paper.inference
import astrogwb.paper.snr
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {jax.devices()}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT / "notebooks",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
