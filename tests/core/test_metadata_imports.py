"""Metadata records must be importable without initializing the XLA backend.

This is the core-side twin of ``tests/paper/test_cli.py``. The records live
beside the code they describe (:mod:`astrogwb.populations.metadata`,
:mod:`astrogwb.waveform.metadata`, :mod:`astrogwb.simulators`), so importing one
runs its parent package, which imports JAX and NumPyro. That is fine: importing
JAX does not initialize a backend. What would be fatal is *creating an array or
querying devices* while a record is imported or built, because
``astrogwb.paper.runtime.configure_runtime`` -- reached after the ``Snakefile``
and ``scripts/run_mcmc.py`` have already validated their configs -- must still
be able to choose the platform and device count.

A late ``numpyro.set_host_device_count(2)`` that still yields two devices proves
nothing initialized the backend first.
"""

from __future__ import annotations

import subprocess
import sys

BACKEND_FREE = """
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {jax.devices()}"
"""


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_importing_every_metadata_module_leaves_the_backend_uninitialized() -> None:
    _run(
        """
import astrogwb.distributions.config
import astrogwb.populations.metadata
import astrogwb.waveform.metadata
import astrogwb.simulators.polarization_power.metadata
import astrogwb.simulators.spectra.metadata
"""
        + BACKEND_FREE
    )


def test_building_a_population_still_reaches_the_registry() -> None:
    """The deferred import is a deferral, not a removal.

    ``build()`` has to keep working; it just may not cost anything until it is
    called. Asserting both halves is what stops the import being "fixed" by
    dropping the edge instead of moving it.
    """
    _run(
        """
import sys

from astrogwb.populations.metadata import PopulationMetadata

record = PopulationMetadata(
    model_name='bns_md_cosmological',
    model_kwargs={'minimum_redshift': 0.0, 'maximum_redshift': 1.0, 'n_grid': 8},
)
record.check_registered()
"""
        + BACKEND_FREE
    )
