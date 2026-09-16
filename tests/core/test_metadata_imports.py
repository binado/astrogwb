"""The metadata package must stay importable without JAX.

This is the core-side twin of ``tests/paper/test_cli.py``. The models in
:mod:`astrogwb.metadata` are the wire format ``astrogwb.paper.config.catalogs``
validates against at module scope, and the ``Snakefile`` imports that layer to
build its DAG -- so a JAX import reaching them would be paid on every
``--dry-run``.

The trap this guards is not obvious from reading the modules: importing a
submodule executes its parent package first, so a record living under
:mod:`astrogwb.populations` or :mod:`astrogwb.waveform` drags in the population
models or the generators -- and JAX with them -- however carefully the record's
own imports are written. That is why these models sit in a top-level package,
and why the edges back into those layers are taken inside method bodies.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_metadata_does_not_import_jax() -> None:
    code = """
import sys

import astrogwb.metadata
import astrogwb.metadata.population

assert 'jax' not in sys.modules, 'importing astrogwb.metadata imported jax'
assert 'numpyro' not in sys.modules, 'importing astrogwb.metadata imported numpyro'
assert 'h5py' not in sys.modules, 'importing astrogwb.metadata imported h5py'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_building_a_population_still_reaches_the_registry() -> None:
    """The deferred import is a deferral, not a removal.

    ``build()`` has to keep working; it just may not cost anything until it is
    called. Asserting both halves is what stops the import being "fixed" by
    dropping the edge instead of moving it.
    """
    code = """
import sys

from astrogwb.metadata import PopulationMetadata

record = PopulationMetadata(
    model_name='bns_md_cosmological',
    model_kwargs={'minimum_redshift': 0.0, 'maximum_redshift': 1.0, 'n_grid': 8},
    density_sites=('redshift',),
    seed=1,
)
assert 'jax' not in sys.modules, 'constructing the record imported jax'

record.check_registered()
assert 'jax' in sys.modules, 'build() did not reach the registry'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
