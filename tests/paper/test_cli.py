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


def test_config_accessors_are_lazy_and_leave_the_backend_free() -> None:
    """The accessors must cost nothing until called, then cost only what they must.

    Two properties, and both regress silently:

    1. `astrogwb.paper.config` is executed by the `Snakefile`'s
       `from astrogwb.paper.config.runs import ...`, so importing it -- or the
       accessor *names* -- must not read a file, import pydantic, or import
       numpyro. Module-level dicts would break all three, and the file read
       would fail outright from any cwd but the repository root.
    2. `priors()` does import numpyro, but must leave the XLA backend
       uninitialized: `scripts/run_mcmc.py` validates its config before
       `configure_runtime` runs, and a late `set_host_device_count` that still
       yields two devices proves nothing consumed that configuration early.
       This is what breaks if anyone adds a `.sample()` smoke check or a
       `jnp.asarray` to the loader. :func:`waveform_generator` is not on that
       path -- constructing a Ripple kernel does initialize the backend -- so
       this test only pins that importing its *name* does not import JAX.

    Unlike the test above, this one runs from the repository root: the
    accessors resolve `config/*.json` against the caller's cwd by design.
    """
    code = """
import sys
import astrogwb.paper.config
assert 'numpyro' not in sys.modules, 'importing the package imported numpyro'
assert 'pydantic' not in sys.modules, 'importing the package imported pydantic'
assert 'jax' not in sys.modules, 'importing the package imported jax'

from astrogwb.paper.config import fiducials, networks, priors, waveform_generator
assert 'numpyro' not in sys.modules, 'importing the accessor names imported numpyro'
assert 'jax' not in sys.modules, 'importing waveform_generator imported jax'
assert callable(waveform_generator)

assert len(fiducials()) == 10
assert networks()['ET-2L-aligned-CE-Hanford'] == ('S1', 'R1', 'C1')
assert 'numpyro' not in sys.modules, 'fiducials()/networks() are not stdlib-only'

assert len(priors()) == 10
assert 'numpyro' in sys.modules

import numpyro
numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"priors() initialized the backend: {jax.devices()}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_the_accessors_hand_back_copies() -> None:
    """`@cache` returns the same object every call; the accessors must not.

    A notebook that does `p = priors(); p['H0'] = ...` would otherwise poison
    every later reader in the process.
    """
    from astrogwb.paper.config import fiducials, networks, priors

    fiducials(REPO_ROOT)["H0"] = 999.0
    networks(REPO_ROOT)["ET-2L-aligned"] = ("nope",)
    del priors(REPO_ROOT)["H0"]

    assert fiducials(REPO_ROOT)["H0"] == 67.66
    assert networks(REPO_ROOT)["ET-2L-aligned"] == ("S1", "R1")
    assert "H0" in priors(REPO_ROOT)
