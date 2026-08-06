"""Tests for the paper's amplitude-marginalization glue.

``build_amplitude_quadrature`` is the only JAX-touching bridge between the
validated config and astrogwb's numerical marginalization; the physics itself
(the H0^3/H0^2 split) is covered in core's ``test_amplitude_scalings.py``, not
here.
"""

from __future__ import annotations

import typing

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_paper.amplitude import amplitude_grid, build_amplitude_quadrature
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import AmplitudeParameter, build_run_config
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


# --------------------------------------------------------------------------- #
# amplitude_grid
# --------------------------------------------------------------------------- #


def test_amplitude_grid_uniform_prior_spans_low_to_high() -> None:
    prior = dist.Uniform(20.0, 140.0)
    grid = amplitude_grid(prior, num_nodes=101, span_sigma=10.0)
    np.testing.assert_allclose(np.asarray(grid), np.linspace(20.0, 140.0, 101))


def test_amplitude_grid_normal_prior_spans_span_sigma() -> None:
    prior = dist.Normal(70.0, 5.0)
    grid = amplitude_grid(prior, num_nodes=101, span_sigma=10.0)
    np.testing.assert_allclose(np.asarray(grid), np.linspace(20.0, 120.0, 101))


def test_amplitude_grid_rejects_unsupported_prior_type() -> None:
    prior = dist.Exponential(1.0)
    with pytest.raises(TypeError, match="unsupported amplitude prior type"):
        amplitude_grid(prior, num_nodes=101, span_sigma=10.0)


# --------------------------------------------------------------------------- #
# build_amplitude_quadrature
# --------------------------------------------------------------------------- #


def _marginalized_config(**kwargs):
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = []
    return build_run_config(raw, **kwargs)


def test_build_amplitude_quadrature_matches_config_grid_settings() -> None:
    config = _marginalized_config()
    quadrature = build_amplitude_quadrature(config)

    assert quadrature.grid.shape == (config.analysis.amplitude_num_nodes,)
    np.testing.assert_allclose(float(quadrature.grid[0]), 20.0)
    np.testing.assert_allclose(float(quadrature.grid[-1]), 140.0)

    # A uniform prior's log density is already normalized on its own support.
    integral = float(jnp.trapezoid(jnp.exp(quadrature.log_prior), quadrature.grid))
    np.testing.assert_allclose(integral, 1.0, rtol=1e-6)


def test_build_amplitude_quadrature_rejects_a_non_marginalized_config() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="amplitude-marginalized config"):
        build_amplitude_quadrature(config)


# --------------------------------------------------------------------------- #
# save() for an amplitude-marginalized run
# --------------------------------------------------------------------------- #


def _toy_marginalized_mcmc():
    """A minimal MCMC publishing the deterministics ``save()`` consumes.

    The reconstruction only needs the amplitude sufficient statistics, so this
    skips the catalog entirely: it exercises the save path, not the physics.
    """
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    def model() -> None:
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.deterministic("template_merger_rate", 1e-8 * jnp.exp(0.01 * x))
        numpyro.deterministic("amplitude_mle", 1.0 + 0.01 * x)
        numpyro.deterministic("template_optimal_snr", 30.0 + 0.0 * x)
        numpyro.deterministic("importance_relative_ess", 0.9 + 0.0 * x)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=20,
        num_samples=10,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0))
    return mcmc


@pytest.mark.integration
def test_save_writes_reconstructed_amplitude_and_quadrature(tmp_path) -> None:
    """``save()`` must produce a readable NetCDF carrying phi and its grid.

    Regression guard: ``az.from_numpyro`` returns an xarray ``DataTree``, whose
    ``__setitem__`` silently accepts a Dataset-style ``(dims, values)`` tuple as
    an object scalar and then fails at ``to_netcdf``.
    """
    import xarray as xr
    from astrogwb_paper.cli.run_mcmc import save

    config = _marginalized_config(outdir=tmp_path)
    quadrature = build_amplitude_quadrature(config)

    nc_path = save(
        _toy_marginalized_mcmc(),
        config,
        catalog_path=tmp_path / "catalog.h5",
        quadrature=quadrature,
    )

    tree = xr.open_datatree(nc_path)
    posterior = tree["posterior"].dataset
    for name in ("H0", "total_merger_rate", "quadrature_effective_nodes"):
        assert posterior[name].dims == ("chain", "draw"), name
        assert posterior[name].shape == (2, 10), name
        assert np.all(np.isfinite(posterior[name].values)), name

    # H0 must land inside the prior grid it was drawn against.
    h0 = posterior["H0"].values
    assert np.all(h0 >= float(quadrature.grid[0]))
    assert np.all(h0 <= float(quadrature.grid[-1]))


# --------------------------------------------------------------------------- #
# Cross-check against core's AMPLITUDE_PARAMETERS
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_amplitude_parameter_literal_matches_core_amplitude_parameters() -> None:
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        AMPLITUDE_PARAMETERS,
    )

    literal_values = typing.get_args(AmplitudeParameter)
    assert set(literal_values) == set(AMPLITUDE_PARAMETERS)
