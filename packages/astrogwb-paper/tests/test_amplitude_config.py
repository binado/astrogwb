"""Tests for the paper's amplitude-marginalization glue.

``build_amplitude_quadrature`` and ``draw_amplitude_posterior`` are the only
JAX-touching bridge between the validated config and astrogwb's numerical
marginalization; the physics itself (the H0^3/H0^2 split) is covered in core's
``test_amplitude_scalings.py``, not here.
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import xarray as xr
from astrogwb.sampling.amplitude import make_amplitude_quadrature
from astrogwb_paper.amplitude import (
    amplitude_grid,
    build_amplitude_quadrature,
    draw_amplitude_posterior,
)
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


def _marginalized_config():
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = []
    return build_run_config(raw)


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
# draw_amplitude_posterior
# --------------------------------------------------------------------------- #


def _synthetic_posterior(n_chain: int, n_draw: int, seed: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "amplitude_mle": (
                ("chain", "draw"),
                rng.uniform(0.8, 1.2, size=(n_chain, n_draw)),
            ),
            "template_optimal_snr": (
                ("chain", "draw"),
                rng.uniform(50.0, 150.0, size=(n_chain, n_draw)),
            ),
        }
    )


def _toy_quadrature():
    grid = jnp.linspace(0.5, 1.5, 2001)
    log_prior = jnp.zeros_like(grid)
    return make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=lambda marginalized_parameter: marginalized_parameter,
        mean_energy_flux_amplitude=lambda marginalized_parameter: jnp.ones_like(
            marginalized_parameter
        ),
    )


def test_draw_amplitude_posterior_is_chunk_size_invariant() -> None:
    """Chunking is purely a memory knob: results must not depend on chunk_size."""
    posterior = _synthetic_posterior(4, 37)
    quadrature = _toy_quadrature()
    key = jax.random.key(0)

    phi_unchunked, nodes_unchunked = draw_amplitude_posterior(
        posterior, quadrature=quadrature, rng_key=key, chunk_size=10_000
    )
    phi_chunked, nodes_chunked = draw_amplitude_posterior(
        posterior, quadrature=quadrature, rng_key=key, chunk_size=7
    )

    assert phi_unchunked.shape == (4, 37)
    np.testing.assert_array_equal(np.asarray(phi_unchunked), np.asarray(phi_chunked))
    # quadrature_effective_nodes reduces over K=2001 grid points; XLA may pick a
    # different summation order for a different vmap batch size, so allow for
    # floating-point reduction-order noise rather than requiring bit-equality.
    np.testing.assert_allclose(
        np.asarray(nodes_unchunked), np.asarray(nodes_chunked), rtol=1e-10
    )


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
