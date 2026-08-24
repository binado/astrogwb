from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb_paper.catalogs import (
    compute_fiducial_injection_spectrum,
    compute_proposal_logprob,
    validate_matching_frequency_grids,
)
from astrogwb_paper.config.figures import load_fiducials
from astrogwb_paper.config.mcmc import ProposalConfig

_REDSHIFT = jnp.array([0.1, 1.0, 2.0])
_LUMINOSITY_DISTANCE = jnp.array([450.0, 6800.0, 16_000.0])
_POLARIZATION_POWER = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
_SAMPLES = {"redshift": _REDSHIFT, "luminosity_distance": _LUMINOSITY_DISTANCE}


def _proposal(epsilon: float) -> ProposalConfig:
    return ProposalConfig(
        uniform_mixing_fraction=epsilon,
        minimum_redshift=0.0,
        maximum_redshift=20.0,
        n_grid=256,
        H0=67.66,
        Omega_m=0.3096,
        gamma=1.42,
        kappa=4.62,
        z_peak=1.84,
    )


def test_zero_fraction_proposal_is_the_md_density() -> None:
    proposal = _proposal(0.0)

    actual = compute_proposal_logprob(_REDSHIFT, proposal)
    _, _, expected = compute_merger_rate_distance_and_logprob(
        {**proposal.model_dump(), "local_merger_rate": 1.0},
        {"redshift": _REDSHIFT},
        redshift_grid=jnp.linspace(0.0, 20.0, 256),
    )

    np.testing.assert_allclose(actual, expected)


def test_one_fraction_proposal_is_uniform_density() -> None:
    actual = compute_proposal_logprob(_REDSHIFT, _proposal(1.0))

    np.testing.assert_allclose(actual, -np.log(20.0))


def test_interior_fraction_uses_stable_mixture_density() -> None:
    proposal = _proposal(0.2)
    md = compute_proposal_logprob(_REDSHIFT, _proposal(0.0))

    actual = compute_proposal_logprob(_REDSHIFT, proposal)
    expected = np.logaddexp(
        np.log(0.8) + np.asarray(md),
        np.log(0.2 / 20.0),
    )

    np.testing.assert_allclose(actual, expected)
    assert np.all(np.isfinite(actual))


def test_injection_and_proposal_frequency_grids_must_match() -> None:
    with pytest.raises(ValueError, match="identical frequency grids"):
        validate_matching_frequency_grids(jnp.array([2.0, 3.0]), jnp.array([2.0, 4.0]))


def test_fiducial_injection_spectrum_uses_unit_weights() -> None:
    fiducials = load_fiducials()
    grid = jnp.linspace(0.0, 20.0, 256)

    rate, actual = compute_fiducial_injection_spectrum(
        _POLARIZATION_POWER,
        _SAMPLES,
        fiducials=fiducials,
        redshift_grid=grid,
    )
    expected_rate, _, _ = compute_merger_rate_distance_and_logprob(
        fiducials,
        _SAMPLES,
        redshift_grid=grid,
    )
    expected = spectral_density(
        _POLARIZATION_POWER,
        jnp.ones(3),
        expected_rate,
        average_mode="analytic_inclination",
    )

    np.testing.assert_allclose(rate, expected_rate)
    np.testing.assert_allclose(actual, expected)
