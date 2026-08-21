from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb_paper.catalogs import (
    CatalogArrays,
    compute_fiducial_injection_spectrum,
    compute_proposal_logprob,
    validate_matching_frequency_grids,
)
from astrogwb_paper.config.figures import load_fiducials
from astrogwb_paper.config.mcmc import ProposalConfig


def _catalog() -> CatalogArrays:
    return CatalogArrays(
        frequencies=jnp.array([2.0, 3.0]),
        polarization_power=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        samples={
            "redshift": jnp.array([0.1, 1.0, 2.0]),
            "luminosity_distance": jnp.array([450.0, 6800.0, 16_000.0]),
        },
    )


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
    catalog = _catalog()
    proposal = _proposal(0.0)

    actual = compute_proposal_logprob(catalog.samples, proposal)
    _, _, expected = compute_merger_rate_distance_and_logprob(
        {**proposal.model_dump(), "local_merger_rate": 1.0},
        catalog.samples,
        redshift_grid=jnp.linspace(0.0, 20.0, 256),
    )

    np.testing.assert_allclose(actual, expected)


def test_one_fraction_proposal_is_uniform_density() -> None:
    actual = compute_proposal_logprob(_catalog().samples, _proposal(1.0))

    np.testing.assert_allclose(actual, -np.log(20.0))


def test_interior_fraction_uses_stable_mixture_density() -> None:
    catalog = _catalog()
    proposal = _proposal(0.2)
    md = compute_proposal_logprob(catalog.samples, _proposal(0.0))

    actual = compute_proposal_logprob(catalog.samples, proposal)
    expected = np.logaddexp(
        np.log(0.8) + np.asarray(md),
        np.log(0.2 / 20.0),
    )

    np.testing.assert_allclose(actual, expected)
    assert np.all(np.isfinite(actual))


def test_injection_and_proposal_frequency_grids_must_match() -> None:
    proposal = _catalog()
    proposal = CatalogArrays(
        frequencies=jnp.array([2.0, 4.0]),
        polarization_power=proposal.polarization_power,
        samples=proposal.samples,
    )

    with pytest.raises(ValueError, match="identical frequency grids"):
        validate_matching_frequency_grids(_catalog(), proposal)


def test_fiducial_injection_spectrum_uses_unit_weights() -> None:
    injection = _catalog()
    fiducials = load_fiducials()
    grid = jnp.linspace(0.0, 20.0, 256)

    rate, actual = compute_fiducial_injection_spectrum(
        injection,
        fiducials=fiducials,
        redshift_grid=grid,
    )
    expected_rate, _, _ = compute_merger_rate_distance_and_logprob(
        fiducials,
        injection.samples,
        redshift_grid=grid,
    )
    expected = spectral_density(
        injection.polarization_power,
        jnp.ones(3),
        expected_rate,
        average_mode="analytic_inclination",
    )

    np.testing.assert_allclose(rate, expected_rate)
    np.testing.assert_allclose(actual, expected)
