from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb_paper.catalogs import (
    PROPOSAL_REDSHIFT_LOGPDF,
    CatalogArrays,
    compute_fiducial_injection_spectrum,
    validate_catalog_samples,
    validate_matching_frequency_grids,
)
from astrogwb_paper.config.figures import load_fiducials


def _catalog(*, with_logpdf: bool = True) -> CatalogArrays:
    samples = {
        "redshift": jnp.array([0.1, 1.0, 2.0]),
        "luminosity_distance": jnp.array([450.0, 6800.0, 16_000.0]),
    }
    if with_logpdf:
        samples[PROPOSAL_REDSHIFT_LOGPDF] = jnp.array([-2.0, -1.0, -3.0])
    return CatalogArrays(
        frequencies=jnp.array([2.0, 3.0]),
        polarization_power=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        samples=samples,
    )


def test_proposal_catalog_requires_stored_density() -> None:
    with pytest.raises(ValueError, match=PROPOSAL_REDSHIFT_LOGPDF):
        validate_catalog_samples(
            _catalog(with_logpdf=False),
            label="proposal",
            z_min=0.0,
            z_max=20.0,
            require_proposal_density=True,
        )


def test_catalog_validation_rejects_nonfinite_density() -> None:
    catalog = _catalog()
    catalog.samples[PROPOSAL_REDSHIFT_LOGPDF] = jnp.array([-2.0, -jnp.inf, -3.0])

    with pytest.raises(ValueError, match="must be finite"):
        validate_catalog_samples(
            catalog,
            label="proposal",
            z_min=0.0,
            z_max=20.0,
            require_proposal_density=True,
        )


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
    injection = _catalog(with_logpdf=False)
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
