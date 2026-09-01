"""Fast checks for the realistic mock population and its analytic spectrum."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.constants import ISCO_ALPHA
from astrogwb.gwb import (
    analytic_spectral_density_from_mass_moments,
    spectral_density,
    uniform_prior_mass_moments,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    madau_dickinson_rate,
)
from astrogwb_mock_population import (
    FIDUCIALS,
    MOCK_MAXIMUM_COMPONENT_MASS,
    MOCK_MINIMUM_COMPONENT_MASS,
    Z_MAX,
    Z_MIN,
    make_redshift_grid,
)
from conftest import F_MAX, F_MIN

CATALOG_DF = 8.0
SMALL_CATALOG_SIZE = 256
LARGE_CATALOG_SIZE = 1024


def test_mock_catalog_defaults_cover_the_production_band(mock_catalog_factory) -> None:
    """The realistic default trades frequency resolution for catalog size."""
    catalog = mock_catalog_factory()
    frequencies = np.asarray(catalog.frequency.values)
    redshift = np.asarray(catalog.source_parameters.sel(parameter="redshift").values)

    assert catalog.sizes == {"frequency": 512, "sample": 1024, "parameter": 5}
    assert float(catalog.attrs["df"]) == CATALOG_DF
    assert float(catalog.attrs["minimum_frequency"]) == 2.0
    assert float(catalog.attrs["maximum_frequency"]) == 4096.0
    np.testing.assert_allclose(np.diff(frequencies), CATALOG_DF, rtol=0.0, atol=0.0)
    assert frequencies[0] == 2.0
    assert frequencies[-1] == 4090.0
    assert np.all(frequencies > 0.0)
    assert np.all(redshift >= Z_MIN)


def test_mock_catalog_rejects_more_sources_than_the_population(
    mock_catalog_factory,
) -> None:
    with pytest.raises(
        ValueError,
        match="requested 1025 sources, but the population contains only 1024",
    ):
        mock_catalog_factory(num_sources=1025)


def _source_frame_merger_rate(redshift: jax.Array, hyperparameters) -> jax.Array:
    r"""Return the absolute source-frame merger-rate density in Gpc^-3 yr^-1.

    Deliberately *without* the source-to-detector time dilation.
    :func:`analytic_spectral_density_from_mass_moments` carries that inside its
    :math:`(1 + z)^{4/3}` factor, whereas
    :func:`compute_merger_rate_distance_and_logprob` applies ``/(1 + z)``
    inside its own density, :math:`p(z) \propto \psi(z)/(1+z)\,dV_c/dz`.
    Dividing here as well would double-count it -- exactly the class of error
    the two independent paths are being crossed to detect.
    """
    return hyperparameters["local_merger_rate"] * madau_dickinson_rate(
        redshift,
        hyperparameters["gamma"],
        hyperparameters["kappa"],
        hyperparameters["z_peak"],
    )


def test_catalog_contraction_matches_the_analytic_spectrum(
    mock_catalog_factory,
) -> None:
    r"""Larger Monte-Carlo catalogs converge toward the analytic spectrum."""
    catalog_sizes = (SMALL_CATALOG_SIZE, LARGE_CATALOG_SIZE)
    catalogs = {
        num_sources: mock_catalog_factory(
            num_sources=num_sources,
            f_min=F_MIN,
            f_max=F_MAX,
            df=CATALOG_DF,
        )
        for num_sources in catalog_sizes
    }
    frequencies = jnp.asarray(catalogs[LARGE_CATALOG_SIZE].frequency.values)
    mass_moments = uniform_prior_mass_moments(
        frequencies,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MOCK_MINIMUM_COMPONENT_MASS,
        maximum_component_mass=MOCK_MAXIMUM_COMPONENT_MASS,
        # The same truncation the catalog's polarization power was built with.
        alpha=ISCO_ALPHA,
    )
    analytic = analytic_spectral_density_from_mass_moments(
        frequencies,
        FIDUCIALS,
        _source_frame_merger_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
    )

    ratios: dict[int, np.ndarray] = {}
    for num_sources, catalog in catalogs.items():
        polarization_power = jnp.asarray(catalog.polarization_power.values)
        samples = {
            str(name): jnp.asarray(catalog.source_parameters.sel(parameter=name).values)
            for name in catalog.parameter.values
        }
        total_merger_rate, _, _ = compute_merger_rate_distance_and_logprob(
            FIDUCIALS, samples, redshift_grid=make_redshift_grid()
        )
        contracted = spectral_density(
            polarization_power,
            jnp.ones(num_sources),
            total_merger_rate,
            average_mode="analytic_inclination",
        )
        ratios[num_sources] = np.asarray(contracted / analytic)
        assert np.all(np.isfinite(ratios[num_sources]))

    residuals = {
        num_sources: float(np.sqrt(np.mean((ratio - 1.0) ** 2)))
        for num_sources, ratio in ratios.items()
    }
    assert residuals[LARGE_CATALOG_SIZE] < residuals[SMALL_CATALOG_SIZE]
    np.testing.assert_allclose(
        np.mean(ratios[LARGE_CATALOG_SIZE]), 1.0, rtol=0.05, atol=0.0
    )
    np.testing.assert_allclose(
        ratios[LARGE_CATALOG_SIZE],
        np.mean(ratios[LARGE_CATALOG_SIZE]),
        rtol=0.03,
        atol=0.0,
    )
