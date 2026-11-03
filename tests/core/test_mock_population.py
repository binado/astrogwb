"""Fast checks for the realistic mock population and its analytic spectrum."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from astrogwb_mock_population import (
    F_MAX,
    F_MIN,
    FIDUCIALS,
    MOCK_MAXIMUM_COMPONENT_MASS,
    MOCK_MINIMUM_COMPONENT_MASS,
    POPULATION_PARAMS,
    Z_MAX,
    Z_MIN,
    catalog_samples,
    make_redshift_grid,
)
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.background import (
    analytic_spectral_density_from_mass_moments,
    spectral_density,
    uniform_prior_mass_moments,
)
from astrogwb.constants import ISCO_ALPHA
from astrogwb.distributions.rates import madau_dickinson_rate

CATALOG_DF = 8.0
SMALL_CATALOG_SIZE = 256
LARGE_CATALOG_SIZE = 1024


def _source_frame_merger_rate(redshift: jax.Array, hyperparameters) -> jax.Array:
    r"""Return the absolute source-frame merger-rate density in Gpc^-3 yr^-1.

    Without the time dilation:
    :func:`analytic_spectral_density_from_mass_moments` carries it in its
    :math:`(1 + z)^{4/3}` factor.
    """
    return madau_dickinson_rate(
        redshift,
        hyperparameters["gamma"],
        hyperparameters["kappa"],
        hyperparameters["z_peak"],
        hyperparameters["local_merger_rate"],
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
            frequency_resolution=CATALOG_DF,
        )
        for num_sources in catalog_sizes
    }
    frequencies = jnp.asarray(catalogs[LARGE_CATALOG_SIZE][0]["frequencies"])
    mass_moments = uniform_prior_mass_moments(
        frequencies,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MOCK_MINIMUM_COMPONENT_MASS,
        maximum_component_mass=MOCK_MAXIMUM_COMPONENT_MASS,
        # The same truncation the catalog's intensity was built with.
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
    for num_sources, (data, _) in catalogs.items():
        intensity = jnp.asarray(data["intensity"])
        samples = catalog_samples(data)
        total_merger_rate, _, _ = reference_merger_rate_distance_and_logprob(
            POPULATION_PARAMS,
            samples["redshift"],
            redshift_grid=make_redshift_grid(),
        )
        contracted = spectral_density(
            intensity,
            jnp.ones(num_sources),
            total_merger_rate,
        )
        analytic_values = np.asarray(analytic)
        contracted_values = np.asarray(contracted)
        # Above the smallest sampled cutoff, a finite catalog can be exactly
        # zero even while the analytic population spectrum remains positive.
        # Compare only on the common support where every sampled source emits.
        valid = np.all(np.asarray(intensity) > 0.0, axis=1)
        ratios[num_sources] = contracted_values[valid] / analytic_values[valid]
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
