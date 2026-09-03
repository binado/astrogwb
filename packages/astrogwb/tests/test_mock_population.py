"""Fast checks for the realistic mock population and its analytic spectrum."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from astrogwb.constants import ISCO_ALPHA
from astrogwb.gwb import (
    analytic_spectral_density_from_mass_moments,
    omega_gw_from_spectral_density,
    spectral_density,
    spectral_density_from_omega_gw,
    uniform_prior_mass_moments,
)
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
    madau_dickinson_rate,
)
from astrogwb_mock_population import (
    F_MAX,
    F_MIN,
    FIDUCIALS,
    FIXTURES_DIR,
    MOCK_MAXIMUM_COMPONENT_MASS,
    MOCK_MINIMUM_COMPONENT_MASS,
    MOCK_POPULATION_PATH,
    MOCK_POPULATION_SEED,
    Z_MAX,
    Z_MIN,
    catalog_samples,
    make_redshift_grid,
)

CATALOG_DF = 8.0
SMALL_CATALOG_SIZE = 256
LARGE_CATALOG_SIZE = 1024


def test_population_graph_matches_shared_mock_constants() -> None:
    config = yaml.safe_load(
        (FIXTURES_DIR / "mock_bns_population.yaml").read_text(encoding="utf-8")
    )
    parameters = config["parameters"]

    redshift_arguments = parameters["redshift"]["sampler"]["arguments"]
    assert redshift_arguments["hubble_constant"] == FIDUCIALS["H0"]
    assert redshift_arguments["omega_m"] == FIDUCIALS["Omega_m"]
    assert redshift_arguments["gamma"] == FIDUCIALS["gamma"]
    assert redshift_arguments["kappa"] == FIDUCIALS["kappa"]
    assert redshift_arguments["z_peak"] == FIDUCIALS["z_peak"]
    assert redshift_arguments["z_min"] == Z_MIN
    assert redshift_arguments["z_max"] == Z_MAX

    mass_arguments = parameters["mass_pair"]["sampler"]["arguments"]
    assert mass_arguments["m1_min"] == MOCK_MINIMUM_COMPONENT_MASS
    assert mass_arguments["m1_max"] == MOCK_MAXIMUM_COMPONENT_MASS
    assert mass_arguments["m2_min"] == MOCK_MINIMUM_COMPONENT_MASS
    assert mass_arguments["m2_max"] == MOCK_MAXIMUM_COMPONENT_MASS


def test_committed_population_provenance_matches_shared_constants() -> None:
    header = MOCK_POPULATION_PATH.read_text(encoding="utf-8").splitlines()[0]
    provenance = dict(
        field.split("=", maxsplit=1) for field in header.removeprefix("# ").split()
    )

    assert provenance["seed"] == str(MOCK_POPULATION_SEED)
    assert provenance["num_samples"] == str(LARGE_CATALOG_SIZE)


def test_mock_catalog_defaults_cover_the_production_band(mock_catalog_factory) -> None:
    """The realistic default trades frequency resolution for catalog size."""
    catalog = mock_catalog_factory()
    waveform = catalog.waveform_metadata
    frequencies = np.asarray(waveform.frequencies)
    redshift = np.asarray(catalog.source_parameters["redshift"])

    assert waveform.frequencies.size == 512
    assert catalog.population_metadata.num_samples == 1024
    assert len(catalog.source_parameters) == 5
    assert waveform.df == CATALOG_DF
    assert waveform.minimum_frequency == 2.0
    assert waveform.maximum_frequency == 4096.0
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
    frequencies = jnp.asarray(
        catalogs[LARGE_CATALOG_SIZE].waveform_metadata.frequencies
    )
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
        polarization_power = jnp.asarray(catalog.polarization_power)
        samples = catalog_samples(catalog)
        total_merger_rate, _, _ = compute_merger_rate_distance_and_logprob(
            FIDUCIALS, samples, redshift_grid=make_redshift_grid()
        )
        contracted = spectral_density(
            polarization_power,
            jnp.ones(num_sources),
            total_merger_rate,
            average_mode="analytic_inclination",
        )
        analytic_values = np.asarray(analytic)
        contracted_values = np.asarray(contracted)
        # Above the smallest sampled cutoff, a finite catalog can be exactly
        # zero even while the analytic population spectrum remains positive.
        # Compare only on the common support where every sampled source emits.
        valid = np.all(np.asarray(polarization_power) > 0.0, axis=1)
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


def _analytic_spectral_density(frequencies: jax.Array) -> jax.Array:
    """The population spectrum the catalog contraction estimates.

    The same physics with no sampling anywhere: a uniform ordered mass prior
    over the pinned graph's component-mass bounds, the same Madau-Dickinson
    rate, and the same ISCO truncation the catalog's polarization power was
    built with.
    """
    mass_moments = uniform_prior_mass_moments(
        frequencies,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MOCK_MINIMUM_COMPONENT_MASS,
        maximum_component_mass=MOCK_MAXIMUM_COMPONENT_MASS,
        alpha=ISCO_ALPHA,
    )
    return analytic_spectral_density_from_mass_moments(
        frequencies,
        FIDUCIALS,
        _source_frame_merger_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
    )


def test_catalog_omega_gw_matches_the_analytic_spectrum(mock_catalog_factory) -> None:
    r"""The same comparison, carried through into :math:`\Omega_{\rm gw}` units.

    ``omega_gw_from_spectral_density`` and its inverse are the units the
    stochastic-background literature quotes in, and nothing else in this suite
    exercises them against a realistic spectrum.

    Three things are asserted, and the second is the interesting one:

    1. the catalog's :math:`\Omega_{\rm gw}` matches the analytic population's
       on the common support, at the tolerance
       ``test_catalog_contraction_matches_the_analytic_spectrum`` uses for
       :math:`S_h`;
    2. the *residual is flat across frequency*. Below the smallest sampled
       cutoff every source contributes at every bin and both spectra are
       essentially :math:`\propto f^{-7/3}`, so the whole Monte-Carlo error
       collapses to a single normalization offset. A frequency-dependent
       residual there would mean the catalog and the analytic path disagree
       about the *shape*, which no amount of resampling would fix;
    3. the pair round-trips. :math:`4\pi^2 f^3/(3H_0^2)` and its reciprocal are
       one multiplication each, so this holds to round-off or one of them is
       wrong.
    """
    catalog = mock_catalog_factory(
        num_sources=LARGE_CATALOG_SIZE, f_min=F_MIN, f_max=F_MAX, df=CATALOG_DF
    )
    frequencies = jnp.asarray(catalog.waveform_metadata.frequencies)
    polarization_power = jnp.asarray(catalog.polarization_power)
    samples = catalog_samples(catalog)
    total_merger_rate, _, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=make_redshift_grid()
    )
    contracted = spectral_density(
        polarization_power,
        jnp.ones(LARGE_CATALOG_SIZE),
        total_merger_rate,
        average_mode="analytic_inclination",
    )
    analytic = _analytic_spectral_density(frequencies)

    to_omega = partial(
        omega_gw_from_spectral_density,
        frequencies=frequencies,
        hubble_constant=FIDUCIALS["H0"],
    )
    omega_catalog = np.asarray(to_omega(contracted))
    omega_analytic = np.asarray(to_omega(analytic))

    # The same common-support restriction the S_h comparison uses: above the
    # smallest sampled cutoff a finite catalog can be exactly zero while the
    # analytic population spectrum is not.
    valid = np.all(np.asarray(polarization_power) > 0.0, axis=1)
    assert int(np.sum(valid)) >= 8, "too few fully-supported bins to compare"
    ratio = omega_catalog[valid] / omega_analytic[valid]
    assert np.all(np.isfinite(ratio))

    np.testing.assert_allclose(np.mean(ratio), 1.0, rtol=0.05, atol=0.0)
    np.testing.assert_allclose(ratio, np.mean(ratio), rtol=0.03, atol=0.0)

    recovered = spectral_density_from_omega_gw(
        jnp.asarray(omega_catalog),
        frequencies,
        hubble_constant=FIDUCIALS["H0"],
    )
    np.testing.assert_allclose(
        np.asarray(recovered)[valid],
        np.asarray(contracted)[valid],
        rtol=1e-12,
        atol=0.0,
    )
