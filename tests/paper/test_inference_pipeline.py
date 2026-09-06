"""End-to-end coverage of the shared inference-input pipeline.

These tests build the smallest synthetic catalog pair that survives validation
and exercise the application-library preparation shared by pipeline scripts.

The catalogs are deliberately tiny and their frequency grid deliberately
straddles the analysis band, so the frequency slice actually selects a strict
subset -- which is what makes the "``samples`` must NOT be masked" assertion
meaningful.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from catalog_fixtures import make_catalog, save_catalog
from config_fixtures import example_raw
from numpyro.infer.util import log_density

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.detector import gaussian_bin_scale
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.paper.catalogs import (
    compute_proposal_logprob,
    load_run_catalog,
    samples_from_catalog,
)
from astrogwb.paper.config.catalogs import (
    MadauDickinsonProposal,
    MixtureProposal,
    ProposalComponent,
    UniformRedshiftProposal,
    resolve_proposal,
)
from astrogwb.paper.config.mcmc import (
    ProposalConfig,
    RunConfig,
    build_run_config,
)
from astrogwb.paper.inference import prepare_inference_inputs, prepare_observation
from astrogwb.sampling import gwb_spectral_density_model, spectral_density_model

pytestmark = pytest.mark.integration

# Uniform grid (df = 10 Hz); the band below keeps the middle three bins.
FREQUENCIES = np.linspace(10.0, 50.0, 5)
BAND = (20.0, 40.0)
N_BAND = 3
N_SOURCES = 8
# Catalogs span below the assembled analysis window (minimum_redshift 0.3):
# linspace(0.05, 1.5, 8) keeps 6 samples after truncation.
N_RETAINED = 6


def _write_catalog(
    path: Path,
    *,
    proposal: bool,
    seed: int,
    frequencies: np.ndarray = FREQUENCIES,
) -> Path:
    rng = np.random.default_rng(seed)
    n_freq = frequencies.size
    shape = (n_freq, N_SOURCES)
    redshift = np.linspace(0.05, 1.5, N_SOURCES)
    source_parameters: dict[str, Any] = {
        "redshift": redshift,
        # Only needs to be positive and finite: the model divides by the
        # catalog's fiducial distances rather than re-deriving them.
        "luminosity_distance": 1.0e3 * (1.0 + redshift),
        "mass_1": np.full(N_SOURCES, 1.4),
        "mass_2": np.full(N_SOURCES, 1.4),
    }
    catalog = make_catalog(
        frequencies=frequencies,
        polarization_power=rng.uniform(0.0, 1.0, size=shape),
        source_parameters=source_parameters,
        approximant="Toy",
        minimum_frequency=float(frequencies[0]),
        maximum_frequency=float(frequencies[-1]),
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=float(frequencies[1] - frequencies[0]),
    )
    save_catalog(path, catalog)
    return path


def _proposal(config: RunConfig) -> ProposalConfig:
    """The density production derives from the catalog file, built inline here.

    These tests write synthetic catalogs with no provenance attrs, so the
    descriptor is constructed from the run's own fiducials -- the same values a
    real catalog would have recorded, since check_fiducials_match requires them
    to agree.
    """
    return resolve_proposal(
        MixtureProposal(
            components=(
                ProposalComponent(
                    weight=1.0,
                    density=MadauDickinsonProposal(
                        z_min=0.0,
                        z_max=20.0,
                        gamma=config.fiducials["gamma"],
                        kappa=config.fiducials["kappa"],
                        z_peak=config.fiducials["z_peak"],
                        H0=config.fiducials["H0"],
                        Omega_m=config.fiducials["Omega_m"],
                    ),
                ),
            )
        ),
        minimum_redshift=config.cosmology.minimum_redshift,
        maximum_redshift=config.cosmology.maximum_redshift,
        label="test-catalog",
    )


@pytest.fixture
def injection_catalog(tmp_path: Path) -> xr.Dataset:
    return load_run_catalog(
        _write_catalog(tmp_path / "injection.h5", proposal=False, seed=0),
        label="injection",
    )


@pytest.fixture
def proposal_catalog(tmp_path: Path) -> xr.Dataset:
    return load_run_catalog(
        _write_catalog(tmp_path / "proposal.h5", proposal=True, seed=1),
        label="proposal",
    )


def _config(**overrides: Any) -> RunConfig:
    raw = example_raw()
    f_min, f_max = BAND
    raw["analysis"] = {**raw["analysis"], "f_min": f_min, "f_max": f_max}
    raw["cosmology"] = {**raw["cosmology"], "n_grid": 32}
    raw["sampler"] = {
        **raw["sampler"],
        "num_warmup": 2,
        # 4 is the floor: `run` ends in mcmc.print_summary(), whose split R-hat
        # asserts at least four draws per chain.
        "num_samples": 4,
        "num_chains": 1,
        "progress_bar": False,
    }
    raw.update(overrides)
    return build_run_config(raw)


# --------------------------------------------------------------------------- #
# prepare_observation / prepare_inference_inputs
# --------------------------------------------------------------------------- #
def test_prepare_observation_keeps_arrays_unmasked(
    injection_catalog: xr.Dataset,
) -> None:
    config = _config()

    observation = prepare_observation(
        injection_catalog,
        fiducials=config.fiducials,
        grid=config.analysis_grid,
    )

    assert observation.frequencies.shape == FREQUENCIES.shape
    assert observation.spectral_density.shape == FREQUENCIES.shape
    assert observation.redshift_grid.shape == (config.cosmology.n_grid,)
    assert float(observation.total_merger_rate) > 0.0
    # The mask is carried alongside, not applied: notebooks plot the full band.
    assert observation.df == 10.0
    np.testing.assert_array_equal(
        np.asarray(observation.frequency_mask), [False, True, True, True, False]
    )
    np.testing.assert_allclose(
        np.asarray(observation.frequencies)[np.asarray(observation.frequency_mask)],
        [20.0, 30.0, 40.0],
    )


def test_prepared_catalog_slices_frequency_arrays_but_not_samples(
    injection_catalog: xr.Dataset, proposal_catalog: xr.Dataset
) -> None:
    config = _config()

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=config.fiducials,
        proposal_config=_proposal(config),
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )
    kwargs = inputs.masked_model_kwargs()
    catalog = inputs.estimator.catalog

    assert kwargs["observed_spectral_density"].shape == (N_BAND,)
    assert kwargs["scale"].shape == (N_BAND,)
    assert catalog.polarization_power.shape == (N_BAND, N_RETAINED)
    # `source_parameters` is per-source, not per-frequency. Masking it would
    # silently truncate the population and change every posterior without
    # erroring; the cached proposal arrays follow the same axis.
    for name, values in catalog.source_parameters.items():
        assert values.shape == (N_RETAINED,), name
    assert catalog.proposal_log_prob.shape == (N_RETAINED,)
    assert catalog.log_reference_distance.shape == (N_RETAINED,)
    # The likelihood takes data only: the catalog and the averaging convention
    # travel on the estimator, and the bin width is consumed into `scale`.
    assert set(kwargs) == {"observed_spectral_density", "scale"}


def test_masked_model_kwargs_scale_is_the_masked_gaussian_bin_scale(
    injection_catalog: xr.Dataset, proposal_catalog: xr.Dataset
) -> None:
    """The scale is now prepared here, not derived inside the sampling model."""
    config = _config()

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=config.fiducials,
        proposal_config=_proposal(config),
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )
    mask = np.asarray(inputs.observation.frequency_mask)

    np.testing.assert_allclose(
        np.asarray(inputs.masked_model_kwargs()["scale"]),
        np.asarray(
            gaussian_bin_scale(
                inputs.effective_psd[mask],
                config.analysis_grid.observation_time,
                # The catalog's bin width, never measured off the masked band.
                10.0,
            )
        ),
    )


def test_mismatched_frequency_grids_are_rejected(
    injection_catalog: xr.Dataset, proposal_catalog: xr.Dataset, tmp_path: Path
) -> None:
    shifted = _write_catalog(
        tmp_path / "shifted.h5",
        proposal=True,
        seed=2,
        frequencies=FREQUENCIES + 1.0,
    )
    config = _config()

    with pytest.raises(ValueError, match="identical frequency grids"):
        prepare_inference_inputs(
            injection_catalog,
            load_run_catalog(shifted, label="proposal"),
            fiducials=config.fiducials,
            proposal_config=_proposal(config),
            grid=config.analysis_grid,
            detectors=config.analysis.detectors,
        )


@pytest.mark.parametrize("uncovered", [0.0, np.inf])
def test_bins_without_network_coverage_narrow_the_band(
    injection_catalog: xr.Dataset,
    proposal_catalog: xr.Dataset,
    monkeypatch: pytest.MonkeyPatch,
    uncovered: float,
) -> None:
    """An uncovered bin is dropped, not fatal -- each survivor still has width df."""
    config = _config()
    effective_noise = np.ones(FREQUENCIES.shape)
    effective_noise[2] = uncovered
    monkeypatch.setattr(
        "astrogwb.paper.inference.compute_effective_psd",
        lambda *_args, **_kwargs: effective_noise,
    )

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=config.fiducials,
        proposal_config=_proposal(config),
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )
    kwargs = inputs.masked_model_kwargs()

    # The band was [20, 30, 40] Hz; 30 Hz is uncovered, so the surviving band is
    # gappy -- which is only sound because `df` is the catalog's attribute.
    np.testing.assert_array_equal(
        np.asarray(inputs.observation.frequency_mask),
        [False, True, False, True, False],
    )
    assert kwargs["scale"].shape == (N_BAND - 1,)
    assert kwargs["observed_spectral_density"].shape == (N_BAND - 1,)
    assert inputs.estimator.catalog.polarization_power.shape == (
        N_BAND - 1,
        N_RETAINED,
    )


def test_a_band_with_fewer_than_two_usable_bins_is_rejected(
    injection_catalog: xr.Dataset,
    proposal_catalog: xr.Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    effective_noise = np.full(FREQUENCIES.shape, np.inf)
    effective_noise[1] = 1.0
    monkeypatch.setattr(
        "astrogwb.paper.inference.compute_effective_psd",
        lambda *_args, **_kwargs: effective_noise,
    )

    with pytest.raises(ValueError, match="only 1 usable frequency bin"):
        prepare_inference_inputs(
            injection_catalog,
            proposal_catalog,
            fiducials=config.fiducials,
            proposal_config=_proposal(config),
            grid=config.analysis_grid,
            detectors=config.analysis.detectors,
        )


def test_catalog_without_stored_proposal_density_is_accepted(
    injection_catalog: xr.Dataset,
) -> None:
    config = _config()

    inputs = prepare_inference_inputs(
        injection_catalog,
        injection_catalog,
        fiducials=config.fiducials,
        proposal_config=_proposal(config),
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )

    assert "proposal_redshift_logpdf" not in inputs.proposal.parameter.values


# --------------------------------------------------------------------------- #
# Migration parity: the estimator path against the legacy callback path
#
# These are the acceptance tests for the estimator migration, and they can only
# exist while both paths are present. `propagate_catalog` divides the stored
# power by xi(z)^2 but leaves `source_parameters["luminosity_distance"]` as the
# *EM* distance; the legacy callback compensated for that internally, whereas
# `ImportanceCatalog` wants the effective distance the stored power already
# corresponds to. Getting that conversion wrong biases every weight by xi^2 --
# silently, since the weights stay finite and plausible either way.
# --------------------------------------------------------------------------- #
def _non_gr_config() -> RunConfig:
    """A config whose fiducial propagation is deliberately *not* GR.

    The shipped fiducials have ``xi_0 == 1``, which makes every GW/EM ratio
    exactly 1 -- so a parity test on them would pass whether the conversion is
    applied, omitted, or applied twice. Only ``xi_0`` and ``xi_n`` move; the MD
    constants ``check_fiducials_match`` pins are untouched.
    """
    return _config(fiducials={**example_raw()["fiducials"], "xi_0": 1.7, "xi_n": 2.3})


def _mixture_proposal(config: RunConfig) -> ProposalConfig:
    """An MD/uniform mixture: a density no single ``Population`` can express."""
    return resolve_proposal(
        MixtureProposal(
            components=(
                ProposalComponent(
                    weight=0.7,
                    density=MadauDickinsonProposal(
                        z_min=0.0,
                        z_max=20.0,
                        gamma=config.fiducials["gamma"],
                        kappa=config.fiducials["kappa"],
                        z_peak=config.fiducials["z_peak"],
                        H0=config.fiducials["H0"],
                        Omega_m=config.fiducials["Omega_m"],
                    ),
                ),
                ProposalComponent(
                    weight=0.3,
                    density=UniformRedshiftProposal(z_min=0.0, z_max=20.0),
                ),
            )
        ),
        minimum_redshift=config.cosmology.minimum_redshift,
        maximum_redshift=config.cosmology.maximum_redshift,
        label="test-catalog",
    )


def test_log_reference_distance_is_the_effective_distance_of_the_stored_power(
    injection_catalog: xr.Dataset, proposal_catalog: xr.Dataset
) -> None:
    config = _non_gr_config()

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=config.fiducials,
        proposal_config=_proposal(config),
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )

    # `inputs.proposal` carries the untouched EM distances; the propagation
    # correction lives only in its power array and in the cached reference.
    samples = samples_from_catalog(inputs.proposal)
    em_distance = np.asarray(samples["luminosity_distance"])
    expected = np.log(em_distance) + np.asarray(
        log_gw_em_ratio(
            samples["redshift"], config.fiducials["xi_0"], config.fiducials["xi_n"]
        )
    )

    np.testing.assert_array_equal(
        np.asarray(inputs.estimator.catalog.log_reference_distance), expected
    )
    # The conversion is not a no-op at these fiducials, which is what makes the
    # equality above worth asserting.
    assert not np.allclose(expected, np.log(em_distance))


@pytest.mark.parametrize("mixture", [False, True], ids=["ordinary", "mixture"])
@pytest.mark.parametrize("offset", [0.0, 0.13], ids=["fiducial", "off-fiducial"])
def test_estimator_reproduces_the_legacy_model_log_density(
    injection_catalog: xr.Dataset,
    proposal_catalog: xr.Dataset,
    mixture: bool,
    offset: float,
) -> None:
    """End-to-end: the same log posterior and the same diagnostics."""
    config = _non_gr_config()
    proposal_config = _mixture_proposal(config) if mixture else _proposal(config)

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=config.fiducials,
        proposal_config=proposal_config,
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )

    observation = inputs.observation
    mask = np.asarray(observation.frequency_mask)
    samples = samples_from_catalog(inputs.proposal)
    band = inputs.proposal.isel(frequency=mask)
    legacy_kwargs: dict[str, Any] = {
        "polarization_power": jnp.asarray(band.polarization_power.values),
        "samples": samples,
        "observed_spectral_density": observation.spectral_density[mask],
        "effective_psd": inputs.effective_psd[mask],
        "observation_time": inputs.observation_time,
        "df": observation.df,
    }
    legacy_model = partial(
        spectral_density_model,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=make_merger_rate_and_log_weights_fn(
            fiducials=dict(config.fiducials),
            redshift_grid=observation.redshift_grid,
            proposal_logprob=compute_proposal_logprob(
                samples["redshift"], proposal_config
            ),
        ),
        priors=config.priors,
    )
    migrated_model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=inputs.estimator,
        priors=config.priors,
    )

    params = {
        name: float(value) * (1.0 + offset)
        for name, value in config.fiducials.items()
        if name in config.priors
    }
    legacy_density, legacy_trace = log_density(legacy_model, (), legacy_kwargs, params)
    migrated_density, migrated_trace = log_density(
        migrated_model, (), inputs.masked_model_kwargs(), params
    )

    # The target side forms the effective distance linearly and then logs it,
    # where the legacy path added two logs -- so parity is scientific, not
    # bitwise. The prior terms are identical and cancel.
    np.testing.assert_allclose(
        float(migrated_density), float(legacy_density), rtol=1e-9
    )
    for site in ("total_merger_rate", "importance_relative_ess"):
        np.testing.assert_allclose(
            np.asarray(migrated_trace[site]["value"]),
            np.asarray(legacy_trace[site]["value"]),
            rtol=1e-9,
            err_msg=site,
        )
