"""End-to-end coverage of the shared inference-input pipeline and its runners.

Nothing exercised ``run_mcmc.run`` or ``profile_model.build_potential`` before:
both were long, near-identical, catalog-dependent functions with no test. These
build the smallest synthetic catalog pair that survives validation and drive
the real entrypoints through it.

The catalogs are deliberately tiny and their frequency grid deliberately
straddles the analysis band, so the frequency slice actually selects a strict
subset -- which is what makes the "``samples`` must NOT be masked" assertion
meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from astrogwb.waveform import make_catalog, save_catalog
from astrogwb_paper.catalogs import CatalogSource
from astrogwb_paper.cli.profile_model import build_potential
from astrogwb_paper.cli.run_mcmc import run
from astrogwb_paper.config.banks import MadauDickinsonProposal, resolve_proposal
from astrogwb_paper.config.mcmc import (
    CatalogSpec,
    ProposalConfig,
    RunConfig,
    build_run_config,
)
from astrogwb_paper.inference import prepare_inference_inputs, prepare_observation
from config_fixtures import example_raw

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


def _source(
    bank_path: Path, *, num_samples: int = N_SOURCES, role: str = "proposal"
) -> CatalogSource:
    spec = CatalogSpec(md_bank="test-bank", num_samples=num_samples)
    return CatalogSource(bank_path, None, spec, role)


def _proposal(config: RunConfig) -> ProposalConfig:
    """The density production derives from bank provenance, built inline here.

    These tests write synthetic banks with no provenance attrs, so the
    descriptor is constructed from the run's own fiducials -- the same values a
    real bank would have recorded, since check_fiducials_match requires them to
    agree.
    """
    return resolve_proposal(
        MadauDickinsonProposal(
            z_min=0.0,
            z_max=20.0,
            gamma=config.fiducials["gamma"],
            kappa=config.fiducials["kappa"],
            z_peak=config.fiducials["z_peak"],
            H0=config.fiducials["H0"],
            Omega_m=config.fiducials["Omega_m"],
        ),
        None,
        uniform_mixing_fraction=0.0,
        minimum_redshift=config.cosmology.minimum_redshift,
        maximum_redshift=config.cosmology.maximum_redshift,
    )


@pytest.fixture
def injection_catalog(tmp_path: Path) -> CatalogSource:
    return _source(
        _write_catalog(tmp_path / "injection.h5", proposal=False, seed=0),
        role="injection",
    )


@pytest.fixture
def proposal_catalog(tmp_path: Path) -> CatalogSource:
    return _source(_write_catalog(tmp_path / "proposal.h5", proposal=True, seed=1))


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
    injection_catalog: CatalogSource,
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


def test_masked_model_kwargs_slices_frequency_arrays_but_not_samples(
    injection_catalog: CatalogSource, proposal_catalog: CatalogSource
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

    assert kwargs["observed_spectral_density"].shape == (N_BAND,)
    assert kwargs["effective_psd"].shape == (N_BAND,)
    assert kwargs["polarization_power"].shape == (N_BAND, N_RETAINED)
    # The bin width is the catalog's, never measured off the masked band.
    assert kwargs["df"] == 10.0
    assert kwargs["observation_time"] == config.analysis_grid.observation_time
    assert "frequencies" not in kwargs
    # `samples` is per-source, not per-frequency. Masking it would silently
    # truncate the population and change every posterior without erroring.
    for name, values in kwargs["samples"].items():
        assert values.shape == (N_RETAINED,), name


def test_mismatched_frequency_grids_are_rejected(
    injection_catalog: CatalogSource, proposal_catalog: CatalogSource, tmp_path: Path
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
            _source(shifted),
            fiducials=config.fiducials,
            proposal_config=_proposal(config),
            grid=config.analysis_grid,
            detectors=config.analysis.detectors,
        )


@pytest.mark.parametrize("uncovered", [0.0, np.inf])
def test_bins_without_network_coverage_narrow_the_band(
    injection_catalog: CatalogSource,
    proposal_catalog: CatalogSource,
    monkeypatch: pytest.MonkeyPatch,
    uncovered: float,
) -> None:
    """An uncovered bin is dropped, not fatal -- each survivor still has width df."""
    config = _config()
    effective_noise = np.ones(FREQUENCIES.shape)
    effective_noise[2] = uncovered
    monkeypatch.setattr(
        "astrogwb_paper.inference.compute_effective_psd",
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
    assert kwargs["effective_psd"].shape == (N_BAND - 1,)
    assert kwargs["observed_spectral_density"].shape == (N_BAND - 1,)
    assert kwargs["df"] == 10.0


def test_a_band_with_fewer_than_two_usable_bins_is_rejected(
    injection_catalog: CatalogSource,
    proposal_catalog: CatalogSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    effective_noise = np.full(FREQUENCIES.shape, np.inf)
    effective_noise[1] = 1.0
    monkeypatch.setattr(
        "astrogwb_paper.inference.compute_effective_psd",
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
    injection_catalog: CatalogSource,
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
# The two runner entrypoints
# --------------------------------------------------------------------------- #
def test_build_potential_returns_a_finite_potential(
    injection_catalog: CatalogSource, proposal_catalog: CatalogSource
) -> None:
    config = _config()

    potential_fn, init_params = build_potential(
        config, injection_catalog, proposal_catalog, _proposal(config), jax
    )

    assert set(init_params) == set(config.sampled_params)
    assert np.isfinite(float(potential_fn(init_params)))


def test_run_samples_every_sampled_parameter(
    injection_catalog: CatalogSource, proposal_catalog: CatalogSource
) -> None:
    config = _config()

    mcmc, marginalization = run(
        config,
        injection_catalog,
        proposal_catalog,
        _proposal(config),
        jax,
        "sequential",
    )

    assert marginalization is None
    posterior = mcmc.get_samples()
    for name in config.sampled_params:
        assert posterior[name].shape == (config.sampler.num_samples,), name
