"""End-to-end coverage of the shared inference-input pipeline and its runners.

Nothing exercised ``run_mcmc.run`` or ``profile_model.build_potential`` before:
both were long, near-identical, catalog-dependent functions with no test. These
build the smallest synthetic catalog pair that survives validation and drive
the real entrypoints through it.

The catalogs are deliberately tiny and their frequency grid deliberately
straddles the analysis band, so the frequency mask actually selects a strict
subset -- which is what makes the "``samples`` must NOT be masked" assertion
meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb_paper.cli.profile_model import build_potential
from astrogwb_paper.cli.run_mcmc import run
from astrogwb_paper.config.mcmc import RunConfig, build_run_config
from astrogwb_paper.inference import prepare_inference_inputs, prepare_observation
from config_fixtures import example_raw
from pluscross import WaveformCatalog, save_catalog

pytestmark = pytest.mark.integration

# Uniform grid (df = 10 Hz); the band below keeps the middle three bins.
FREQUENCIES = np.linspace(10.0, 50.0, 5)
BAND = (20.0, 40.0)
N_BAND = 3
N_SOURCES = 8


def _write_catalog(
    path: Path,
    *,
    proposal: bool,
    seed: int,
    frequencies: np.ndarray = FREQUENCIES,
) -> Path:
    rng = np.random.default_rng(seed)
    n_freq = frequencies.size
    shape = (N_SOURCES, n_freq)
    redshift = np.linspace(0.05, 1.5, N_SOURCES)
    source_parameters: dict[str, Any] = {
        "redshift": redshift,
        # Only needs to be positive and finite: the model divides by the
        # catalog's fiducial distances rather than re-deriving them.
        "luminosity_distance": 1.0e3 * (1.0 + redshift),
        "mass_1": np.full(N_SOURCES, 1.4),
        "mass_2": np.full(N_SOURCES, 1.4),
    }
    catalog = WaveformCatalog(
        frequencies=frequencies,
        plus=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        cross=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        source_parameters=source_parameters,
        approximant="Toy",
        minimum_frequency=float(frequencies[0]),
        maximum_frequency=float(frequencies[-1]),
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )
    save_catalog(path, catalog)
    return path


@pytest.fixture
def injection_catalog(tmp_path: Path) -> Path:
    return _write_catalog(tmp_path / "injection.h5", proposal=False, seed=0)


@pytest.fixture
def proposal_catalog(tmp_path: Path) -> Path:
    return _write_catalog(tmp_path / "proposal.h5", proposal=True, seed=1)


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
def test_prepare_observation_keeps_arrays_unmasked(injection_catalog: Path) -> None:
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
    assert int(jnp.sum(observation.frequency_mask)) == N_BAND
    np.testing.assert_allclose(
        np.asarray(observation.frequencies)[np.asarray(observation.frequency_mask)],
        [20.0, 30.0, 40.0],
    )


def test_masked_model_kwargs_masks_frequencies_but_not_samples(
    injection_catalog: Path, proposal_catalog: Path
) -> None:
    config = _config()

    inputs = prepare_inference_inputs(
        injection_catalog,
        proposal_catalog,
        fiducials=config.fiducials,
        proposal_config=config.proposal,
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )
    kwargs = inputs.masked_model_kwargs()

    assert kwargs["frequencies"].shape == (N_BAND,)
    assert kwargs["observed_spectral_density"].shape == (N_BAND,)
    assert kwargs["effective_psd"].shape == (N_BAND,)
    assert kwargs["polarization_power"].shape == (N_BAND, N_SOURCES)
    # `samples` is per-source, not per-frequency. Masking it would silently
    # truncate the population and change every posterior without erroring.
    assert kwargs["samples"] is inputs.proposal.samples
    for name, values in kwargs["samples"].items():
        assert values.shape == (N_SOURCES,), name


def test_mismatched_frequency_grids_are_rejected(
    injection_catalog: Path, proposal_catalog: Path, tmp_path: Path
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
            shifted,
            fiducials=config.fiducials,
            proposal_config=config.proposal,
            grid=config.analysis_grid,
            detectors=config.analysis.detectors,
        )


def test_catalog_without_stored_proposal_density_is_accepted(
    injection_catalog: Path,
) -> None:
    config = _config()

    inputs = prepare_inference_inputs(
        injection_catalog,
        injection_catalog,
        fiducials=config.fiducials,
        proposal_config=config.proposal,
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
    )

    assert "proposal_redshift_logpdf" not in inputs.proposal.samples


# --------------------------------------------------------------------------- #
# The two runner entrypoints
# --------------------------------------------------------------------------- #
def test_build_potential_returns_a_finite_potential(
    injection_catalog: Path, proposal_catalog: Path
) -> None:
    config = _config()

    potential_fn, init_params = build_potential(
        config, injection_catalog, proposal_catalog, jax
    )

    assert set(init_params) == set(config.sampled_params)
    assert np.isfinite(float(potential_fn(init_params)))


def test_run_samples_every_sampled_parameter(
    injection_catalog: Path, proposal_catalog: Path
) -> None:
    config = _config()

    mcmc, marginalization = run(
        config, injection_catalog, proposal_catalog, jax, "sequential"
    )

    assert marginalization is None
    posterior = mcmc.get_samples()
    for name in config.sampled_params:
        assert posterior[name].shape == (config.sampler.num_samples,), name
