"""The population, waveform and background stages, end to end.

Everything is generated for real on the closed-form inspiral: a tiny Poisson
population and a four-bin grid. The Monte Carlo check compares in units of the
sample's own standard error, so it holds for any seed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from astrogwb.detector import isotropic_overlap_reduction_function
from astrogwb.populations import PopulationMetadata as PopulationSpec
from astrogwb.simulators import (
    BackgroundArtifact,
    BackgroundSimulator,
    DetectorConfiguration,
    PopulationArtifact,
    PopulationMetadata,
    PopulationSimulator,
    WaveformArtifact,
    WaveformSimulator,
    batch_keys,
)
from astrogwb.simulators.population import EXTRINSIC_COLUMNS
from astrogwb.waveform import WaveformMetadata
from astrogwb.waveform.polarization_power import polarization_stokes

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0

WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)
DETECTORS = DetectorConfiguration(
    detectors=["S1", "R1", "C1"],
    psd_references={
        "S1": "ET_COBA_15km_psd.txt",
        "R1": "ET_COBA_15km_psd.txt",
        "C1": "cosmic_explorer_strain_psd.txt",
    },
)


def _population(observation_time: float) -> PopulationMetadata:
    return PopulationMetadata(
        population=PopulationSpec(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": 0.0,
                "maximum_redshift": 5.0,
                "n_grid": 64,
            },
        ),
        hyperparameters={
            "H0": 67.66,
            "Omega_m": 0.3096,
            "gamma": 1.42,
            "kappa": 4.62,
            "z_peak": 1.84,
            "minimum_mass": 1.0,
            "mass_width": 1.5,
            "local_merger_rate": 750.0,
        },
        observation_time=observation_time,
    )


@pytest.fixture(scope="module")
def population() -> PopulationArtifact:
    """About fifty sources."""
    return PopulationSimulator(_population(1e-4), chunk_size=16)(batch_keys(41, 1)[0])


@pytest.fixture(scope="module")
def waveforms() -> WaveformSimulator:
    return WaveformSimulator(WAVEFORM, chunk_size=8)


@pytest.mark.parametrize("chunk_size", [3, 64])
def test_population_draws_its_extrinsics_whatever_the_chunk_size(
    population: PopulationArtifact, chunk_size: int
) -> None:
    redrawn = PopulationSimulator(_population(1e-4), chunk_size=chunk_size)(
        batch_keys(41, 1)[0]
    )

    assert population.count > 0
    assert set(EXTRINSIC_COLUMNS) <= set(population.source_parameters)
    assert redrawn.source_parameters.keys() == population.source_parameters.keys()
    for name, values in population.source_parameters.items():
        np.testing.assert_array_equal(redrawn.source_parameters[name], values)


@pytest.mark.parametrize(
    "approximant",
    ["AnalyticInspiral", pytest.param("IMRPhenomD", marks=pytest.mark.integration)],
)
def test_polarizations_reduce_to_the_generators_stokes_parameters(
    population: PopulationArtifact, approximant: str
) -> None:
    # A 1 Hz grid puts the band's edges on Ripple's FFT grid.
    generator = WaveformMetadata.model_validate(
        {
            **WAVEFORM.model_dump(exclude={"alpha"}),
            "approximant": approximant,
            "frequency_resolution": 1.0,
        }
    ).build()
    sources = population.source_parameters

    plus, cross = generator.generate_polarizations_batch(sources)

    stokes = np.asarray(generator.generate_stokes_batch(sources))
    np.testing.assert_allclose(
        np.asarray(polarization_stokes(plus, cross)),
        stokes,
        rtol=1e-10,
        atol=1e-12 * np.abs(stokes).max(),
    )


def test_pair_spectra_average_to_the_orf_times_the_sky_averaged_spectrum(
    waveforms: WaveformSimulator,
) -> None:
    r"""Over isotropic sources, :math:`E[Y_{ab}] = \gamma_{ab} S_h`.

    One realization is split into groups; in each, the ratio of every pair
    spectrum to the group's sky-averaged spectrum :math:`\sum_i I_i / T`
    isolates the response, and its mean over groups is :math:`\gamma_{ab}`.
    """
    realization = PopulationSimulator(_population(4e-3), chunk_size=1024)(
        batch_keys(7, 1)[0]
    )
    background = BackgroundSimulator(DETECTORS, chunk_size=8)
    generator = WAVEFORM.build()
    group = 16
    ratios = []
    for start in range(0, realization.count - group + 1, group):
        sources = {
            name: values[start : start + group]
            for name, values in realization.source_parameters.items()
        }
        part = dataclasses.replace(realization, source_parameters=sources)
        power = np.asarray(generator.generate_batch(sources)).sum(axis=1)
        spectra = background(waveforms(part)).pair_spectral_density
        ratios.append(spectra / (power / realization.observation_seconds))
    ratios = np.asarray(ratios)
    expected = np.stack(
        [
            isotropic_overlap_reduction_function(waveforms.frequencies, first, second)
            for first, second in DETECTORS.pairs()
        ]
    )
    errors = ratios.std(axis=0, ddof=1) / np.sqrt(ratios.shape[0])

    assert ratios.shape[0] >= 50
    assert np.all(np.abs(ratios.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * errors)


def test_standalone_stages_read_back_the_fused_background(
    tmp_path: Path, population: PopulationArtifact, waveforms: WaveformSimulator
) -> None:
    background = BackgroundSimulator(DETECTORS, chunk_size=8)
    fused = background(waveforms(population))

    population.write(tmp_path / "population")
    waveforms(PopulationArtifact.open(tmp_path / "population")).write(
        tmp_path / "waveforms"
    )
    background(WaveformArtifact.open(tmp_path / "waveforms")).write(
        tmp_path / "background"
    )
    standalone = BackgroundArtifact.open(tmp_path / "background" / "metadata.toml")

    np.testing.assert_array_equal(
        standalone.pair_spectral_density, fused.pair_spectral_density
    )
    assert standalone.key() == fused.key()
    assert standalone.count == population.count
    assert standalone.detectors == DETECTORS


def test_an_empty_realization_has_zero_spectra(
    tmp_path: Path, waveforms: WaveformSimulator
) -> None:
    empty = PopulationSimulator(_population(1e-12))(batch_keys(41, 1)[0])

    written = waveforms(empty).write(tmp_path / "waveforms")
    background = BackgroundSimulator(DETECTORS)(written)

    assert empty.count == 0
    assert background.pair_spectral_density.shape == (3, waveforms.frequencies.size)
    assert not np.any(background.pair_spectral_density)
