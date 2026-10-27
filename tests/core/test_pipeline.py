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
from astrogwb_tiny_pipeline import DETECTORS, WAVEFORM
from astrogwb_tiny_pipeline import population_metadata as _population

from astrogwb.detector import isotropic_overlap_reduction_function
from astrogwb.simulators import (
    BackgroundSimulator,
    PopulationArtifact,
    PopulationSimulator,
    WaveformSimulator,
    batch_keys,
    simulate_realization,
)
from astrogwb.simulators.population import EXTRINSIC_COLUMNS
from astrogwb.waveform import WaveformMetadata
from astrogwb.waveform.polarization_power import polarization_stokes

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0


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


def test_a_staged_realization_reads_back_the_fused_background(
    tmp_path: Path, population: PopulationArtifact, waveforms: WaveformSimulator
) -> None:
    fused = BackgroundSimulator(DETECTORS, chunk_size=8)(waveforms(population))

    simulate_realization(
        tmp_path, population=_population(1e-4), seed=41, source_chunk_size=16
    )
    simulate_realization(
        tmp_path, waveform=WAVEFORM, keep_polarizations=True, chunk_size=8
    )
    staged = simulate_realization(
        tmp_path, background=DETECTORS, chunk_size=8
    ).background()

    np.testing.assert_array_equal(
        staged.pair_spectral_density, fused.pair_spectral_density
    )
    assert staged.key() == fused.key()
    assert staged.count == population.count
    assert staged.detectors == DETECTORS


def test_an_empty_realization_has_zero_spectra(
    tmp_path: Path, waveforms: WaveformSimulator
) -> None:
    empty = PopulationSimulator(_population(1e-12))(batch_keys(41, 1)[0])

    written = waveforms(empty).write(tmp_path / "polarization.h5")
    background = BackgroundSimulator(DETECTORS)(written)

    assert empty.count == 0
    assert background.pair_spectral_density.shape == (3, waveforms.frequencies.size)
    assert not np.any(background.pair_spectral_density)
