"""The population and background stages, end to end.

Everything is generated for real on the closed-form inspiral: a tiny Poisson
population and a four-bin grid. The Monte Carlo check compares in units of the
sample's own standard error, so it holds for any seed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from astrogwb_tiny_pipeline import NETWORKS, WAVEFORM
from astrogwb_tiny_pipeline import population_metadata as _population

from astrogwb.detector import DetectorRegistry, isotropic_overlap_reduction_function
from astrogwb.populations.joint import EXTRINSIC_SITES
from astrogwb.simulators import (
    BackgroundArtifact,
    BackgroundSimulator,
    PopulationArtifact,
    PopulationSimulator,
    combine_pair_spectra,
    realization_key,
)
from astrogwb.waveform import WaveformMetadata, polarization_power, waveform_generator

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0


@pytest.fixture(scope="module")
def population() -> PopulationArtifact:
    """About fifty sources."""
    return PopulationSimulator(_population(1e-4))(realization_key(41, 0))


def test_population_is_reproducible(population: PopulationArtifact) -> None:
    redrawn = PopulationSimulator(_population(1e-4))(realization_key(41, 0))

    assert population.count > 0
    assert set(EXTRINSIC_SITES) <= set(population.source_parameters)
    assert redrawn.source_parameters.keys() == population.source_parameters.keys()
    for name, values in population.source_parameters.items():
        np.testing.assert_array_equal(redrawn.source_parameters[name], values)


@pytest.mark.parametrize(
    "approximant",
    ["AnalyticInspiral", pytest.param("IMRPhenomD", marks=pytest.mark.integration)],
)
def test_polarizations_reduce_to_the_generators_power(
    population: PopulationArtifact, approximant: str
) -> None:
    # A 1 Hz grid puts the band's edges on Ripple's FFT grid.
    generator = waveform_generator(
        WaveformMetadata.model_validate(
            {
                **WAVEFORM.model_dump(exclude={"alpha"}),
                "approximant": approximant,
                "frequency_resolution": 1.0,
            }
        )
    )
    sources = population.source_parameters

    plus, cross = generator.generate_polarizations_batch(sources)

    power = np.asarray(generator.generate_batch(sources))
    np.testing.assert_allclose(
        np.asarray(polarization_power(plus, cross)),
        power,
        rtol=1e-10,
        atol=1e-12 * np.abs(power).max(),
    )


def test_pair_spectra_average_to_the_orf_times_the_sky_averaged_spectrum() -> None:
    r"""Over isotropic sources, :math:`E[Y_{ab}] = \gamma_{ab} S_h`.

    One realization is split into groups; in each, the ratio of every pair
    spectrum to the group's sky-averaged spectrum :math:`\sum_i I_i / T`
    isolates the response, and its mean over groups is :math:`\gamma_{ab}`.
    """
    realization = PopulationSimulator(_population(4e-3))(realization_key(7, 0))
    background = BackgroundSimulator(
        WAVEFORM,
        DetectorRegistry(
            detectors=NETWORKS.detectors,
            networks={"triangle": NETWORKS.networks["triangle"]},
        ),
        chunk_size=8,
    )
    generator = waveform_generator(WAVEFORM)
    group = 16
    ratios = []
    for start in range(0, realization.count - group + 1, group):
        sources = {
            name: values[start : start + group]
            for name, values in realization.source_parameters.items()
        }
        part = dataclasses.replace(realization, source_parameters=sources)
        power = np.asarray(generator.generate_batch(sources)).sum(axis=1)
        spectra = background(part).spectra["triangle"]
        ratios.append(spectra / (power / realization.observation_seconds))
    ratios = np.asarray(ratios)
    expected = np.stack(
        [
            isotropic_overlap_reduction_function(
                background.frequencies, *NETWORKS.build_detectors(pair)
            )
            for pair in NETWORKS.pairs("triangle")
        ]
    )
    errors = ratios.std(axis=0, ddof=1) / np.sqrt(ratios.shape[0])

    assert ratios.shape[0] >= 50
    assert np.all(np.abs(ratios.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * errors)


def test_an_empty_realization_has_zero_spectra() -> None:
    empty = PopulationSimulator(_population(1e-12))(realization_key(41, 0))

    simulator = BackgroundSimulator(WAVEFORM, NETWORKS)
    background = simulator(empty)

    assert empty.count == 0
    assert all(values.shape == (0,) for values in empty.source_parameters.values())
    for name in NETWORKS.networks:
        pairs = background.spectra[name]
        assert pairs.shape == (len(NETWORKS.pairs(name)), simulator.frequencies.size)
        assert not np.any(pairs)


@pytest.fixture(scope="module")
def projected(population: PopulationArtifact) -> BackgroundArtifact:
    """The tiny population projected onto both networks."""
    return BackgroundSimulator(WAVEFORM, NETWORKS, chunk_size=8)(population)


@pytest.mark.parametrize("name", list(NETWORKS.networks))
def test_each_network_holds_its_rows_of_a_single_network_projection(
    projected: BackgroundArtifact,
    population: PopulationArtifact,
    name: str,
) -> None:
    alone_registry = DetectorRegistry(
        detectors=NETWORKS.detectors, networks={name: NETWORKS.networks[name]}
    )
    alone = BackgroundSimulator(WAVEFORM, alone_registry, chunk_size=8)(population)

    np.testing.assert_array_equal(projected.spectra[name], alone.spectra[name])


@pytest.mark.parametrize("name", list(NETWORKS.networks))
def test_combining_pair_spectra_of_a_pure_signal_recovers_its_spectrum(
    projected: BackgroundArtifact, name: str
) -> None:
    spectrum = 1e-47 * (1.0 + np.arange(projected.frequencies.size))
    overlap = projected.noise()[name].pair_overlap

    combined = combine_pair_spectra(
        overlap * spectrum, projected.frequencies, NETWORKS, name
    )

    np.testing.assert_allclose(combined, spectrum)


def test_custom_detector_geometry_is_used_for_projection(
    population: PopulationArtifact,
) -> None:
    custom = NETWORKS.detectors["E1"].model_copy(
        update={
            "geometry": dataclasses.replace(
                NETWORKS.detectors["E1"].geometry,
                longitude_rad=NETWORKS.detectors["E1"].geometry.longitude_rad + 0.2,
            )
        }
    )
    registry = DetectorRegistry(
        detectors={**NETWORKS.detectors, "X1": custom},
        networks={"custom": ("X1", "E2")},
    )
    reference = DetectorRegistry(
        detectors={**NETWORKS.detectors, "X1": NETWORKS.detectors["E1"]},
        networks={"custom": ("X1", "E2")},
    )
    moved = BackgroundSimulator(WAVEFORM, registry)(population).spectra["custom"]
    original = BackgroundSimulator(WAVEFORM, reference)(population).spectra["custom"]
    assert not np.array_equal(moved, original)


def test_psd_override_preserves_pair_spectra_and_changes_combination(
    population: PopulationArtifact,
) -> None:
    mixed = DetectorRegistry(
        detectors=NETWORKS.detectors,
        networks={"mixed": ("C1", "R1", "S1")},
    )
    changed = DetectorRegistry(
        detectors={
            **mixed.detectors,
            "C1": mixed.detectors["C1"].model_copy(
                update={"psd_reference": "ET_COBA_15km_psd.txt"}
            ),
        },
        networks=mixed.networks,
    )
    baseline = BackgroundSimulator(WAVEFORM, mixed)(population)
    overridden = BackgroundSimulator(WAVEFORM, changed)(population)
    np.testing.assert_array_equal(
        baseline.spectra["mixed"], overridden.spectra["mixed"]
    )
    frequencies = baseline.frequencies
    assert not np.array_equal(
        combine_pair_spectra(baseline.spectra["mixed"], frequencies, mixed, "mixed"),
        combine_pair_spectra(
            overridden.spectra["mixed"], frequencies, changed, "mixed"
        ),
    )


@pytest.fixture(scope="module")
def empty(population: PopulationArtifact) -> PopulationArtifact:
    """A population with no sources, so the background is pure noise."""
    return dataclasses.replace(
        population,
        source_parameters={k: v[:0] for k, v in population.source_parameters.items()},
    )


def test_pure_noise_matches_the_pair_scale(empty: PopulationArtifact) -> None:
    segment_duration = 600.0
    simulator = BackgroundSimulator(
        WAVEFORM, NETWORKS, chunk_size=8, segment_duration=segment_duration
    )
    residuals = []
    for draw in range(120):
        background = simulator(dataclasses.replace(empty, rng_key=(0, draw)))
        noise = background.noise()
        for name, spectrum in background.spectra.items():
            scale = noise[name].pair_scale
            residuals.append((spectrum / scale).ravel())
    residuals = np.concatenate(residuals)
    segments = int(empty.observation_seconds // segment_duration)
    size = residuals.size
    # A mean of unit-variance terms, and a variance whose excess kurtosis is
    # that of a product of Gaussians (6) averaged over the segments.
    mean_bound = MAX_STANDARD_ERRORS / np.sqrt(size)
    variance_bound = MAX_STANDARD_ERRORS * np.sqrt((2.0 + 6.0 / segments) / size)

    assert abs(residuals.mean()) < mean_bound
    assert abs(residuals.var() - 1.0) < variance_bound


def test_spectra_do_not_depend_on_the_segment_batch_size(
    population: PopulationArtifact,
) -> None:
    batch_size = 7
    reference = BackgroundSimulator(
        WAVEFORM, NETWORKS, chunk_size=8, segment_duration=60.0, segment_batch_size=64
    )(population)
    batched = BackgroundSimulator(
        WAVEFORM,
        NETWORKS,
        chunk_size=8,
        segment_duration=60.0,
        segment_batch_size=batch_size,
    )(population)

    for name, spectrum in reference.spectra.items():
        np.testing.assert_allclose(batched.spectra[name], spectrum)


def test_noisy_background_averages_to_the_incoherent_signal(
    population: PopulationArtifact,
    projected: BackgroundArtifact,
) -> None:
    """New labels and noise per key, same sources: the mean is ``5 K / T``."""
    segment_duration = 60.0
    simulator = BackgroundSimulator(
        WAVEFORM, NETWORKS, chunk_size=8, segment_duration=segment_duration
    )
    draws = np.stack(
        [
            simulator(dataclasses.replace(population, rng_key=(0, draw))).spectra[
                "pair"
            ]
            for draw in range(150)
        ]
    )
    expected = projected.spectra["pair"]
    error = draws.std(axis=0, ddof=1) / np.sqrt(len(draws))

    assert np.all(abs(draws.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * error)
