"""The population and background stages, end to end.

Everything is generated for real on the closed-form inspiral: a tiny Poisson
population and a four-bin grid. The Monte Carlo check compares in units of the
sample's own standard error, so it holds for any seed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from astrogwb_tiny_pipeline import NETWORKS, WAVEFORM, assert_same_spectra
from astrogwb_tiny_pipeline import population_metadata as _population

from astrogwb.detector import DetectorRegistry, isotropic_overlap_reduction_function
from astrogwb.simulators import (
    BackgroundArtifact,
    BackgroundSimulator,
    PopulationArtifact,
    PopulationSimulator,
    batch_keys,
    combine_pair_spectra,
    simulate_realization,
)
from astrogwb.simulators.background import background_key
from astrogwb.simulators.population import EXTRINSIC_COLUMNS
from astrogwb.waveform import WaveformMetadata, polarization_stokes

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0


@pytest.fixture(scope="module")
def population() -> PopulationArtifact:
    """About fifty sources."""
    return PopulationSimulator(_population(1e-4), chunk_size=16)(batch_keys(41, 1)[0])


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
def test_polarizations_reduce_to_the_generators_power(
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

    power = np.asarray(generator.generate_batch(sources))
    np.testing.assert_allclose(
        np.asarray(polarization_stokes(plus, cross)[0]),
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
    realization = PopulationSimulator(_population(4e-3), chunk_size=1024)(
        batch_keys(7, 1)[0]
    )
    background = BackgroundSimulator(
        WAVEFORM,
        DetectorRegistry(
            detectors=NETWORKS.detectors,
            networks={"triangle": NETWORKS.networks["triangle"]},
        ),
        chunk_size=8,
    )
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
        spectra = background(part).spectra["triangle"].pair_spectral_density
        ratios.append(spectra / (power / realization.observation_seconds))
    ratios = np.asarray(ratios)
    expected = np.stack(
        [
            isotropic_overlap_reduction_function(background.frequencies, first, second)
            for first, second in NETWORKS.pairs("triangle")
        ]
    )
    errors = ratios.std(axis=0, ddof=1) / np.sqrt(ratios.shape[0])

    assert ratios.shape[0] >= 50
    assert np.all(np.abs(ratios.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * errors)


def test_a_staged_realization_reads_back_the_fused_background(
    tmp_path: Path, population: PopulationArtifact
) -> None:
    fused = BackgroundSimulator(WAVEFORM, NETWORKS, chunk_size=8)(population)

    simulate_realization(
        tmp_path, population=_population(1e-4), seed=41, source_chunk_size=16
    )
    simulate_realization(tmp_path, waveform=WAVEFORM, chunk_size=8)
    staged = simulate_realization(
        tmp_path, background=NETWORKS, chunk_size=8
    ).background()

    assert_same_spectra(staged, fused)
    assert staged.key() == fused.key()
    assert staged.count == population.count
    assert tuple(staged.spectra) == tuple(sorted(NETWORKS.networks))


def test_an_empty_realization_has_zero_spectra() -> None:
    empty = PopulationSimulator(_population(1e-12))(batch_keys(41, 1)[0])

    simulator = BackgroundSimulator(WAVEFORM, NETWORKS)
    background = simulator(empty)

    assert empty.count == 0
    for name in NETWORKS.networks:
        pairs, combined = background.spectra[name]
        assert pairs.shape == (len(NETWORKS.pairs(name)), simulator.frequencies.size)
        assert not np.any(pairs) and not np.any(combined)


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

    np.testing.assert_array_equal(
        projected.spectra[name].pair_spectral_density,
        alone.spectra[name].pair_spectral_density,
    )


@pytest.mark.parametrize("name", list(NETWORKS.networks))
def test_the_network_spectrum_combines_its_pair_spectra(
    projected: BackgroundArtifact, name: str
) -> None:
    pairs, combined = projected.spectra[name]

    np.testing.assert_array_equal(
        combined,
        combine_pair_spectra(pairs, projected.frequencies, NETWORKS, name),
    )


def test_custom_detector_geometry_is_used_for_projection(
    population: PopulationArtifact,
) -> None:
    custom = NETWORKS.detectors["E1"].model_copy(
        update={
            "geometry": NETWORKS.detectors["E1"].geometry.model_copy(
                update={
                    "longitude_rad": NETWORKS.detectors["E1"].geometry.longitude_rad
                    + 0.2
                }
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
    moved = (
        BackgroundSimulator(WAVEFORM, registry)(population)
        .spectra["custom"]
        .pair_spectral_density
    )
    original = (
        BackgroundSimulator(WAVEFORM, reference)(population)
        .spectra["custom"]
        .pair_spectral_density
    )
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
    baseline = BackgroundSimulator(WAVEFORM, mixed)(population).spectra["mixed"]
    overridden = BackgroundSimulator(WAVEFORM, changed)(population).spectra["mixed"]
    np.testing.assert_array_equal(
        baseline.pair_spectral_density, overridden.pair_spectral_density
    )
    assert not np.array_equal(baseline.spectral_density, overridden.spectral_density)


@pytest.mark.parametrize(
    "detectors", [["S1", "R1", "C1"], ["C1", "R1", "S1"], ["R1", "C1", "S1"]]
)
def test_declaration_order_changes_neither_pairs_nor_key(
    detectors: list[str],
) -> None:
    reordered = DetectorRegistry(
        detectors=NETWORKS.detectors,
        networks={"triangle": tuple(detectors), "pair": NETWORKS.networks["pair"]},
    )

    assert reordered.pairs("triangle") == [("C1", "R1"), ("C1", "S1"), ("R1", "S1")]
    assert background_key("w", reordered, ["triangle"]) == background_key(
        "w", NETWORKS, ["triangle"]
    )


def test_noise_curves_are_part_of_the_key() -> None:
    other = NETWORKS.model_copy(
        update={
            "detectors": {
                **NETWORKS.detectors,
                "C1": NETWORKS.detectors["C1"].model_copy(
                    update={"psd_reference": "ET_COBA_15km_psd.txt"}
                ),
            }
        }
    )

    assert background_key("w", other) != background_key("w", NETWORKS)
