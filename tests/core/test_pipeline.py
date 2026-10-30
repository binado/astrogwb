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
from astrogwb.populations.joint import EXTRINSIC_SITES
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
from astrogwb.waveform import WaveformMetadata, polarization_power

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0


@pytest.fixture(scope="module")
def population() -> PopulationArtifact:
    """About fifty sources."""
    return PopulationSimulator(_population(1e-4))(batch_keys(41, 1)[0])


def test_population_is_reproducible(population: PopulationArtifact) -> None:
    redrawn = PopulationSimulator(_population(1e-4))(batch_keys(41, 1)[0])

    assert population.count > 0
    assert set(EXTRINSIC_SITES) <= set(population.source_parameters)
    assert redrawn.source_parameters.keys() == population.source_parameters.keys()
    for name, values in population.source_parameters.items():
        np.testing.assert_array_equal(redrawn.source_parameters[name], values)


def test_population_changes_with_seed(population: PopulationArtifact) -> None:
    other = PopulationSimulator(population.metadata)(batch_keys(42, 1)[0])
    assert other.key() != population.key()
    assert not np.array_equal(
        other.source_parameters["redshift"], population.source_parameters["redshift"]
    )


def test_population_deterministics_replay_exactly(
    population: PopulationArtifact,
) -> None:
    from astrogwb.populations import joint_model
    from astrogwb.populations.evaluation import evaluate_sources

    model = joint_model(
        *population.metadata.population.build()(population.hyperparameters)
    )
    _, replayed = evaluate_sources(
        model, population.source_parameters, density_sites=()
    )
    for name, values in replayed.items():
        np.testing.assert_array_equal(values, population.source_parameters[name])


@pytest.mark.parametrize("sample_prior", [False, True])
def test_population_hyperparameters(sample_prior: bool) -> None:
    from astrogwb.distributions.config import DistributionConfig

    metadata = _population()
    if sample_prior:
        metadata = metadata.model_copy(
            update={
                "hyperparameters": {
                    **metadata.hyperparameters,
                    "H0": DistributionConfig.model_validate(
                        {"dist": "Uniform", "kwargs": {"low": 60.0, "high": 80.0}}
                    ),
                }
            }
        )
    drawn = PopulationSimulator(metadata)(batch_keys(41, 1)[0])
    if sample_prior:
        assert 60.0 < drawn.hyperparameters["H0"] < 80.0
    else:
        assert drawn.hyperparameters == metadata.fixed
    for values in drawn.source_parameters.values():
        assert values.shape == (drawn.count,)
        assert np.all(np.isfinite(values))


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
    realization = PopulationSimulator(_population(4e-3))(batch_keys(7, 1)[0])
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
            isotropic_overlap_reduction_function(
                background.frequencies, *NETWORKS.build_detectors(pair)
            )
            for pair in NETWORKS.pairs("triangle")
        ]
    )
    errors = ratios.std(axis=0, ddof=1) / np.sqrt(ratios.shape[0])

    assert ratios.shape[0] >= 50
    assert np.all(np.abs(ratios.mean(axis=0) - expected) < MAX_STANDARD_ERRORS * errors)


def test_a_staged_realization_reads_back_the_fused_background(
    tmp_path: Path, population: PopulationArtifact
) -> None:
    fused = BackgroundSimulator(WAVEFORM, NETWORKS, chunk_size=8)(population)

    simulate_realization(tmp_path, population=_population(1e-4), seed=41)
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
    assert all(values.shape == (0,) for values in empty.source_parameters.values())
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


def test_native_coordinate_columns_and_ranges(population: PopulationArtifact) -> None:
    columns = population.source_parameters
    assert set(EXTRINSIC_SITES) == {
        "right_ascension",
        "declination",
        "polarization_angle",
        "coa_phase",
    }
    assert np.all(
        (columns["right_ascension"] >= 0) & (columns["right_ascension"] < 2 * np.pi)
    )
    assert np.all(abs(columns["declination"]) <= np.pi / 2)
    assert np.all(
        (columns["polarization_angle"] >= 0) & (columns["polarization_angle"] < np.pi)
    )
    for missing in ("right_ascension", "declination", "polarization_angle"):
        with pytest.raises(ValueError, match="regenerate"):
            dataclasses.replace(
                population,
                source_parameters={k: v for k, v in columns.items() if k != missing},
            )


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4])
def test_old_population_schema_rejected_before_hash_or_io(
    population: PopulationArtifact, tmp_path: Path, version: int | None
) -> None:
    tables = population.tables()
    if version is None:
        tables["population"].pop("schema_version")
    else:
        tables["population"]["schema_version"] = version
    tables["population"]["key"] = "invalid hash"
    with pytest.raises(ValueError, match="regenerate.*schema 5"):
        PopulationArtifact.from_tables(tables, tmp_path / "absent.h5")


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_old_background_schema_rejected_before_io(
    tmp_path: Path, version: int | None
) -> None:
    from astrogwb.simulators.background import background_table

    table = background_table("waveforms", NETWORKS)
    if version is None:
        table.pop("schema_version")
    else:
        table["schema_version"] = version
    table["key"] = "invalid hash"
    with pytest.raises(ValueError, match="regenerate.*schema 4"):
        BackgroundArtifact.open(
            tmp_path / "absent.h5",
            table=table,
            realization={},
            waveform_key="waveforms",
        )


def test_projection_and_coordinate_conventions_are_hashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrogwb.simulators import background, population

    metadata = _population()
    old_population_key = metadata.key()
    monkeypatch.setitem(
        population.COORDINATE_CONVENTIONS, "polarization_angle", "different basis"
    )
    assert metadata.key() != old_population_key
    old_background_key = background_key("waveforms", NETWORKS)
    monkeypatch.setitem(background.PROJECTION_CONVENTIONS, "gmst_rad", 1.0)
    assert background_key("waveforms", NETWORKS) != old_background_key
