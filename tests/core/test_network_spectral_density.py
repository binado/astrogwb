"""The network-response spectral-density simulator, on the closed-form inspiral.

Like ``test_simulators.py``, everything is generated for real on a tiny
population and a four-bin grid. The Monte Carlo check compares in units of the
draws' own standard error, so it holds for any seed.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.detector import network_response, stokes_cross_power
from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityMetadata,
    BackgroundSpectralDensitySimulator,
    JointNetworkSpectralDensitySimulator,
    NetworkSpectralDensityMetadata,
    NetworkSpectralDensitySimulator,
    draw_extrinsic_angles,
    network_spectra_ensemble,
    spectra_ensemble_path,
)
from astrogwb.waveform import WaveformMetadata, quadrupolar_stokes_factors

#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0
DETECTORS = ["S1", "R1", "C1"]
PSD_REFERENCES = {
    "S1": "ET_COBA_15km_psd.txt",
    "R1": "ET_COBA_15km_psd.txt",
    "C1": "cosmic_explorer_strain_psd.txt",
}


@pytest.fixture
def background_metadata() -> BackgroundSpectralDensityMetadata:
    return BackgroundSpectralDensityMetadata(
        waveform=WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=20.0,
            maximum_frequency=50.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            frequency_resolution=10.0,
        ),
        population=PopulationMetadata(
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
            "local_merger_rate": DistributionConfig(
                dist="Uniform", kwargs={"low": 700.0, "high": 800.0}
            ),
        },
        observation_time=1.0,
        count="fixed",
        num_events=4,
    )


@pytest.fixture
def network_metadata(
    background_metadata: BackgroundSpectralDensityMetadata,
) -> NetworkSpectralDensityMetadata:
    return NetworkSpectralDensityMetadata(
        **background_metadata.model_dump(),
        detectors=DETECTORS,
        psd_references=PSD_REFERENCES,
    )


@pytest.fixture
def network_simulator(
    network_metadata: NetworkSpectralDensityMetadata,
) -> NetworkSpectralDensitySimulator:
    return NetworkSpectralDensitySimulator(network_metadata, chunk_size=4)


def test_network_simulator_draws_the_background_simulators_population(
    network_simulator: NetworkSpectralDensitySimulator,
    background_metadata: BackgroundSpectralDensityMetadata,
) -> None:
    background = BackgroundSpectralDensitySimulator(background_metadata, chunk_size=4)
    key = batch_keys(41, 1)[0]

    network_draw, background_draw = network_simulator(key), background(key)

    np.testing.assert_array_equal(
        network_draw["total_merger_rate"], background_draw["total_merger_rate"]
    )
    np.testing.assert_array_equal(
        network_draw["frequencies"], background_draw["frequencies"]
    )


def test_network_simulator_is_deterministic_for_a_key(
    network_simulator: NetworkSpectralDensitySimulator,
) -> None:
    key = batch_keys(41, 1)[0]

    np.testing.assert_array_equal(
        network_simulator(key)["spectral_density"],
        network_simulator(key)["spectral_density"],
    )


def test_network_spectrum_averages_to_the_sky_averaged_spectrum(
    network_simulator: NetworkSpectralDensitySimulator,
    background_metadata: BackgroundSpectralDensityMetadata,
) -> None:
    """Draw by draw the sources agree, so the ratio isolates the response."""
    background = BackgroundSpectralDensitySimulator(background_metadata, chunk_size=4)
    keys = batch_keys(7, 1_000)

    ratios = np.array(
        [
            network_simulator(key)["spectral_density"][0]
            / background(key)["spectral_density"][0]
            for key in keys
        ]
    )
    errors = ratios.std(axis=0) / np.sqrt(ratios.shape[0])

    assert np.all(np.abs(ratios.mean(axis=0) - 1.0) < MAX_STANDARD_ERRORS * errors)
    assert np.all(ratios.std(axis=0) > 0.0)


def test_network_simulator_combines_pairs_as_the_network_response(
    network_simulator: NetworkSpectralDensitySimulator,
    network_metadata: NetworkSpectralDensityMetadata,
) -> None:
    """One unit-amplitude source gives <g> times the likelihood's response."""
    frequencies = np.asarray(network_simulator._generator.frequencies)
    angles = draw_extrinsic_angles(jax.random.key(3), 50)
    cos_inclination = jax.random.uniform(
        jax.random.key(4), (50,), minval=-1.0, maxval=1.0
    )
    stokes = quadrupolar_stokes_factors(cos_inclination)[..., None]
    pair_sums = np.stack(
        [
            stokes_cross_power(
                frequencies,
                DETECTORS[first],
                DETECTORS[second],
                angles["polar_cosine"],
                angles["azimuth"],
                angles["polarization"],
                stokes,
            )
            for first, second in [(0, 1), (0, 2), (1, 2)]
        ]
    )

    combined = network_simulator.combine(pair_sums)
    response = network_response(
        frequencies,
        DETECTORS,
        network_metadata.sensitivities(),
        angles["polar_cosine"],
        angles["azimuth"],
        angles["polarization"],
        cos_inclination,
    )

    np.testing.assert_allclose(combined, 0.8 * np.asarray(response), rtol=1e-10)


def test_network_metadata_key_depends_on_the_noise_curves(
    network_metadata: NetworkSpectralDensityMetadata,
) -> None:
    other = network_metadata.model_copy(
        update={"psd_references": {**PSD_REFERENCES, "C1": "ET_COBA_15km_psd.txt"}}
    )

    assert other.key() != network_metadata.key()


@pytest.fixture
def joint_metadata(
    background_metadata: BackgroundSpectralDensityMetadata,
) -> list[NetworkSpectralDensityMetadata]:
    """Two networks sharing detectors, one pair in the opposite order."""
    return [
        NetworkSpectralDensityMetadata(
            **background_metadata.model_dump(),
            detectors=detectors,
            psd_references={name: PSD_REFERENCES[name] for name in detectors},
        )
        for detectors in (DETECTORS, ["C1", "R1"])
    ]


@pytest.mark.parametrize("index", [0, 1])
def test_joint_simulator_draws_what_each_network_simulator_draws(
    joint_metadata: list[NetworkSpectralDensityMetadata], index: int
) -> None:
    key = batch_keys(41, 1)[0]
    joint = JointNetworkSpectralDensitySimulator(joint_metadata, chunk_size=4)(key)

    single = NetworkSpectralDensitySimulator(joint_metadata[index], chunk_size=4)(key)

    np.testing.assert_allclose(
        joint[index]["spectral_density"], single["spectral_density"], rtol=1e-12
    )


def test_network_spectra_ensemble_reads_back_the_draws_it_wrote(
    tmp_path: Path, joint_metadata: list[NetworkSpectralDensityMetadata]
) -> None:
    drawn = network_spectra_ensemble(joint_metadata, 41, 2, tmp_path, chunk_size=4)

    cached = network_spectra_ensemble(joint_metadata, 41, 2, tmp_path, chunk_size=4)

    for fresh, read in zip(drawn, cached, strict=True):
        np.testing.assert_array_equal(
            read["spectral_density"], fresh["spectral_density"]
        )


def test_network_spectra_ensemble_caches_each_network_under_its_own_key(
    tmp_path: Path, joint_metadata: list[NetworkSpectralDensityMetadata]
) -> None:
    network_spectra_ensemble(joint_metadata, 41, 1, tmp_path, chunk_size=4)

    assert all(
        spectra_ensemble_path(tmp_path, record, 41, 0).is_file()
        for record in joint_metadata
    )


def test_extrinsic_angles_are_isotropic_columns() -> None:
    angles = draw_extrinsic_angles(jax.random.key(0), 10_000)

    assert set(angles) == {"polar_cosine", "azimuth", "polarization", "coa_phase"}
    assert all(values.shape == (10_000,) for values in angles.values())
    assert float(jnp.abs(angles["polar_cosine"].mean())) < 0.05


@pytest.mark.integration
def test_network_spectra_round_trip_through_a_file(
    tmp_path: Path,
    network_simulator: NetworkSpectralDensitySimulator,
    network_metadata: NetworkSpectralDensityMetadata,
) -> None:
    fresh = network_simulator(batch_keys(41, 1)[0])
    path = write(tmp_path / "network.h5", fresh, network_metadata, seed=41)

    data, metadata, _ = load(path, NetworkSpectralDensityMetadata)

    assert metadata == network_metadata
    np.testing.assert_array_equal(
        data["spectral_density"],
        fresh["spectral_density"],
    )
