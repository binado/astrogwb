"""The two simulator nodes, end to end on the closed-form inspiral.

Each test generates for real -- a tiny population, a four-bin grid -- so what
is checked is the node, not a mock of it: a no-cache call and a cached call
agree (which also exercises the serialization round trip), and a seed names its
realization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import ComponentMetadata, PopulationMetadata
from astrogwb.simulators.core import Arrays, split_seed
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerCatalog,
    polarization_power,
)
from astrogwb.simulators.spectra import (
    SpectralDensityCatalog,
    SpectraMetadata,
    spectra,
)
from astrogwb.waveform import WaveformMetadata

FIDUCIALS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}
WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)
POPULATION = PopulationMetadata(
    model_name="bns_madau_dickinson",
    redshift=ComponentMetadata(
        model="madau_dickinson",
        kwargs={"minimum_redshift": 0.0, "maximum_redshift": 5.0, "n_grid": 64},
    ),
    mass=ComponentMetadata(model="ordered_uniform"),
)
CATALOG = CatalogMetadata(
    waveform=WAVEFORM, population=POPULATION, fiducials=FIDUCIALS, num_samples=6
)
SPECTRA = SpectraMetadata(
    waveform=WAVEFORM,
    population=POPULATION,
    hyperparameters={
        **{k: v for k, v in FIDUCIALS.items() if k != "local_merger_rate"},
        "local_merger_rate": DistributionConfig(
            dist="Uniform", kwargs={"low": 700.0, "high": 800.0}
        ),
    },
    observation_time=1.0,
    count="fixed",
    num_events=4,
)


def _leaf(tree: Arrays, name: str) -> np.ndarray:
    value = tree[name]
    assert not isinstance(value, dict)
    return value


def _same(first: Arrays, second: Arrays) -> None:
    assert first.keys() == second.keys()
    for name, value in first.items():
        other = second[name]
        if isinstance(value, dict):
            assert isinstance(other, dict)
            _same(value, other)
        else:
            assert not isinstance(other, dict)
            assert value.dtype == other.dtype
            np.testing.assert_array_equal(value, other)


@pytest.mark.integration
def test_polarization_power_cache_hit_equals_the_fresh_draw(tmp_path: Path) -> None:
    inputs = {"seed": np.uint64(41)}

    fresh = polarization_power(inputs, CATALOG)
    miss = polarization_power(inputs, CATALOG, cache_dir=tmp_path)
    hit = polarization_power(inputs, CATALOG, cache_dir=tmp_path, generate=False)

    _same(fresh, miss)
    _same(fresh, hit)
    catalog = PolarizationPowerCatalog.from_arrays(hit, CATALOG)
    assert catalog.num_samples == CATALOG.num_samples


@pytest.mark.integration
def test_polarization_power_seed_picks_the_realization() -> None:
    one = polarization_power({"seed": np.uint64(41)}, CATALOG)
    again = polarization_power({"seed": np.uint64(41)}, CATALOG)
    other = polarization_power({"seed": np.uint64(42)}, CATALOG)

    _same(one, again)
    assert not np.array_equal(
        _leaf(one, "polarization_power"), _leaf(other, "polarization_power")
    )


@pytest.mark.parametrize("seed", [41, np.int64(41), np.array([41], dtype=np.uint64)])
def test_polarization_power_rejects_a_seed_that_is_not_a_scalar_uint64(
    seed: ArrayLike,
) -> None:
    with pytest.raises(TypeError, match="uint64"):
        polarization_power({"seed": seed}, CATALOG)


@pytest.mark.integration
def test_spectra_cache_hit_equals_the_fresh_draw(tmp_path: Path) -> None:
    inputs = {"seeds": split_seed(41, 3)}

    fresh = spectra(inputs, SPECTRA, chunk_size=4)
    miss = spectra(inputs, SPECTRA, cache_dir=tmp_path, chunk_size=4)
    hit = spectra(inputs, SPECTRA, cache_dir=tmp_path, generate=False)

    _same(fresh, miss)
    _same(fresh, hit)
    catalog = SpectralDensityCatalog.from_arrays(hit, SPECTRA)
    assert catalog.num_draws == 3
    assert np.all(catalog.n_events == 4)


@pytest.mark.integration
def test_a_spectrum_depends_on_its_own_seed_alone() -> None:
    seeds = split_seed(41, 3)

    together = spectra({"seeds": seeds}, SPECTRA)
    alone = spectra({"seeds": seeds[1:2]}, SPECTRA)

    np.testing.assert_array_equal(
        _leaf(together, "spectral_density")[1], _leaf(alone, "spectral_density")[0]
    )


@pytest.mark.integration
def test_spectra_chunk_size_changes_cost_not_the_draws() -> None:
    inputs = {"seeds": split_seed(41, 2)}

    small = spectra(inputs, SPECTRA, chunk_size=1)
    large = spectra(inputs, SPECTRA, chunk_size=64)

    np.testing.assert_allclose(
        _leaf(small, "spectral_density"),
        _leaf(large, "spectral_density"),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("seeds", "error"),
    [
        (np.array([1, 2]), TypeError),
        (np.array([], dtype=np.uint64), TypeError),
        (np.array([[1, 2]], dtype=np.uint64), TypeError),
        (np.array([3, 3], dtype=np.uint64), ValueError),
    ],
)
def test_spectra_validates_its_seeds(seeds: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        spectra({"seeds": seeds}, SPECTRA)
