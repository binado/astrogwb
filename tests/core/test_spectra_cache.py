"""Spectra metadata, its content key, the generator, and the generic cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import jax
import numpy as np
import numpyro.distributions as dist
import pytest

import astrogwb
from astrogwb.metadata import PriorSpec
from astrogwb.simulators.core import artifact_path, simulate
from astrogwb.simulators.spectra import (
    SpectralDensityCatalog,
    SpectraMetadata,
    SpectrumGenerator,
    draw_spectral_density,
    padded_event_capacity,
)

MetadataFactory = Callable[..., SpectraMetadata]

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

RATE_PRIOR = {"dist": "Uniform", "kwargs": {"low": 500.0, "high": 1000.0}}


@pytest.fixture
def make_metadata() -> MetadataFactory:
    """A small, fast draw -- about seventy events -- overridable by keyword."""

    def build(**overrides: Any) -> SpectraMetadata:
        fields: dict[str, Any] = {
            "waveform": {
                "approximant": "AnalyticInspiral",
                "minimum_frequency": 10.0,
                "maximum_frequency": 16.0,
                "reference_frequency": 10.0,
                "sampling_frequency": 64.0,
                "frequency_resolution": 2.0,
            },
            "population": {
                "model_name": "bns_md_cosmological",
                "model_kwargs": {
                    "minimum_redshift": 0.0,
                    "maximum_redshift": 5.0,
                    "n_grid": 64,
                },
                "seed": 7,
            },
            "hyperparameters": FIDUCIALS,
            "num_draws": 3,
            "observation_time": 1e-3,
        }
        fields.update(overrides)
        return SpectraMetadata.model_validate(fields)

    return build


# --------------------------------------------------------------------- #
# The key
# --------------------------------------------------------------------- #
def test_count_mode_configuration_validation(make_metadata: MetadataFactory) -> None:
    poisson = make_metadata()
    assert poisson.count == "poisson"
    assert poisson.num_events is None
    assert poisson.n_max_sigma == 5.0
    assert poisson.key() == make_metadata(n_max_sigma=5.0).key()
    fixed = make_metadata(count="fixed", num_events=7)
    assert fixed.n_max_sigma is None
    assert (
        fixed.key()
        == make_metadata(count="fixed", num_events=7, n_max_sigma=None).key()
    )
    assert fixed.key() != make_metadata(count="fixed", num_events=8).key()

    population = poisson.population.build()
    assert population.merger_rate_fn is not None
    generator = poisson.waveform.build()
    invalid_settings: tuple[dict[str, Any], ...] = (
        {"count": "unknown"},
        {"count": "fixed"},
        {"count": "fixed", "num_events": 0},
        {"count": "fixed", "num_events": -1},
        {"count": "fixed", "num_events": True},
        {"count": "fixed", "num_events": 1.5},
        {"count": "fixed", "num_events": 7, "n_max_sigma": 5.0},
        {"count": "poisson", "num_events": 7},
        {"n_max_sigma": -1.0},
        {"observation_time": 0.0},
        {"count": "fixed", "num_events": 7, "observation_time": 0.0},
    )
    for settings in invalid_settings:
        with pytest.raises(ValueError):
            make_metadata(**settings)
        direct_settings: dict[str, Any] = {
            "observation_time": poisson.observation_time,
            **settings,
        }
        with pytest.raises(ValueError):
            draw_spectral_density(
                source_model=population.source_model,
                merger_rate_fn=population.merger_rate_fn,
                generator=generator,
                hyperparameters=poisson.fixed,
                num_draws=poisson.num_draws,
                rng_key=jax.random.key(0),
                **direct_settings,
            )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_draws": 4},
        {"observation_time": 2e-3},
        {"n_max_sigma": 4.0},
        {"count": "fixed", "num_events": 7},
        {"count": "fixed", "num_events": 8},
        {"version": "0.0.0-other"},
        {"hyperparameters": {**FIDUCIALS, "gamma": 1.5}},
        {"hyperparameters": {**FIDUCIALS, "local_merger_rate": RATE_PRIOR}},
    ],
)
def test_anything_that_changes_the_draws_changes_the_key(
    make_metadata: MetadataFactory, overrides: dict[str, Any]
) -> None:
    assert make_metadata(**overrides).key() != make_metadata().key()


def test_an_edited_prior_bound_changes_the_key(make_metadata: MetadataFactory) -> None:
    narrow = {"dist": "Uniform", "kwargs": {"low": 500.0, "high": 900.0}}
    wide = make_metadata(hyperparameters={**FIDUCIALS, "local_merger_rate": RATE_PRIOR})
    edited = make_metadata(hyperparameters={**FIDUCIALS, "local_merger_rate": narrow})

    assert wide.key() != edited.key()


def test_integer_and_float_spellings_share_a_key(
    make_metadata: MetadataFactory,
) -> None:
    as_float = make_metadata(
        hyperparameters={
            **FIDUCIALS,
            "H0": 70.0,
            "local_merger_rate": {
                "dist": "Uniform",
                "kwargs": {"low": 500.0, "high": 1e3},
            },
        }
    )
    as_int = make_metadata(
        hyperparameters={
            **FIDUCIALS,
            "H0": 70,
            "local_merger_rate": {
                "dist": "Uniform",
                "kwargs": {"low": 500, "high": 1000},
            },
        }
    )

    assert as_float.key() == as_int.key()


def test_fixed_and_sampled_split_by_type(make_metadata: MetadataFactory) -> None:
    metadata = make_metadata(
        hyperparameters={**FIDUCIALS, "local_merger_rate": RATE_PRIOR}
    )

    assert "local_merger_rate" not in metadata.fixed
    assert metadata.sampled == {
        "local_merger_rate": PriorSpec.model_validate(RATE_PRIOR)
    }


# --------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------- #
def test_padded_event_capacity_reaches_into_the_tail() -> None:
    assert padded_event_capacity(100.0, 5.0) == 150
    assert padded_event_capacity(0.0, 5.0) == 1
    with pytest.raises(ValueError, match="non-negative"):
        padded_event_capacity(-1.0, 5.0)


def _draw(metadata: SpectraMetadata, **hyperparameters: Any) -> Any:
    population = metadata.population.build()
    assert population.merger_rate_fn is not None
    return draw_spectral_density(
        source_model=population.source_model,
        merger_rate_fn=population.merger_rate_fn,
        generator=metadata.waveform.build(),
        hyperparameters={**metadata.fixed, **hyperparameters},
        observation_time=metadata.observation_time,
        num_draws=metadata.num_draws,
        rng_key=jax.random.key(0),
    )


def test_draws_are_reproducible_from_their_key(make_metadata: MetadataFactory) -> None:
    jax.config.update("jax_enable_x64", True)
    metadata = make_metadata()

    first, second = _draw(metadata), _draw(metadata)

    np.testing.assert_array_equal(first.spectral_density, second.spectral_density)
    np.testing.assert_array_equal(first.n_events, second.n_events)


def test_sampled_hyperparameters_vary_inside_their_prior(
    make_metadata: MetadataFactory,
) -> None:
    jax.config.update("jax_enable_x64", True)
    metadata = make_metadata()

    draws = _draw(metadata, local_merger_rate=dist.Uniform(500.0, 1000.0))

    rates = draws.hyperparameters["local_merger_rate"]
    assert np.unique(rates).size == metadata.num_draws
    assert np.all((rates >= 500.0) & (rates <= 1000.0))
    np.testing.assert_array_equal(draws.hyperparameters["H0"], FIDUCIALS["H0"])
    # The Poisson mean follows each row's own rate.
    assert np.unique(draws.total_merger_rate).size == metadata.num_draws


# --------------------------------------------------------------------- #
# The generator and the cache
# --------------------------------------------------------------------- #
def test_fixed_generation_cache_round_trip(
    make_metadata: MetadataFactory, tmp_path: Path
) -> None:
    metadata = make_metadata(count="fixed", num_events=7)
    generated = simulate(metadata, SpectrumGenerator(batch_size=3), tmp_path)
    loaded = simulate(metadata, SpectrumGenerator(), tmp_path, generate=False)

    assert loaded.metadata == metadata
    assert loaded.count == "fixed"
    assert loaded.num_events == 7
    assert loaded.n_max_sigma is None
    assert loaded.spectral_density.shape == (
        metadata.num_draws,
        loaded.frequencies.size,
    )
    assert loaded.total_merger_rate.shape == (metadata.num_draws,)
    assert all(
        values.shape == (metadata.num_draws,)
        for values in loaded.hyperparameters.values()
    )
    np.testing.assert_array_equal(loaded.n_events, np.full(metadata.num_draws, 7))
    np.testing.assert_array_equal(loaded.spectral_density, generated.spectral_density)
    with pytest.raises(ValueError, match="n_events.*num_events"):
        replace(loaded, n_events=np.zeros(metadata.num_draws, dtype=np.int64))


@dataclass
class _CountingGenerator:
    """A SpectrumGenerator that records how often it actually ran."""

    artifact: ClassVar[type[SpectralDensityCatalog]] = SpectralDensityCatalog
    calls: list[str] = field(default_factory=list)

    def __call__(self, metadata: SpectraMetadata) -> SpectralDensityCatalog:
        self.calls.append(metadata.key())
        return SpectrumGenerator()(metadata)


def test_generator_records_its_metadata(make_metadata: MetadataFactory) -> None:
    metadata = make_metadata(
        hyperparameters={**FIDUCIALS, "local_merger_rate": RATE_PRIOR}
    )

    spectra = SpectrumGenerator()(metadata)

    assert spectra.metadata == metadata
    assert spectra.version == astrogwb.__version__
    assert spectra.spectral_density.shape == (3, spectra.frequencies.size)


def test_batch_size_does_not_change_the_draws(make_metadata: MetadataFactory) -> None:
    metadata = make_metadata()

    small = SpectrumGenerator(batch_size=8)(metadata)
    large = SpectrumGenerator(batch_size=512)(metadata)

    np.testing.assert_array_equal(small.n_events, large.n_events)
    np.testing.assert_allclose(small.spectral_density, large.spectral_density)


def test_generator_refuses_metadata_for_another_version(
    make_metadata: MetadataFactory,
) -> None:
    with pytest.raises(ValueError, match="installed"):
        SpectrumGenerator()(make_metadata(version="0.0.0-other"))


def test_simulate_without_a_cache_always_generates(
    make_metadata: MetadataFactory,
) -> None:
    generator = _CountingGenerator()
    metadata = make_metadata()

    simulate(metadata, generator)
    simulate(metadata, generator)

    assert generator.calls == [metadata.key()] * 2


def test_miss_writes_under_the_key_and_hit_reuses_it(
    make_metadata: MetadataFactory, tmp_path: Path
) -> None:
    generator = _CountingGenerator()
    metadata = make_metadata()

    generated = simulate(metadata, generator, tmp_path)
    loaded = simulate(metadata, generator, tmp_path)

    assert generator.calls == [metadata.key()]
    assert artifact_path(metadata, tmp_path) == tmp_path / f"{metadata.key()}.h5"
    assert artifact_path(metadata, tmp_path).is_file()
    assert loaded.metadata == metadata
    np.testing.assert_array_equal(loaded.spectral_density, generated.spectral_density)
    np.testing.assert_array_equal(
        loaded.hyperparameters["H0"], generated.hyperparameters["H0"]
    )


def test_a_file_under_the_wrong_key_is_refused(
    make_metadata: MetadataFactory, tmp_path: Path
) -> None:
    requested = make_metadata()
    other = make_metadata(num_draws=2)
    SpectrumGenerator()(other).save(artifact_path(requested, tmp_path))

    with pytest.raises(ValueError, match="not the requested"):
        simulate(requested, _CountingGenerator(), tmp_path)
