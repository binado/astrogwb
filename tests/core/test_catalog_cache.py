"""The catalog request, its content key, and the generate-or-load cache."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import astrogwb
from astrogwb.catalog import (
    PolarizationPowerCatalog,
    catalog_path,
    check_catalog_answers,
    generate,
    load_or_generate,
)
from astrogwb.metadata import CATALOG_KEY_LENGTH, CatalogRequest

RequestFactory = Callable[..., CatalogRequest]


@pytest.fixture
def fiducials() -> dict[str, float]:
    return {
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
        "local_merger_rate": 770.0,
        "minimum_mass": 1.0,
        "mass_width": 1.5,
    }


@pytest.fixture
def waveform() -> dict[str, Any]:
    return {
        "approximant": "AnalyticInspiral",
        "minimum_frequency": 10.0,
        "maximum_frequency": 16.0,
        "reference_frequency": 10.0,
        "sampling_frequency": 64.0,
        "frequency_resolution": 2.0,
    }


@pytest.fixture
def population() -> dict[str, Any]:
    return {
        "model_name": "bns_md_cosmological",
        "model_kwargs": {
            "minimum_redshift": 0.0,
            "maximum_redshift": 5.0,
            "n_grid": 64,
        },
    }


@pytest.fixture
def make_request(
    population: dict[str, Any],
    waveform: dict[str, Any],
    fiducials: dict[str, float],
) -> RequestFactory:
    """Build a small request, overriding any block by keyword."""

    def build(**overrides: Any) -> CatalogRequest:
        blocks: dict[str, Any] = {
            "population": population,
            "waveform": waveform,
            "fiducials": fiducials,
            "seed": 7,
            "num_samples": 16,
            "version": astrogwb.__version__,
        }
        blocks.update(overrides)
        return CatalogRequest.from_blocks(**blocks)

    return build


def test_key_of_fixed_request_matches_pinned_digest(
    make_request: RequestFactory,
) -> None:
    """Pinned so an accidental change to the canonical form is caught.

    Changing it on purpose invalidates every cached catalog, which is fine --
    but it should be a decision, so update this value deliberately.
    """
    key = make_request(version="0.0.0-test").key()

    assert len(key) == CATALOG_KEY_LENGTH
    assert key == "575d13b8e3f84c2d"


@pytest.mark.parametrize(
    "override",
    [
        lambda blocks: {"seed": 8},
        lambda blocks: {"num_samples": 32},
        lambda blocks: {"version": "0.0.0-other"},
        lambda blocks: {"fiducials": {**blocks["fiducials"], "gamma": 1.5}},
        lambda blocks: {"waveform": {**blocks["waveform"], "maximum_frequency": 18.0}},
        lambda blocks: {
            "population": {
                **blocks["population"],
                "model_kwargs": {
                    **blocks["population"]["model_kwargs"],
                    "n_grid": 128,
                },
            }
        },
    ],
    ids=["seed", "num_samples", "version", "fiducial", "waveform", "model_kwarg"],
)
def test_key_when_any_field_changes_changes(
    make_request: RequestFactory,
    population: dict[str, Any],
    waveform: dict[str, Any],
    fiducials: dict[str, float],
    override: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    blocks = {"population": population, "waveform": waveform, "fiducials": fiducials}

    assert make_request(**override(blocks)).key() != make_request().key()


def test_key_with_integer_or_float_kwarg_is_the_same(
    make_request: RequestFactory, population: dict[str, Any]
) -> None:
    as_float = {
        **population,
        "model_kwargs": {**population["model_kwargs"], "n_grid": 64.0},
    }

    assert make_request(population=as_float).key() == make_request().key()


def test_key_with_reordered_fiducials_is_the_same(
    make_request: RequestFactory, fiducials: dict[str, float]
) -> None:
    reordered = dict(reversed(list(fiducials.items())))

    assert make_request(fiducials=reordered).key() == make_request().key()


def test_from_blocks_with_population_seed_raises(
    make_request: RequestFactory, population: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="may not declare seed"):
        make_request(population={**population, "seed": 3})


def test_from_blocks_without_version_uses_installed_version(
    population: dict[str, Any],
    waveform: dict[str, Any],
    fiducials: dict[str, float],
) -> None:
    request = CatalogRequest.from_blocks(
        population=population,
        waveform=waveform,
        fiducials=fiducials,
        seed=7,
        num_samples=16,
    )

    assert request.version == astrogwb.__version__


def test_generate_with_foreign_version_raises(make_request: RequestFactory) -> None:
    with pytest.raises(ValueError, match="is installed"):
        generate(make_request(version="0.0.0-other"))


def test_saved_catalog_when_loaded_answers_its_request(
    make_request: RequestFactory, tmp_path: Path
) -> None:
    request = make_request()
    path = tmp_path / "catalog.h5"
    generate(request).save(path)

    restored = PolarizationPowerCatalog.load(path)

    assert CatalogRequest.from_catalog(restored).key() == request.key()


def test_load_or_generate_on_miss_writes_only_the_keyed_file(
    make_request: RequestFactory, tmp_path: Path
) -> None:
    request = make_request()

    load_or_generate(request, tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [f"{request.key()}.h5"]


def test_load_or_generate_on_hit_does_not_rewrite_the_file(
    make_request: RequestFactory, tmp_path: Path
) -> None:
    """A regeneration would replace the file, and with it the inode."""
    request = make_request()
    first = load_or_generate(request, tmp_path)
    inode = catalog_path(request, tmp_path).stat().st_ino

    second = load_or_generate(request, tmp_path)

    assert catalog_path(request, tmp_path).stat().st_ino == inode
    np.testing.assert_array_equal(first.polarization_power, second.polarization_power)


def test_load_or_generate_with_file_under_wrong_key_raises(
    make_request: RequestFactory, tmp_path: Path
) -> None:
    other = make_request(seed=8)
    generate(make_request()).save(catalog_path(other, tmp_path))

    with pytest.raises(ValueError, match="not the requested"):
        load_or_generate(other, tmp_path)


def test_check_catalog_answers_with_other_request_raises(
    make_request: RequestFactory,
) -> None:
    catalog = generate(make_request())

    with pytest.raises(ValueError, match="not the requested"):
        check_catalog_answers(catalog, make_request(num_samples=32), label="memory")
