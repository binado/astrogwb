"""End-to-end generation: a resolved request in, a self-describing catalog out.

The RNG properties the persisted catalogs rely on -- reproducibility and prefix
stability across sizes -- are properties of the population declaration and are
pinned in ``tests/core/test_populations.py``; the key and the cache are pinned
in ``tests/core/test_catalog_cache.py``. What is left for this layer is that
the Ripple-backed production path generates what a request asks for, and that
the workflow's entrypoint refuses to file a draw under another's key.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.catalog import PolarizationPowerCatalog, generate
from astrogwb.metadata import CatalogRequest

RequestFactory = Callable[..., CatalogRequest]


@pytest.fixture(scope="module")
def generate_catalog() -> ModuleType:
    """Import ``scripts/generate_catalog.py``, which is not an installed module."""
    path = REPO_ROOT / "scripts" / "generate_catalog.py"
    spec = importlib.util.spec_from_file_location("generate_catalog_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_request() -> RequestFactory:
    """A tiny Ripple-backed request, with the hyperparameters a real one inherits."""

    def build(
        model: str = "bns_md_cosmological",
        *,
        extra_kwargs: dict[str, Any] | None = None,
        extra_fiducials: dict[str, float] | None = None,
    ) -> CatalogRequest:
        return CatalogRequest.from_blocks(
            population={
                "model_name": model,
                "model_kwargs": {
                    "minimum_redshift": 0.0,
                    "maximum_redshift": 20.0,
                    "n_grid": 256,
                    **(extra_kwargs or {}),
                },
            },
            waveform={
                "approximant": "TaylorF2",
                "sampling_frequency": 512.0,
                "minimum_frequency": 16.0,
                "maximum_frequency": 64.0,
                "reference_frequency": 16.0,
                "frequency_resolution": 1.0,
            },
            fiducials={
                "H0": 67.66,
                "Omega_m": 0.3096,
                "gamma": 1.42,
                "kappa": 4.62,
                "z_peak": 1.84,
                "local_merger_rate": 770.0,
                "minimum_mass": 1.0,
                "mass_width": 1.5,
                **(extra_fiducials or {}),
            },
            seed=41,
            num_samples=8,
        )

    return build


@pytest.mark.integration
def test_generate_with_ripple_request_records_the_request(
    make_request: RequestFactory, tmp_path: Path
) -> None:
    request = make_request()
    path = tmp_path / "toy.h5"
    generate(request).save(path)

    restored = PolarizationPowerCatalog.load(path)

    assert CatalogRequest.from_catalog(restored).key() == request.key()


@pytest.mark.integration
def test_generate_with_same_request_is_reproducible(
    make_request: RequestFactory,
) -> None:
    first = generate(make_request())
    second = generate(make_request())

    np.testing.assert_array_equal(first.polarization_power, second.polarization_power)


@pytest.mark.integration
def test_generate_with_guard_mixture_records_its_fraction(
    make_request: RequestFactory,
) -> None:
    """The eps in the config is the eps the file records and reweights by."""
    catalog = generate(
        make_request(
            "bns_md_uniform_mixture", extra_kwargs={"uniform_mixing_fraction": 0.1}
        )
    )

    assert catalog.population_model_kwargs["uniform_mixing_fraction"] == 0.1


@pytest.mark.integration
def test_generate_with_gaussian_mass_model_records_its_fiducials(
    make_request: RequestFactory,
) -> None:
    catalog = generate(
        make_request(
            "bns_md_gaussian_cosmological",
            extra_fiducials={"mass_mean": 1.33, "mass_sigma": 0.09},
        )
    )

    assert catalog.population_model_name == "bns_md_gaussian_cosmological"
    assert catalog.fiducials["mass_mean"] == pytest.approx(1.33)


def test_cli_with_unregistered_model_fails_before_generating(
    generate_catalog: ModuleType, make_request: RequestFactory, tmp_path: Path
) -> None:
    request = make_request("no_such_population")

    with pytest.raises(ValueError, match="bns_md_cosmological"):
        generate_catalog.main(
            [
                "--request",
                request.model_dump_json(),
                "--output",
                str(tmp_path / f"{request.key()}.h5"),
            ]
        )


def test_cli_with_output_not_named_by_key_raises(
    generate_catalog: ModuleType, make_request: RequestFactory, tmp_path: Path
) -> None:
    """The workflow names the file and the request separately; they must agree."""
    with pytest.raises(ValueError, match="not named by the request's key"):
        generate_catalog.main(
            [
                "--request",
                make_request().model_dump_json(),
                "--output",
                str(tmp_path / "md-imrphenom-s41-n32768.h5"),
            ]
        )


def test_cli_with_invalid_request_exits(generate_catalog: ModuleType) -> None:
    with pytest.raises(SystemExit):
        generate_catalog.parse_args(["--request", "null", "--output", "out.h5"])


@pytest.mark.integration
def test_cli_writes_the_catalog_under_its_key(
    generate_catalog: ModuleType, make_request: RequestFactory, tmp_path: Path
) -> None:
    """The surface `rule waveform_catalog` drives, end to end."""
    request = make_request()
    output = tmp_path / f"{request.key()}.h5"

    generate_catalog.main(
        ["--request", request.model_dump_json(), "--output", str(output)]
    )

    restored = PolarizationPowerCatalog.load(output)
    assert CatalogRequest.from_catalog(restored).key() == request.key()
