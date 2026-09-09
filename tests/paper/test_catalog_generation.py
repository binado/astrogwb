"""End-to-end generation: config layers in, a self-describing catalog out.

The RNG properties the persisted catalogs rely on -- reproducibility and prefix
stability across sizes -- are properties of the population declaration and are
pinned in ``tests/core/test_populations.py``. What is left for this layer is
that the committed config schema actually reaches the generator, and that what
lands on disk describes itself well enough to load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.catalog import Catalog

CATALOG_TOML = """
num_samples = 8
seed = 41

[population]
model = "{model}"

[population.kwargs]
z_min = 0.0
z_max = 20.0
n_grid = 256
{extra_kwargs}

[population.params]
H0 = 67.66
Omega_m = 0.3096
gamma = 1.42
kappa = 4.62
z_peak = 1.84
local_merger_rate = 770.0

[waveform]
approximant = "TaylorF2"
sampling_frequency = 512.0
minimum_frequency = 16.0
maximum_frequency = 64.0
reference_frequency = 16.0
frequency_resolution = 1.0
chunk_size = 8
"""


@pytest.fixture(scope="module")
def generate_catalog():
    """Import ``scripts/generate_catalog.py``, which is not an installed module."""
    path = REPO_ROOT / "scripts" / "generate_catalog.py"
    spec = importlib.util.spec_from_file_location("generate_catalog_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path, *, model: str, extra_kwargs: str = "") -> Path:
    path = tmp_path / "toy-catalog.toml"
    path.write_text(
        CATALOG_TOML.format(model=model, extra_kwargs=extra_kwargs), encoding="utf-8"
    )
    return path


@pytest.mark.integration
def test_generation_produces_a_catalog_that_describes_itself(
    generate_catalog, tmp_path: Path
) -> None:
    definition = generate_catalog.load_catalog_layers(
        [_config(tmp_path, model="bns_md_cosmological")]
    )
    catalog = generate_catalog.build_catalog(definition)

    assert catalog.num_samples == 8
    assert catalog.seed == 41
    assert catalog.population_model_name == "bns_md_cosmological"
    assert catalog.population_model_kwargs == {
        "z_min": 0.0,
        "z_max": 20.0,
        "n_grid": 256,
    }
    assert catalog.fiducials["local_merger_rate"] == 770.0
    assert catalog.density_sites == ("redshift",)
    assert catalog.polarization_power.shape[1] == 8

    # Round-tripping is the real assertion: loading re-executes the recorded
    # population and compares every derived column against the file.
    path = tmp_path / "toy-catalog.h5"
    catalog.save(path)
    restored = Catalog.load(path)
    np.testing.assert_array_equal(
        restored.polarization_power, catalog.polarization_power
    )
    assert restored.fiducials == catalog.fiducials


@pytest.mark.integration
def test_generation_is_reproducible_from_the_same_config(
    generate_catalog, tmp_path: Path
) -> None:
    definition = generate_catalog.load_catalog_layers(
        [_config(tmp_path, model="bns_md_cosmological")]
    )
    first = generate_catalog.build_catalog(definition)
    second = generate_catalog.build_catalog(definition)

    for name, values in first.source_parameters.items():
        np.testing.assert_array_equal(values, second.source_parameters[name])
    np.testing.assert_array_equal(first.polarization_power, second.polarization_power)


@pytest.mark.integration
def test_the_guard_mixture_is_generated_from_its_declared_fraction(
    generate_catalog, tmp_path: Path
) -> None:
    """The eps in the config is the eps the file records and reweights by."""
    definition = generate_catalog.load_catalog_layers(
        [
            _config(
                tmp_path,
                model="bns_md_uniform_mixture",
                extra_kwargs="uniform_mixing_fraction = 0.1",
            )
        ]
    )
    catalog = generate_catalog.build_catalog(definition)

    assert catalog.population_model_name == "bns_md_uniform_mixture"
    assert catalog.population_model_kwargs["uniform_mixing_fraction"] == 0.1
    path = tmp_path / "guard.h5"
    catalog.save(path)
    assert Catalog.load(path).population_model_kwargs["uniform_mixing_fraction"] == 0.1


def test_an_unregistered_model_fails_before_any_waveform_is_generated(
    generate_catalog, tmp_path: Path
) -> None:
    definition = generate_catalog.load_catalog_layers(
        [_config(tmp_path, model="no_such_population")]
    )
    with pytest.raises(ValueError, match="bns_md_cosmological"):
        generate_catalog.build_catalog(definition)
