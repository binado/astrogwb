"""End-to-end generation: merged config blocks in, a self-describing catalog out.

The RNG properties the persisted catalogs rely on -- reproducibility and prefix
stability across sizes -- are properties of the population declaration and are
pinned in ``tests/core/test_populations.py``. What is left for this layer is
that the committed config schema actually reaches the generator, and that what
lands on disk describes itself well enough to load.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.paper.config.catalogs import CatalogDefinition
from astrogwb.populations import DEFAULT_DENSITY_SITES

#: The hyperparameters a real catalog inherits from ``config/fiducials.json``.
TOY_FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}

TOY_WAVEFORM: dict[str, object] = {
    "approximant": "TaylorF2",
    "sampling_frequency": 512.0,
    "minimum_frequency": 16.0,
    "maximum_frequency": 64.0,
    "reference_frequency": 16.0,
    "frequency_resolution": 1.0,
}


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


def test_the_script_enables_x64_before_it_draws() -> None:
    """The script must configure x64 itself, not inherit it from Ripple's import.

    ``build_catalog`` draws the population before it builds the Ripple-backed
    generator, and ``import ripplegw`` -- which turns x64 on globally -- happens
    only inside that generator. Left to that side effect, every source column is
    drawn and persisted at float32. The paper conftest enables x64 for the whole
    in-process suite, so this runs in a fresh interpreter where it starts off,
    and asserts ripplegw is still unimported: that is what proves the flag came
    from the script rather than from the import it must not depend on.
    """
    code = f"""
import importlib.util
import sys

import jax

assert not jax.config.x64_enabled, "x64 was already on before the script ran"
path = {str(REPO_ROOT / "scripts" / "generate_catalog.py")!r}
spec = importlib.util.spec_from_file_location("generate_catalog_script", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert jax.config.x64_enabled, "importing generate_catalog did not enable x64"
assert "ripplegw" not in sys.modules, "x64 came from importing ripplegw"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _blocks(
    *,
    model: str,
    extra_kwargs: Mapping[str, float] | None = None,
    extra_fiducials: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """The blocks the workflow's `jq` pass hands the generator, already merged."""
    return {
        "name": "toy-catalog",
        "seed": 41,
        "num_samples": 8,
        "population": {
            "model_name": model,
            "model_kwargs": {
                "minimum_redshift": 0.0,
                "maximum_redshift": 20.0,
                "n_grid": 256,
                **(extra_kwargs or {}),
            },
        },
        "fiducials": {**TOY_FIDUCIALS, **(extra_fiducials or {})},
        "waveform": TOY_WAVEFORM,
    }


def _definition(**kwargs) -> CatalogDefinition:
    """Validate those blocks the way ``generate_catalog.main`` does."""
    return CatalogDefinition.model_validate(_blocks(**kwargs))


@pytest.mark.integration
def test_generation_produces_a_catalog_that_describes_itself(
    generate_catalog, tmp_path: Path
) -> None:
    definition = _definition(model="bns_md_cosmological")
    catalog = generate_catalog.build_catalog(definition)

    assert catalog.num_samples == 8
    assert catalog.seed == 41
    assert catalog.population_model_name == "bns_md_cosmological"
    assert catalog.population_model_kwargs == {
        "minimum_redshift": 0.0,
        "maximum_redshift": 20.0,
        "n_grid": 256,
    }
    assert catalog.fiducials["local_merger_rate"] == 770.0
    # Generation records the default factor set; the literal pins that default.
    assert catalog.density_sites == DEFAULT_DENSITY_SITES
    assert DEFAULT_DENSITY_SITES == (
        "redshift",
        "source_frame_mass_1",
        "source_frame_mass_2",
    )
    assert catalog.polarization_power.shape[1] == 8

    # Round-tripping is the real assertion: loading re-executes the recorded
    # population and compares every derived column against the file.
    path = tmp_path / "toy-catalog.h5"
    catalog.save(path)
    restored = PolarizationPowerCatalog.load(path)
    np.testing.assert_array_equal(
        restored.polarization_power, catalog.polarization_power
    )
    assert restored.fiducials == catalog.fiducials


@pytest.mark.integration
def test_generation_is_reproducible_from_the_same_config(
    generate_catalog, tmp_path: Path
) -> None:
    definition = _definition(model="bns_md_cosmological")
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
    definition = _definition(
        model="bns_md_uniform_mixture",
        extra_kwargs={"uniform_mixing_fraction": 0.1},
    )
    catalog = generate_catalog.build_catalog(definition)

    assert catalog.population_model_name == "bns_md_uniform_mixture"
    assert catalog.population_model_kwargs["uniform_mixing_fraction"] == 0.1
    path = tmp_path / "guard.h5"
    catalog.save(path)
    assert (
        PolarizationPowerCatalog.load(path).population_model_kwargs[
            "uniform_mixing_fraction"
        ]
        == 0.1
    )


@pytest.mark.integration
def test_the_gaussian_mass_model_is_generated_from_its_declared_name(
    generate_catalog, tmp_path: Path
) -> None:
    definition = _definition(
        model="bns_md_gaussian_cosmological",
        extra_fiducials={"mass_mean": 1.33, "mass_sigma": 0.09},
    )
    catalog = generate_catalog.build_catalog(definition)

    assert catalog.population_model_name == "bns_md_gaussian_cosmological"
    assert catalog.fiducials["mass_mean"] == 1.33
    assert catalog.fiducials["mass_sigma"] == 0.09
    path = tmp_path / "gaussian.h5"
    catalog.save(path)
    assert (
        PolarizationPowerCatalog.load(path).population_model_name
        == "bns_md_gaussian_cosmological"
    )


def test_an_unregistered_model_fails_before_any_waveform_is_generated(
    generate_catalog,
) -> None:
    definition = _definition(model="no_such_population")
    with pytest.raises(ValueError, match="bns_md_cosmological"):
        generate_catalog.build_catalog(definition)


@pytest.mark.integration
def test_the_cli_takes_the_merged_blocks_on_argv(
    generate_catalog, tmp_path: Path
) -> None:
    """The workflow merges with `jq` and passes blocks, not paths.

    This is the surface `rule waveform_catalog` actually drives, so it is
    pinned here rather than left to the Snakefile's shell string alone.
    """
    blocks = _blocks(model="bns_md_cosmological")
    output = tmp_path / "cli.h5"
    generate_catalog.main(
        [
            "--name",
            "toy-catalog",
            "--population",
            json.dumps(blocks["population"]),
            "--fiducials",
            json.dumps(blocks["fiducials"]),
            "--waveform",
            json.dumps(blocks["waveform"]),
            "--seed",
            "41",
            "--num-samples",
            "8",
            "--output",
            str(output),
        ]
    )

    catalog = PolarizationPowerCatalog.load(output)
    assert catalog.seed == 41
    assert catalog.num_samples == 8
    assert catalog.population_model_name == "bns_md_cosmological"
    assert catalog.fiducials["local_merger_rate"] == 770.0


def test_the_cli_rejects_a_block_that_is_not_a_json_object(generate_catalog) -> None:
    """A `jq` filter that selects a missing key yields `null`, not an object."""
    with pytest.raises(SystemExit):
        generate_catalog.parse_args(
            [
                "--name",
                "toy",
                "--population",
                "null",
                "--fiducials",
                "{}",
                "--waveform",
                "{}",
                "--seed",
                "1",
                "--num-samples",
                "1",
                "--output",
                "out.h5",
            ]
        )
