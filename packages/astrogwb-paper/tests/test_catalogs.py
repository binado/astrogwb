from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from astrogwb_paper.config.catalogs import (
    CATALOGS_PATH,
    INJECTION_CATALOG_NAME,
    catalog_recipe,
    load_catalogs,
    proposal_config,
)
from astrogwb_paper.config.loading import deep_merge, load_mapping
from astrogwb_paper.paths import paper_project_root
from pydantic import ValidationError

PAPER_ROOT = paper_project_root()


def _inventory_copy() -> dict:
    return load_mapping(PAPER_ROOT / CATALOGS_PATH)


def _write_inventory(path: Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_inventory_declares_injection_and_five_proposals() -> None:
    catalogs = load_catalogs()

    assert list(catalogs) == [
        "injection-bns-n32768-eps=0-df1",
        "bns-n8192-eps=0.1-df1",
        "bns-n16384-eps=0.1-df1",
        "bns-n32768-eps=0.1-df1",
        "bns-n16384-eps=0.01-df1",
        "bns-n16384-eps=0.001-df1",
    ]
    assert INJECTION_CATALOG_NAME == "injection-bns-n32768-eps=0-df1"
    assert catalogs[INJECTION_CATALOG_NAME].population.uniform_mixing_fraction == 0
    assert catalogs["bns-n16384-eps=0.1-df1"].population.num_samples == 16384
    assert (
        catalogs["bns-n16384-eps=0.01-df1"].population.uniform_mixing_fraction == 0.01
    )
    assert all(
        recipe.waveform.frequency_resolution == 1.0 for recipe in catalogs.values()
    )


def test_population_config_paths_are_shared_by_catalogs() -> None:
    populations = [recipe.population for recipe in load_catalogs().values()]

    assert {population.base_config for population in populations} == {
        Path("inputs/populations/population.base.yaml")
    }
    assert {population.md_redshift_config for population in populations} == {
        Path("inputs/populations/population.md.yaml")
    }
    assert {population.uniform_redshift_config for population in populations} == {
        Path("inputs/populations/population.uniform-redshift.yaml")
    }


def test_population_overlays_differ_only_in_redshift() -> None:
    population = next(iter(load_catalogs().values())).population
    base = load_mapping(PAPER_ROOT / population.base_config)
    md = deep_merge(base, load_mapping(PAPER_ROOT / population.md_redshift_config))
    uniform = deep_merge(
        base, load_mapping(PAPER_ROOT / population.uniform_redshift_config)
    )

    md_parameters = dict(md["parameters"])
    uniform_parameters = dict(uniform["parameters"])
    md_parameters.pop("redshift")
    uniform_parameters.pop("redshift")
    assert md_parameters == uniform_parameters


def test_proposal_config_is_expanded_from_population_fragments() -> None:
    proposal = proposal_config(catalog_recipe("bns-n16384-eps=0.1-df1"))

    assert proposal == {
        "uniform_mixing_fraction": 0.1,
        "z_min": 0.0,
        "z_max": 20.0,
        "n_grid": 4096,
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
    }


def test_unknown_catalog_name_lists_choices() -> None:
    with pytest.raises(ValueError, match="unknown catalog 'missing'"):
        catalog_recipe("missing")


def test_catalog_inventory_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["extra"] = 1
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="unknown top-level keys: extra"):
        load_catalogs(inventory)


def test_catalog_recipe_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n8192-eps=0.1-df1"]["population"]["num_sample"] = 1
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValidationError, match="num_sample"):
        load_catalogs(inventory)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_uniform_fraction_accepts_endpoints(tmp_path: Path, epsilon: float) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n8192-eps=0.1-df1"]["population"][
        "uniform_mixing_fraction"
    ] = epsilon
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    assert (
        load_catalogs(inventory)[
            "bns-n8192-eps=0.1-df1"
        ].population.uniform_mixing_fraction
        == epsilon
    )


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_uniform_fraction_rejects_values_outside_unit_interval(
    tmp_path: Path, epsilon: float
) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n8192-eps=0.1-df1"]["population"][
        "uniform_mixing_fraction"
    ] = epsilon
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValidationError):
        load_catalogs(inventory)
