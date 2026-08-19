from pathlib import Path

import pytest
from astrogwb_paper.config.catalogs import (
    CATALOGS_PATH,
    catalog_recipe,
    load_catalogs,
    load_inventory,
)
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def test_single_yaml_is_the_only_catalog_inventory() -> None:
    assert (PAPER_ROOT / CATALOGS_PATH).is_file()
    catalogs_dir = PAPER_ROOT / "inputs" / "catalogs"
    assert not catalogs_dir.exists()


def test_catalog_inventory_loads_three_shared_recipes() -> None:
    catalogs = load_catalogs()

    assert set(catalogs) == {"bns-n8192-df1", "bns-n16384-df1", "bns-n32768-df1"}
    assert catalog_recipe("bns-n16384-df1").num_samples == 16384
    assert {name: recipe.num_samples for name, recipe in catalogs.items()} == {
        "bns-n8192-df1": 8192,
        "bns-n16384-df1": 16384,
        "bns-n32768-df1": 32768,
    }
    for recipe in catalogs.values():
        assert recipe.frequency_resolution == 1.0
        assert recipe.population_config == Path("examples/bns_population.yaml")
        assert recipe.approximant == "IMRPhenomXAS_NRTidalv3"


def test_unknown_catalog_name_lists_choices() -> None:
    with pytest.raises(ValueError, match="unknown catalog 'missing'"):
        catalog_recipe("missing")


def test_catalog_inventory_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    inventory = tmp_path / "catalogs.yaml"
    inventory.write_text(
        "base: {seed: 1}\ncatalogs: {a: {}}\nextra: 1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown top-level keys: extra"):
        load_inventory(inventory)


def test_catalog_recipe_rejects_missing_settings(tmp_path: Path) -> None:
    inventory = tmp_path / "catalogs.yaml"
    inventory.write_text(
        "base:\n  population:\n    config: population.yaml\ncatalogs:\n  incomplete: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid catalog recipe 'incomplete'"):
        load_catalogs(inventory)
