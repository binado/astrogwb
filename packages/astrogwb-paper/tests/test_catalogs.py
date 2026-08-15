from pathlib import Path

import pytest
from astrogwb_paper.config.catalogs import load_catalog_recipe
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def test_catalog_recipe_loads_default_recipe() -> None:
    recipe = load_catalog_recipe(
        PAPER_ROOT / "inputs" / "catalogs" / "bns-n16384-df1.toml"
    )

    assert recipe.n_samples == 16384
    assert recipe.frequency_resolution == 1.0
    assert recipe.population_config == Path("examples/bns_population.yaml")


def test_catalog_recipe_rejects_missing_settings(tmp_path: Path) -> None:
    recipe = tmp_path / "incomplete.toml"
    recipe.write_text("[population]\nconfig = 'population.yaml'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid catalog recipe 'incomplete'"):
        load_catalog_recipe(recipe)
