from pathlib import Path

import pytest

from astrogwb.catalogs import catalog_path, load_catalog_recipes, population_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_catalog_registry_loads_default_recipe() -> None:
    recipes = load_catalog_recipes(REPO_ROOT / "configs" / "catalogs.toml")

    recipe = recipes["bns-n16384-df1"]
    assert recipe.n_samples == 16384
    assert recipe.frequency_resolution == 1.0
    assert population_path(recipe.catalog_id) == Path("out/populations/bns-n16384-df1.h5")
    assert catalog_path(recipe.catalog_id) == Path("out/catalogs/bns-n16384-df1.h5")
    assert "bns-n8192-df1" in recipes


def test_catalog_registry_rejects_path_like_id(tmp_path: Path) -> None:
    registry = tmp_path / "catalogs.toml"
    registry.write_text("[catalogs.'bad/id']\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid catalog id"):
        load_catalog_recipes(registry)
