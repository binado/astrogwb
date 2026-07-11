from pathlib import Path

import pytest

from astrogwb.config.catalogs import catalog_path, load_catalog_recipe, population_path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_catalog_recipe_loads_default_recipe() -> None:
    recipe = load_catalog_recipe(
        REPO_ROOT / "configs" / "catalogs" / "bns-n16384-df1.toml"
    )

    assert recipe.n_samples == 16384
    assert recipe.frequency_resolution == 1.0
    assert population_path(recipe.catalog_id) == Path(
        "out/populations/bns-n16384-df1.h5"
    )
    assert catalog_path(recipe.catalog_id) == Path("out/catalogs/bns-n16384-df1.h5")


def test_catalog_recipe_rejects_invalid_filename(tmp_path: Path) -> None:
    recipe = tmp_path / "bad id.toml"
    recipe.write_text("[population]\n[waveform]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid catalog id"):
        load_catalog_recipe(recipe)


def test_catalog_recipe_rejects_missing_settings(tmp_path: Path) -> None:
    recipe = tmp_path / "incomplete.toml"
    recipe.write_text("[population]\nconfig = 'population.yaml'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid catalog recipe 'incomplete'"):
        load_catalog_recipe(recipe)
