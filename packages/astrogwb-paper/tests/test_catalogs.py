from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from astrogwb.waveform import make_catalog, save_catalog
from astrogwb_paper.catalogs import compose_catalog, truncate_catalog_samples
from astrogwb_paper.config.catalogs import (
    CATALOGS_PATH,
    INJECTION_CATALOG_NAME,
    CatalogComposition,
    analysis_proposal_config,
    bank_recipe,
    catalog_recipe,
    generation_redshift_support,
    load_banks,
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


# --------------------------------------------------------------------------- #
# Inventory loading
# --------------------------------------------------------------------------- #
def test_inventory_declares_four_banks_and_eight_catalogs() -> None:
    banks = load_banks()
    catalogs = load_catalogs()

    assert list(banks) == [
        "md-imrphenom-s41",
        "md-imrphenom-s42",
        "md-taylorf2-s41",
        "uniform-imrphenom-s51",
    ]
    assert list(catalogs) == [
        "injection-bns-n32768-eps=0-df1",
        "bns-n32768-eps=0-df1-taylorf2",
        "bns-n8192-eps=0-df1",
        "bns-n16384-eps=0-df1",
        "bns-n32768-eps=0-df1",
        "bns-n16384-eps=0.1-df1",
        "bns-n16384-eps=0.01-df1",
        "bns-n16384-eps=0.001-df1",
    ]
    assert INJECTION_CATALOG_NAME == "injection-bns-n32768-eps=0-df1"
    assert catalogs[INJECTION_CATALOG_NAME].uniform_mixing_fraction == 0
    assert catalogs[INJECTION_CATALOG_NAME].md_bank == "md-imrphenom-s41"
    assert catalogs["bns-n16384-eps=0-df1"].num_samples == 16384
    assert catalogs["bns-n16384-eps=0-df1"].uniform_mixing_fraction == 0
    assert catalogs["bns-n16384-eps=0.01-df1"].uniform_mixing_fraction == 0.01
    assert catalogs["bns-n16384-eps=0.01-df1"].uniform_bank == "uniform-imrphenom-s51"
    assert all(bank.waveform.frequency_resolution == 1.0 for bank in banks.values())


def test_taylorf2_bank_only_changes_the_fiducial_waveform_approximant() -> None:
    banks = load_banks()
    imrphenom = banks["md-imrphenom-s41"].model_dump()
    taylorf2 = banks["md-taylorf2-s41"].model_dump()

    assert imrphenom["waveform"]["approximant"] == "IMRPhenomXAS_NRTidalv3"
    assert taylorf2["waveform"]["approximant"] == "TaylorF2"
    taylorf2["name"] = imrphenom["name"]
    taylorf2["waveform"]["approximant"] = imrphenom["waveform"]["approximant"]
    assert taylorf2 == imrphenom


def test_population_config_paths_are_shared_by_banks() -> None:
    populations = [bank.population for bank in load_banks().values()]

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
    population = next(iter(load_banks().values())).population
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


def test_proposal_config_is_expanded_from_the_md_bank() -> None:
    proposal = proposal_config(catalog_recipe("bns-n16384-eps=0.1-df1"))

    assert proposal == {
        "uniform_mixing_fraction": 0.1,
        "minimum_redshift": 0.0,
        "maximum_redshift": 20.0,
        "n_grid": 4096,
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
    }


def test_analysis_proposal_config_replaces_only_the_support() -> None:
    composition = catalog_recipe("bns-n16384-eps=0.1-df1")
    generation = proposal_config(composition)

    analysis = analysis_proposal_config(
        composition, minimum_redshift=0.3, maximum_redshift=20.0
    )

    assert analysis == {**generation, "minimum_redshift": 0.3, "maximum_redshift": 20.0}
    assert generation_redshift_support(composition) == (0.0, 20.0)


def test_unknown_catalog_name_lists_choices() -> None:
    with pytest.raises(ValueError, match="unknown catalog 'missing'"):
        catalog_recipe("missing")


def test_unknown_bank_name_lists_choices() -> None:
    with pytest.raises(ValueError, match="unknown bank 'missing'"):
        bank_recipe("missing")


def test_catalog_inventory_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["extra"] = 1
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="unknown top-level keys: extra"):
        load_catalogs(inventory)


def test_bank_recipe_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["banks"]["md-imrphenom-s42"]["num_sample"] = 1
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValidationError, match="num_sample"):
        load_banks(inventory)


def test_composition_naming_an_unknown_bank_is_rejected(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n16384-eps=0-df1"]["md_bank"] = "does-not-exist"
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="names unknown bank 'does-not-exist'"):
        load_catalogs(inventory)


def test_composition_requesting_more_than_the_bank_holds_is_rejected(
    tmp_path: Path,
) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n8192-eps=0-df1"]["num_samples"] = 10_000_000
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="holds only"):
        load_catalogs(inventory)


def test_composition_mixing_banks_with_different_waveforms_is_rejected(
    tmp_path: Path,
) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n16384-eps=0.1-df1"]["md_bank"] = "md-taylorf2-s41"
    raw["catalogs"]["bns-n16384-eps=0.1-df1"]["mixture_seed"] = 99
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="different waveform settings"):
        load_catalogs(inventory)


def test_composition_with_colliding_seeds_is_rejected(tmp_path: Path) -> None:
    raw = _inventory_copy()
    # md-imrphenom-s42 has seed 42; collide the mixture_seed with it.
    raw["catalogs"]["bns-n16384-eps=0.1-df1"]["mixture_seed"] = 42
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="must all be distinct"):
        load_catalogs(inventory)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_uniform_fraction_accepts_endpoints(epsilon: float) -> None:
    kwargs = {
        "name": "t",
        "md_bank": "m",
        "num_samples": 1,
        "uniform_mixing_fraction": epsilon,
    }
    if epsilon > 0.0:
        kwargs |= {"uniform_bank": "u", "mixture_seed": 3}
    assert CatalogComposition.model_validate(kwargs).uniform_mixing_fraction == epsilon


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_uniform_fraction_rejects_values_outside_unit_interval(epsilon: float) -> None:
    with pytest.raises(ValidationError):
        CatalogComposition.model_validate(
            {
                "name": "t",
                "md_bank": "m",
                "num_samples": 1,
                "uniform_mixing_fraction": epsilon,
            }
        )


def test_positive_fraction_requires_uniform_bank_and_mixture_seed() -> None:
    with pytest.raises(ValidationError, match="requires both"):
        CatalogComposition.model_validate(
            {
                "name": "t",
                "md_bank": "m",
                "num_samples": 1,
                "uniform_mixing_fraction": 0.1,
            }
        )


def test_zero_fraction_forbids_uniform_bank_and_mixture_seed() -> None:
    with pytest.raises(ValidationError, match="forbids"):
        CatalogComposition.model_validate(
            {
                "name": "t",
                "md_bank": "m",
                "num_samples": 1,
                "uniform_bank": "u",
                "mixture_seed": 3,
            }
        )


# --------------------------------------------------------------------------- #
# truncate_catalog_samples (unaffected by the bank/composition split)
# --------------------------------------------------------------------------- #
def _catalog(redshift: np.ndarray, *, offset: float = 0.0) -> xr.Dataset:
    return make_catalog(
        frequencies=np.linspace(10.0, 50.0, 5),
        polarization_power=np.arange(5 * redshift.size, dtype=float).reshape(
            5, redshift.size
        )
        + offset,
        source_parameters={
            "redshift": redshift,
            "luminosity_distance": 1.0e3 * (1.0 + redshift),
        },
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )


def test_truncate_catalog_samples_keeps_only_window_samples() -> None:
    catalog = _catalog(np.array([0.05, 0.25, 0.4, 1.5, 21.0]))

    truncated = truncate_catalog_samples(
        catalog, label="proposal", minimum_redshift=0.3, maximum_redshift=20.0
    )

    kept = truncated.source_parameters.sel(parameter="redshift").values
    np.testing.assert_allclose(kept, [0.4, 1.5])
    # Rows stay consistent across every variable sharing the sample dim.
    assert truncated.polarization_power.shape == (5, 2)
    np.testing.assert_allclose(
        truncated.polarization_power.values,
        catalog.polarization_power.isel(sample=[2, 3]).values,
    )


def test_truncate_catalog_samples_rejects_empty_window() -> None:
    catalog = _catalog(np.array([0.05, 0.25]))

    with pytest.raises(ValueError, match="no samples in the analysis redshift window"):
        truncate_catalog_samples(
            catalog, label="injection", minimum_redshift=0.3, maximum_redshift=20.0
        )


def test_truncate_catalog_samples_requires_distance_column() -> None:
    redshift = np.array([0.4, 1.5])
    catalog = make_catalog(
        frequencies=np.linspace(10.0, 50.0, 5),
        polarization_power=np.ones((5, redshift.size)),
        source_parameters={"redshift": redshift},
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )

    with pytest.raises(ValueError, match="missing required parameter"):
        truncate_catalog_samples(
            catalog, label="injection", minimum_redshift=0.3, maximum_redshift=20.0
        )


# --------------------------------------------------------------------------- #
# compose_catalog
# --------------------------------------------------------------------------- #
def _bank_file(path: Path, n: int, *, offset: float = 0.0) -> Path:
    """A synthetic bank whose redshift encodes sample identity: offset + index."""
    redshift = offset + np.arange(n, dtype=float)
    save_catalog(path, _catalog(redshift))
    return path


def test_compose_catalog_eps0_is_a_bit_identical_bank_prefix(tmp_path: Path) -> None:
    md_path = _bank_file(tmp_path / "md.h5", 10)
    composition = CatalogComposition(name="t", md_bank="md", num_samples=4)

    composed = compose_catalog(md_path, None, composition)

    np.testing.assert_array_equal(
        composed.source_parameters.sel(parameter="redshift").values,
        [0.0, 1.0, 2.0, 3.0],
    )
    full = xr.open_dataset(md_path, engine="h5netcdf")
    np.testing.assert_array_equal(
        composed.polarization_power.values,
        full.polarization_power.isel(sample=slice(0, 4)).values,
    )


def test_compose_catalog_rejects_oversized_request(tmp_path: Path) -> None:
    md_path = _bank_file(tmp_path / "md.h5", 4)
    composition = CatalogComposition(name="t", md_bank="md", num_samples=8)

    with pytest.raises(
        ValueError, match="holds 4 samples, but the composition needs 8"
    ):
        compose_catalog(md_path, None, composition)


def test_compose_catalog_requires_uniform_bank_and_mixture_seed_when_mixing() -> None:
    composition = CatalogComposition.model_construct(
        name="t",
        md_bank="md",
        uniform_bank=None,
        num_samples=4,
        uniform_mixing_fraction=0.2,
        mixture_seed=None,
    )

    with pytest.raises(ValueError, match="requires both uniform_bank_path"):
        compose_catalog(Path("unused.h5"), None, composition)


def test_compose_catalog_mixture_component_counts_are_binomial(tmp_path: Path) -> None:
    md_path = _bank_file(tmp_path / "md.h5", 2000, offset=0.0)
    uniform_path = _bank_file(tmp_path / "uniform.h5", 2000, offset=1_000_000.0)
    composition = CatalogComposition(
        name="t",
        md_bank="md",
        uniform_bank="uniform",
        num_samples=1000,
        uniform_mixing_fraction=0.2,
        mixture_seed=7,
    )

    composed = compose_catalog(md_path, uniform_path, composition)

    assert composed.sizes["sample"] == 1000
    redshift = np.asarray(composed.source_parameters.sel(parameter="redshift").values)
    n_uniform = int(np.sum(redshift >= 1_000_000.0))
    # Binomial(1000, 0.2): mean 200, std ~12.6 -- generous tolerance for a fixed seed.
    assert 130 < n_uniform < 270


def test_compose_catalog_prefix_is_itself_a_valid_mixture_sample(
    tmp_path: Path,
) -> None:
    """A prefix of a composed catalog reproduces the smaller composition exactly.

    This is what makes `num_samples`, like `uniform_mixing_fraction`, a free
    composition parameter with no extra generation cost: the RNG stream is
    the same regardless of how many samples are ultimately requested.
    """
    md_path = _bank_file(tmp_path / "md.h5", 200, offset=0.0)
    uniform_path = _bank_file(tmp_path / "uniform.h5", 200, offset=1_000_000.0)

    def _composition(n: int) -> CatalogComposition:
        return CatalogComposition(
            name=f"t{n}",
            md_bank="md",
            uniform_bank="uniform",
            num_samples=n,
            uniform_mixing_fraction=0.3,
            mixture_seed=11,
        )

    big = compose_catalog(md_path, uniform_path, _composition(50))
    small = compose_catalog(md_path, uniform_path, _composition(20))

    big_redshift = np.asarray(big.source_parameters.sel(parameter="redshift").values)
    small_redshift = np.asarray(
        small.source_parameters.sel(parameter="redshift").values
    )
    np.testing.assert_array_equal(big_redshift[:20], small_redshift)
