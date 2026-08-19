from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from astrogwb_paper.cli.assemble_population import parse_args
from astrogwb_paper.config.catalogs import (
    ASSEMBLY_OPERATIONS,
    CATALOGS_PATH,
    INJECTION_CATALOG_NAME,
    catalog_recipe,
    injection_recipe,
    load_catalog_inventory,
    load_catalogs,
)
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def _inventory_copy() -> dict:
    return load_mapping(PAPER_ROOT / CATALOGS_PATH)


def _write_inventory(path: Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_single_yaml_is_the_only_catalog_inventory() -> None:
    assert (PAPER_ROOT / CATALOGS_PATH).is_file()
    catalogs_dir = PAPER_ROOT / "inputs" / "catalogs"
    assert not catalogs_dir.exists()


def test_inventory_declares_source_pools_injection_and_five_proposals() -> None:
    inventory = load_catalog_inventory()

    assert set(inventory.sources) == {
        "injection-fiducial",
        "proposal-fiducial",
        "proposal-uniform-redshift",
    }
    assert (
        inventory.sources["injection-fiducial"].seed
        != inventory.sources["proposal-fiducial"].seed
    )
    assert inventory.injection.name == "injection"
    assert INJECTION_CATALOG_NAME == "injection-bns-n32768"
    assert inventory.injection.production.operation == "identity"
    assert inventory.injection.production.num_samples == 32768
    assert inventory.injection.production.uniform_redshift_fraction is None

    assert set(inventory.catalogs) == {
        "bns-n8192-df1",
        "bns-n16384-df1",
        "bns-n32768-df1",
        "bns-n16384-eps1e-2-df1",
        "bns-n16384-eps1e-3-df1",
    }
    assert {
        name: recipe.production.num_samples
        for name, recipe in inventory.catalogs.items()
    } == {
        "bns-n8192-df1": 8192,
        "bns-n16384-df1": 16384,
        "bns-n32768-df1": 32768,
        "bns-n16384-eps1e-2-df1": 16384,
        "bns-n16384-eps1e-3-df1": 16384,
    }
    for recipe in inventory.catalogs.values():
        assert recipe.production.operation == "mixture"
        assert recipe.waveform.frequency_resolution == 1.0
        assert recipe.waveform.approximant == "IMRPhenomXAS_NRTidalv3"

    assert (
        inventory.catalogs["bns-n16384-df1"].production.uniform_redshift_fraction == 0.1
    )
    assert [
        c.weight for c in inventory.catalogs["bns-n16384-df1"].production.components
    ] == [0.9, 0.1]
    assert (
        inventory.catalogs[
            "bns-n16384-eps1e-2-df1"
        ].production.uniform_redshift_fraction
        == 0.01
    )
    assert [
        c.weight
        for c in inventory.catalogs["bns-n16384-eps1e-2-df1"].production.components
    ] == [0.99, 0.01]
    assert (
        inventory.catalogs[
            "bns-n16384-eps1e-3-df1"
        ].production.uniform_redshift_fraction
        == 0.001
    )
    assert [
        c.weight
        for c in inventory.catalogs["bns-n16384-eps1e-3-df1"].production.components
    ] == [0.999, 0.001]


def test_catalog_accessors_return_named_recipes() -> None:
    assert catalog_recipe("bns-n16384-df1").production.num_samples == 16384
    assert injection_recipe().production.num_samples == 32768
    assert load_catalogs()["bns-n8192-df1"].name == "bns-n8192-df1"


def test_unknown_catalog_name_lists_choices() -> None:
    with pytest.raises(ValueError, match="unknown catalog 'missing'"):
        catalog_recipe("missing")


def test_catalog_inventory_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["extra"] = 1
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(ValueError, match="unknown top-level keys: extra"):
        load_catalog_inventory(inventory)


def test_proposal_recipe_requires_explicit_uniform_fraction(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["base"]["production"]["uniform_redshift_fraction"] = None
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(
        ValueError,
        match="bns-n8192-df1 production must define uniform_redshift_fraction",
    ):
        load_catalog_inventory(inventory)


def test_proposal_weights_must_match_uniform_fraction(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n8192-df1"]["production"]["uniform_redshift_fraction"] = 0.2
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(
        ValueError,
        match="component weights must match uniform_redshift_fraction",
    ):
        load_catalog_inventory(inventory)


def test_inventory_rejects_an_unsupported_assembly_operation(tmp_path: Path) -> None:
    raw = _inventory_copy()
    raw["catalogs"]["bns-n8192-df1"]["production"]["operation"] = "interleave"
    inventory = tmp_path / "catalogs.yaml"
    _write_inventory(inventory, raw)

    with pytest.raises(
        ValueError,
        match="unsupported production operation 'interleave'",
    ):
        load_catalog_inventory(inventory)


def test_cli_accepts_exactly_the_declared_assembly_operations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The CLI derives --operation choices from ASSEMBLY_OPERATIONS, so the
    # recipe validator and the command line can never drift apart.
    for operation in ASSEMBLY_OPERATIONS:
        args = parse_args(
            [
                "--operation",
                operation,
                "--source",
                "a.h5",
                "--num-samples",
                "4",
                "--seed",
                "1",
                "--output",
                "out.h5",
            ]
        )
        assert args.operation == operation

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--operation",
                "interleave",
                "--source",
                "a.h5",
                "--num-samples",
                "4",
                "--seed",
                "1",
                "--output",
                "out.h5",
            ]
        )
    assert "interleave" in capsys.readouterr().err
