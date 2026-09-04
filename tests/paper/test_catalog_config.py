"""Catalog provenance: extraction, round-trip, and the checks that depend on it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from catalog_fixtures import make_catalog, save_catalog

from astrogwb.catalog import PopulationMetadata
from astrogwb.paper.config.catalogs import (
    MD_FIDUCIAL_NAMES,
    CatalogProvenance,
    MadauDickinsonProposal,
    MixtureProposal,
    ProposalComponent,
    UniformRedshiftProposal,
    check_fiducials_match,
    discover_catalogs,
    extract_mixture_proposal,
    extract_redshift_proposal,
    resolve_proposal,
)
from astrogwb.paper.utils import load_mapping

MD_GRAPH: dict[str, Any] = {
    "parameters": {
        "redshift": {
            "sampler": {
                "function": "madau_dickinson_redshift",
                "arguments": {
                    "z_min": 0.0,
                    "z_max": 20.0,
                    "gamma": 1.42,
                    "kappa": 4.62,
                    "z_peak": 1.84,
                    "hubble_constant": 67.66,
                    "omega_m": 0.3096,
                    "n_grid": 4096,
                },
            }
        }
    }
}
UNIFORM_GRAPH: dict[str, Any] = {
    "parameters": {
        "redshift": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 0.0, "maximum": 20.0},
            }
        }
    }
}


def _provenance(**overrides: Any) -> CatalogProvenance:
    fields: dict[str, Any] = {
        "name": "md-imrphenom-s41-n32768",
        "seed": 41,
        "num_samples": 32768,
        "redshift_proposal": extract_mixture_proposal([(MD_GRAPH, 1.0)]),
    }
    return CatalogProvenance(**{**fields, **overrides})


def _catalog_file(path: Path, metadata: PopulationMetadata | None) -> Path:
    num_samples = 3 if metadata is None else metadata.num_samples
    save_catalog(
        path,
        make_catalog(
            frequencies=np.linspace(10.0, 50.0, 4),
            polarization_power=np.ones((4, num_samples)),
            source_parameters={"redshift": np.linspace(0.5, 2.0, num_samples)},
            approximant="Toy",
            minimum_frequency=10.0,
            maximum_frequency=50.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            df=40.0 / 3.0,
            population_metadata=metadata,
        ),
    )
    return path


# --------------------------------------------------------------------------- #
# extract_redshift_proposal (write side)
# --------------------------------------------------------------------------- #
def test_extract_madau_dickinson_renames_the_cosmology_arguments() -> None:
    proposal = extract_redshift_proposal(MD_GRAPH)

    assert proposal == MadauDickinsonProposal(
        z_min=0.0,
        z_max=20.0,
        gamma=1.42,
        kappa=4.62,
        z_peak=1.84,
        H0=67.66,
        Omega_m=0.3096,
        n_grid=4096,
    )


def test_extract_uniform_reads_the_sampler_support() -> None:
    assert extract_redshift_proposal(UNIFORM_GRAPH) == UniformRedshiftProposal(
        z_min=0.0, z_max=20.0
    )


def test_extract_rejects_an_unrecognised_sampler_rather_than_defaulting() -> None:
    """The descriptor is written once and trusted forever; a guess would be silent."""
    graph = {
        "parameters": {
            "redshift": {"sampler": {"function": "power_law", "arguments": {}}}
        }
    }

    with pytest.raises(ValueError, match="unsupported redshift sampler function"):
        extract_redshift_proposal(graph)


def test_extract_rejects_a_graph_without_a_redshift_parameter() -> None:
    with pytest.raises(TypeError, match="parameters.redshift must be a mapping"):
        extract_redshift_proposal({"parameters": {"mass_1": {}}})


@pytest.mark.parametrize("population", ["madau-dickinson", "uniform-redshift"])
def test_committed_population_graphs_yield_a_proposal(population: str) -> None:
    """Every committed graph must be extractable, or its catalog cannot be built."""
    component = next(
        c
        for definition in discover_catalogs().values()
        for c in definition.components
        if c.population == population
    )
    proposal = extract_redshift_proposal(load_mapping(component.population_path()))

    assert (proposal.z_min, proposal.z_max) == (0.0, 20.0)


def test_committed_populations_agree_on_generation_support() -> None:
    """A mixed proposal needs both components drawn over the same span."""
    catalogs = discover_catalogs()
    supports = {
        component.population: extract_redshift_proposal(
            load_mapping(component.population_path())
        )
        for definition in catalogs.values()
        for component in definition.components
    }
    spans = {(p.z_min, p.z_max) for p in supports.values()}
    assert len(spans) == 1


# --------------------------------------------------------------------------- #
# CatalogProvenance.from_file (read side)
# --------------------------------------------------------------------------- #
def test_provenance_round_trips_through_a_real_catalog_file(tmp_path: Path) -> None:
    provenance = _provenance()
    path = _catalog_file(tmp_path / "bank.h5", provenance.to_population_metadata())

    assert CatalogProvenance.from_file(path) == provenance


def test_provenance_from_file_accepts_str_path(tmp_path: Path) -> None:
    provenance = _provenance()
    path = _catalog_file(tmp_path / "bank.h5", provenance.to_population_metadata())

    assert CatalogProvenance.from_file(str(path)) == provenance


def test_uniform_provenance_round_trips(tmp_path: Path) -> None:
    provenance = _provenance(
        name="uniform-imrphenom-s51-n8192",
        seed=51,
        num_samples=8192,
        redshift_proposal=extract_mixture_proposal([(UNIFORM_GRAPH, 1.0)]),
    )
    path = _catalog_file(tmp_path / "bank.h5", provenance.to_population_metadata())

    assert CatalogProvenance.from_file(path) == provenance


def test_bank_without_proposal_metadata_is_rejected_not_reparsed(
    tmp_path: Path,
) -> None:
    path = _catalog_file(tmp_path / "old.h5", None)

    with pytest.raises(ValueError, match="generated before proposal metadata"):
        CatalogProvenance.from_file(path)


def test_bank_missing_only_the_proposal_attr_is_rejected(tmp_path: Path) -> None:
    provenance = _provenance()
    metadata = PopulationMetadata(
        name=provenance.name,
        seed=provenance.seed,
        num_samples=provenance.num_samples,
    )
    path = _catalog_file(tmp_path / "partial.h5", metadata)

    with pytest.raises(ValueError, match="missing redshift_proposal"):
        CatalogProvenance.from_file(path)


def test_md_role_rejects_a_uniform_only_catalog() -> None:
    mixture = extract_mixture_proposal([(UNIFORM_GRAPH, 1.0)])

    with pytest.raises(ValueError, match="exactly one madau_dickinson component"):
        mixture.madau_dickinson(label="uniform-imrphenom-s51-n8192")


# --------------------------------------------------------------------------- #
# resolve_proposal
# --------------------------------------------------------------------------- #
def _md_only() -> MixtureProposal:
    return extract_mixture_proposal([(MD_GRAPH, 1.0)])


def _mixed(epsilon: float) -> MixtureProposal:
    return extract_mixture_proposal(
        [(MD_GRAPH, 1.0 - epsilon), (UNIFORM_GRAPH, epsilon)]
    )


def test_resolve_proposal_narrows_only_the_support() -> None:
    proposal = resolve_proposal(
        _md_only(),
        minimum_redshift=0.3,
        maximum_redshift=20.0,
        label="catalog",
    )

    assert proposal.model_dump(mode="json") == {
        "uniform_mixing_fraction": 0.0,
        "minimum_redshift": 0.3,
        "maximum_redshift": 20.0,
        "n_grid": 4096,
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
    }


def test_resolve_proposal_reads_the_mixing_fraction_off_the_descriptor() -> None:
    """The run config no longer carries eps: the catalog's own record does."""
    proposal = resolve_proposal(
        _mixed(0.1),
        minimum_redshift=0.3,
        maximum_redshift=20.0,
        label="catalog",
    )

    assert proposal.uniform_mixing_fraction == pytest.approx(0.1)


def test_a_mixture_without_a_madau_dickinson_component_is_rejected() -> None:
    mixture = extract_mixture_proposal([(UNIFORM_GRAPH, 1.0)])

    with pytest.raises(ValueError, match="exactly one madau_dickinson component"):
        resolve_proposal(
            mixture,
            minimum_redshift=0.3,
            maximum_redshift=20.0,
            label="uniform-only",
        )


def test_a_mixture_of_components_with_different_support_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagree on generation redshift support"):
        MixtureProposal(
            components=(
                ProposalComponent(
                    weight=0.9, density=extract_redshift_proposal(MD_GRAPH)
                ),
                ProposalComponent(
                    weight=0.1, density=UniformRedshiftProposal(z_min=0.0, z_max=10.0)
                ),
            )
        )


def test_mixture_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="weights must sum to 1"):
        MixtureProposal(
            components=(
                ProposalComponent(
                    weight=0.5, density=extract_redshift_proposal(MD_GRAPH)
                ),
                ProposalComponent(
                    weight=0.1, density=extract_redshift_proposal(UNIFORM_GRAPH)
                ),
            )
        )


def test_extract_mixture_proposal_normalizes_the_declared_weights() -> None:
    """A def declares raw weights; the descriptor records mixing fractions."""
    mixture = extract_mixture_proposal([(MD_GRAPH, 9.0), (UNIFORM_GRAPH, 1.0)])

    assert mixture.uniform_mixing_fraction == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("z_min", "z_max"), [(-1.0, 20.0), (0.3, 25.0), (5.0, 5.0), (6.0, 5.0)]
)
def test_resolve_proposal_rejects_a_window_outside_generation(
    z_min: float, z_max: float
) -> None:
    with pytest.raises(ValueError, match="must lie within the catalog generation"):
        resolve_proposal(
            _md_only(),
            minimum_redshift=z_min,
            maximum_redshift=z_max,
            label="catalog",
        )


# --------------------------------------------------------------------------- #
# check_fiducials_match
# --------------------------------------------------------------------------- #
def _fiducials() -> dict[str, float]:
    return {
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
        "local_merger_rate": 770.0,
    }


def test_matching_fiducials_pass() -> None:
    check_fiducials_match(_provenance(), _fiducials(), label="catalog")


@pytest.mark.parametrize("name", MD_FIDUCIAL_NAMES)
def test_a_drifted_fiducial_is_reported_by_name(name: str) -> None:
    fiducials = _fiducials() | {name: 1.0}

    with pytest.raises(
        ValueError, match=f"do not match proposal catalog catalog: {name} "
    ):
        check_fiducials_match(_provenance(), fiducials, label="catalog")


@pytest.mark.parametrize("name", MD_FIDUCIAL_NAMES)
def test_a_missing_fiducial_is_reported_rather_than_raising_keyerror(
    name: str,
) -> None:
    fiducials = _fiducials()
    del fiducials[name]

    with pytest.raises(ValueError, match=f"{name} \\(missing from"):
        check_fiducials_match(_provenance(), fiducials, label="catalog")


def test_check_fiducials_match_requires_a_madau_dickinson_component() -> None:
    provenance = _provenance(
        redshift_proposal=extract_mixture_proposal([(UNIFORM_GRAPH, 1.0)])
    )

    with pytest.raises(ValueError, match="exactly one madau_dickinson component"):
        check_fiducials_match(provenance, _fiducials(), label="catalog")
