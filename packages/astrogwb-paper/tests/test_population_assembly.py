from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from astrogwb_paper.catalogs import PROPOSAL_REDSHIFT_LOGPDF
from astrogwb_paper.cli import generate_waveform_catalog
from astrogwb_paper.cli.assemble_population import (
    assemble_population,
    guarded_proposal_logpdf,
)
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def _source(start: int, size: int = 1000) -> dict[str, np.ndarray]:
    values = np.arange(start, start + size, dtype=float)
    return {"redshift": values, "mass": values + 0.5}


def test_identity_preserves_source_rows() -> None:
    source = _source(0, size=4)

    actual, assignments = assemble_population(
        [source], operation="identity", num_samples=4, seed=1
    )

    np.testing.assert_array_equal(actual["redshift"], source["redshift"])
    np.testing.assert_array_equal(assignments, np.zeros(4, dtype=int))


def test_seeded_subsample_is_reproducible_without_replacement() -> None:
    source = _source(0, size=20)

    first, _ = assemble_population(
        [source], operation="subsample", num_samples=10, seed=4
    )
    second, _ = assemble_population(
        [source], operation="subsample", num_samples=10, seed=4
    )

    np.testing.assert_array_equal(first["redshift"], second["redshift"])
    assert len(np.unique(first["redshift"])) == 10


def test_categorical_mixture_uses_requested_guard_fraction() -> None:
    fiducial = _source(0)
    uniform = _source(10_000)

    production, assignments = assemble_population(
        [fiducial, uniform],
        operation="mixture",
        num_samples=1000,
        seed=12,
        weights=[0.8, 0.2],
    )

    assert np.mean(assignments == 1) == pytest.approx(0.2, abs=0.04)
    np.testing.assert_array_equal(
        production["redshift"] >= 10_000,
        assignments == 1,
    )
    assert len(np.unique(production["redshift"])) == 1000


def test_mixture_rejects_insufficient_source_pool() -> None:
    with pytest.raises(ValueError, match="needs .* rows"):
        assemble_population(
            [_source(0, size=1), _source(10, size=1)],
            operation="mixture",
            num_samples=10,
            seed=1,
            weights=[0.5, 0.5],
        )


def test_guarded_logpdf_is_finite_in_fiducial_tail() -> None:
    redshift = np.array([0.0, 1.0, 10.0, 20.0])
    actual = guarded_proposal_logpdf(
        redshift,
        fiducial_config=load_mapping(
            PAPER_ROOT / "inputs/populations/bns_population.yaml"
        ),
        uniform_config=load_mapping(
            PAPER_ROOT / "inputs/populations/bns_population_uniform_redshift.yaml"
        ),
        epsilon=0.2,
    )

    assert np.all(np.isfinite(actual))
    assert actual[0] == pytest.approx(np.log(0.2 / 20.0))
    assert actual[2] > np.log(0.2 / 20.0)


def test_guarded_components_may_differ_only_in_redshift() -> None:
    fiducial = load_mapping(PAPER_ROOT / "inputs/populations/bns_population.yaml")
    uniform = load_mapping(
        PAPER_ROOT / "inputs/populations/bns_population_uniform_redshift.yaml"
    )
    uniform["parameters"]["spin_1z"]["sampler"]["arguments"]["minimum"] = -0.1

    with pytest.raises(ValueError, match="may differ only in redshift"):
        guarded_proposal_logpdf(
            np.array([1.0]),
            fiducial_config=fiducial,
            uniform_config=uniform,
            epsilon=0.2,
        )


def test_waveform_generation_preserves_proposal_density(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    population = {
        "redshift": np.array([0.1, 0.2]),
        "source_frame_mass_1": np.array([1.4, 1.5]),
        "source_frame_mass_2": np.array([1.2, 1.3]),
        PROPOSAL_REDSHIFT_LOGPDF: np.array([-2.0, -2.1]),
    }
    args = argparse.Namespace(
        population=tmp_path / "population.h5",
        output=tmp_path / "catalog.h5",
        approximant="test",
        sampling_frequency=16.0,
        minimum_frequency=2.0,
        maximum_frequency=8.0,
        reference_frequency=4.0,
        frequency_resolution=1.0,
        segment_duration=None,
        chunk_size=2,
    )
    captured = {}
    monkeypatch.setattr(generate_waveform_catalog, "parse_args", lambda: args)
    monkeypatch.setattr(
        generate_waveform_catalog,
        "read_population_catalogue",
        lambda path: population,
    )
    monkeypatch.setattr(
        generate_waveform_catalog,
        "RippleBackend",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        generate_waveform_catalog,
        "_generate_polarizations",
        lambda *args, **kwargs: (
            np.array([2.0, 3.0]),
            np.ones((2, 2), dtype=complex),
            np.ones((2, 2), dtype=complex),
        ),
    )
    monkeypatch.setattr(
        generate_waveform_catalog,
        "save_catalog",
        lambda path, catalog: captured.setdefault("catalog", catalog),
    )

    generate_waveform_catalog.main()

    np.testing.assert_array_equal(
        captured["catalog"].source_parameters[PROPOSAL_REDSHIFT_LOGPDF],
        population[PROPOSAL_REDSHIFT_LOGPDF],
    )
