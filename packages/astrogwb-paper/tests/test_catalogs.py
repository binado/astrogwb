"""Runtime catalog composition: truncation and the in-memory bank mixture.

The registry-validation half of this file went with ``inputs/catalogs.yaml``.
What a composition *is* now lives on
:class:`~astrogwb_paper.config.mcmc.CatalogSpec`, declared inline by each run;
what remains testable here is the composition law itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astrogwb.waveform import make_catalog, save_catalog
from astrogwb_paper.catalogs import CatalogSource, truncate_catalog_samples
from astrogwb_paper.config.mcmc import CatalogSpec
from pydantic import ValidationError


# --------------------------------------------------------------------------- #
# truncate_catalog_samples (unaffected by the registry removal)
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
# CatalogSource.compose
# --------------------------------------------------------------------------- #
def _bank_file(path: Path, n: int, *, offset: float = 0.0) -> Path:
    """A synthetic bank whose redshift encodes sample identity: offset + index."""
    redshift = offset + np.arange(n, dtype=float)
    save_catalog(path, _catalog(redshift))
    return path


def test_compose_catalog_eps0_is_a_bit_identical_bank_prefix(tmp_path: Path) -> None:
    md_path = _bank_file(tmp_path / "md.h5", 10)
    composition = CatalogSpec(md_bank="md", num_samples=4)

    composed = CatalogSource(md_path, None, composition, "catalog").compose()

    np.testing.assert_array_equal(
        composed.source_parameters.sel(parameter="redshift").values,
        [0.0, 1.0, 2.0, 3.0],
    )
    with xr.open_dataset(md_path, engine="h5netcdf") as full:
        np.testing.assert_array_equal(
            composed.polarization_power.values,
            full.polarization_power.isel(sample=slice(0, 4)).values,
        )


def test_compose_catalog_rejects_oversized_request(tmp_path: Path) -> None:
    md_path = _bank_file(tmp_path / "md.h5", 4)
    composition = CatalogSpec(md_bank="md", num_samples=8)

    with pytest.raises(
        ValueError, match="holds 4 samples, but the composition needs 8"
    ):
        CatalogSource(md_path, None, composition, "catalog").compose()


def test_compose_catalog_requires_uniform_bank_and_mixture_seed_when_mixing() -> None:
    composition = CatalogSpec.model_construct(
        md_bank="md",
        uniform_bank=None,
        num_samples=4,
        uniform_mixing_fraction=0.2,
        mixture_seed=None,
    )

    with pytest.raises(ValueError, match="requires both uniform_bank_path"):
        CatalogSource(Path("unused.h5"), None, composition, "catalog").compose()


def test_compose_catalog_mixture_component_counts_are_binomial(tmp_path: Path) -> None:
    md_path = _bank_file(tmp_path / "md.h5", 2000, offset=0.0)
    uniform_path = _bank_file(tmp_path / "uniform.h5", 2000, offset=1_000_000.0)
    composition = CatalogSpec(
        md_bank="md",
        uniform_bank="uniform",
        num_samples=1000,
        uniform_mixing_fraction=0.2,
        mixture_seed=7,
    )

    composed = CatalogSource(md_path, uniform_path, composition, "catalog").compose()

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

    def _composition(n: int) -> CatalogSpec:
        return CatalogSpec(
            md_bank="md",
            uniform_bank="uniform",
            num_samples=n,
            uniform_mixing_fraction=0.3,
            mixture_seed=11,
        )

    big = CatalogSource(md_path, uniform_path, _composition(50), "catalog").compose()
    small = CatalogSource(md_path, uniform_path, _composition(20), "catalog").compose()

    big_redshift = np.asarray(big.source_parameters.sel(parameter="redshift").values)
    small_redshift = np.asarray(
        small.source_parameters.sel(parameter="redshift").values
    )
    np.testing.assert_array_equal(big_redshift[:20], small_redshift)


def test_compose_catalog_rejects_banks_with_different_waveform_settings(
    tmp_path: Path,
) -> None:
    """The registry used to compare bank recipes; the banks now speak for themselves."""
    md_path = _bank_file(tmp_path / "md.h5", 100)
    uniform_path = tmp_path / "uniform.h5"
    other = _catalog(np.arange(100, dtype=float) + 1_000_000.0)
    other.attrs["approximant"] = "SomethingElse"
    save_catalog(uniform_path, other)
    spec = CatalogSpec(
        md_bank="md",
        uniform_bank="uniform",
        num_samples=50,
        uniform_mixing_fraction=0.2,
        mixture_seed=11,
    )

    with pytest.raises(ValueError, match="different waveform settings: approximant"):
        CatalogSource(md_path, uniform_path, spec, "proposal").compose()


# --------------------------------------------------------------------------- #
# CatalogSpec: the mixture invariants the retired registry used to enforce
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_uniform_fraction_accepts_endpoints(epsilon: float) -> None:
    kwargs = {"md_bank": "m", "num_samples": 1, "uniform_mixing_fraction": epsilon}
    if epsilon > 0.0:
        kwargs |= {"uniform_bank": "u", "mixture_seed": 3}
    assert CatalogSpec.model_validate(kwargs).uniform_mixing_fraction == epsilon


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_uniform_fraction_rejects_values_outside_unit_interval(epsilon: float) -> None:
    with pytest.raises(ValidationError):
        CatalogSpec.model_validate(
            {"md_bank": "m", "num_samples": 1, "uniform_mixing_fraction": epsilon}
        )


def test_positive_fraction_requires_uniform_bank_and_mixture_seed() -> None:
    with pytest.raises(ValidationError, match="requires both"):
        CatalogSpec.model_validate(
            {"md_bank": "m", "num_samples": 1, "uniform_mixing_fraction": 0.1}
        )


def test_zero_fraction_forbids_uniform_bank_and_mixture_seed() -> None:
    with pytest.raises(ValidationError, match="forbids"):
        CatalogSpec.model_validate(
            {
                "md_bank": "m",
                "num_samples": 1,
                "uniform_bank": "u",
                "mixture_seed": 3,
            }
        )
