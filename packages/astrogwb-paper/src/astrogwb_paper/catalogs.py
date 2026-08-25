"""Shared loading and validation for injection and proposal waveform catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb.waveform import apply_gw_distance_to_power, load_catalog, open_catalog
from numpy.typing import ArrayLike

from astrogwb_paper.config.catalogs import CatalogComposition
from astrogwb_paper.config.mcmc import ProposalConfig


@dataclass(frozen=True)
class CatalogSource:
    """Bank file location(s) plus the composition to draw them into a catalog."""

    md_bank_path: Path
    uniform_bank_path: Path | None
    composition: CatalogComposition

    def compose(self) -> xr.Dataset:
        return compose_catalog(
            self.md_bank_path, self.uniform_bank_path, self.composition
        )


def catalog_source(
    composition: CatalogComposition, bank_paths: Mapping[str, Path]
) -> CatalogSource:
    """Resolve a composition's bank names against supplied bank file paths.

    Used by the CLI entrypoints, which receive banks as a flat ``NAME=PATH``
    mapping (from repeated ``--bank`` flags) and must match them against the
    bank names a run config's :class:`~astrogwb_paper.config.mcmc.CatalogConfig`
    names for each role.
    """
    md_bank_path = _require_bank_path(composition.md_bank, bank_paths)
    uniform_bank_path = (
        _require_bank_path(composition.uniform_bank, bank_paths)
        if composition.uniform_bank is not None
        else None
    )
    return CatalogSource(md_bank_path, uniform_bank_path, composition)


def _require_bank_path(name: str, bank_paths: Mapping[str, Path]) -> Path:
    try:
        return bank_paths[name]
    except KeyError:
        raise ValueError(
            f"bank {name!r} is required but was not supplied via --bank"
        ) from None


def compose_catalog(
    md_bank_path: Path,
    uniform_bank_path: Path | None,
    composition: CatalogComposition,
) -> xr.Dataset:
    """Compose an in-memory catalog from bank files, per ``composition``.

    Reproduces ``MixtureSimulator``'s law directly with ``jax.random`` instead
    of through ``gwmock_pop``: per-sample component assignments are drawn once
    from ``mixture_seed`` (multinomial with probabilities
    ``[1 - eps, eps]``), then each component contributes its next-in-sequence
    bank samples. Because bank draws are prefix-stable (the same
    construction-time RNG stream regardless of how many samples are later
    requested), this is bit-identical to drawing directly from the
    corresponding single larger mixture population -- so a prefix of the
    result is itself a valid mixture sample.

    ``eps == 0`` short-circuits to a bank prefix with no RNG draw at all,
    which is what makes every existing eps=0 catalog file bit-identical to
    its composed replacement.
    """
    n = composition.num_samples
    epsilon = composition.uniform_mixing_fraction
    if epsilon == 0.0:
        return _bank_prefix(md_bank_path, n, label=composition.md_bank)

    if uniform_bank_path is None or composition.mixture_seed is None:
        raise ValueError(
            f"composition {composition.name!r}: uniform_mixing_fraction > 0 "
            "requires both uniform_bank_path and mixture_seed"
        )

    key = jax.random.key(composition.mixture_seed)
    log_probs = jnp.log(jnp.asarray([1.0 - epsilon, epsilon]))
    assignments = jax.random.categorical(key, log_probs, shape=(n,))
    counts = [int((assignments == component).sum()) for component in (0, 1)]

    md_part = _bank_prefix(md_bank_path, counts[0], label=composition.md_bank)
    uniform_part = _bank_prefix(
        uniform_bank_path, counts[1], label=composition.uniform_bank or ""
    )
    combined = xr.concat([md_part, uniform_part], dim="sample")

    order = jnp.argsort(assignments, stable=True)
    inverse = np.asarray(jnp.argsort(order))
    return combined.isel(sample=inverse)


def _bank_prefix(path: Path, count: int, *, label: str) -> xr.Dataset:
    """Load the first ``count`` samples of a bank, eagerly."""
    catalog = open_catalog(path)
    available = catalog.sizes["sample"]
    if count > available:
        raise ValueError(
            f"bank {label!r} at {path} holds {available} samples, but the "
            f"composition needs {count}"
        )
    return catalog.isel(sample=slice(0, count)).load()


def propagate_catalog(
    catalog: xr.Dataset, *, fiducials: dict[str, float]
) -> xr.Dataset:
    """Apply fiducial GW propagation to an in-memory catalog. Numpy-backed."""
    return apply_gw_distance_to_power(
        catalog,
        xi_0=float(fiducials["xi_0"]),
        xi_n=float(fiducials["xi_n"]),
    )


def load_propagated_catalog(path: Path, *, fiducials: dict[str, float]) -> xr.Dataset:
    """Load a catalog and apply fiducial GW propagation. Numpy-backed."""
    return propagate_catalog(load_catalog(path), fiducials=fiducials)


def samples_from_catalog(catalog: xr.Dataset) -> dict[str, jax.Array]:
    """Unstack ``source_parameters`` into the dict shape the model expects."""
    return {
        str(name): jnp.asarray(catalog.source_parameters.sel(parameter=name).values)
        for name in catalog.parameter.values
    }


def truncate_catalog_samples(
    catalog: xr.Dataset,
    *,
    label: str,
    minimum_redshift: float,
    maximum_redshift: float,
) -> xr.Dataset:
    """Restrict a catalog to samples inside the analysis redshift window.

    Generation draws truncated to the window follow the same law as drawing
    directly from it, so this is how a full-range catalog meets the analysis
    support instead of being rejected for spanning below it.
    """
    required = {"redshift", "luminosity_distance"}
    missing = sorted(required - set(catalog.parameter.values))
    if missing:
        raise ValueError(
            f"{label} catalog samples are missing required parameter(s): "
            + ", ".join(missing)
        )

    redshift = catalog.source_parameters.sel(parameter="redshift").values
    window = (redshift >= minimum_redshift) & (redshift <= maximum_redshift)
    if not window.any():
        raise ValueError(
            f"{label} catalog has no samples in the analysis redshift window "
            f"[{minimum_redshift:.4g}, {maximum_redshift:.4g}]"
        )
    return catalog.isel(sample=window)


def compute_proposal_logprob(
    redshift: jax.typing.ArrayLike,
    proposal: ProposalConfig,
) -> jax.Array:
    """Evaluate the fixed MD/uniform proposal at catalog redshifts."""
    redshift = jnp.asarray(redshift)
    proposal_grid = jnp.linspace(
        proposal.minimum_redshift, proposal.maximum_redshift, proposal.n_grid
    )
    _, _, md_logprob = compute_merger_rate_distance_and_logprob(
        {
            "H0": proposal.H0,
            "Omega_m": proposal.Omega_m,
            "gamma": proposal.gamma,
            "kappa": proposal.kappa,
            "z_peak": proposal.z_peak,
            "local_merger_rate": 1.0,
        },
        {"redshift": redshift},
        redshift_grid=proposal_grid,
    )
    epsilon = proposal.uniform_mixing_fraction
    if epsilon == 0.0:
        return md_logprob

    in_support = (redshift >= proposal.minimum_redshift) & (
        redshift <= proposal.maximum_redshift
    )
    uniform_logprob = jnp.where(
        in_support,
        -jnp.log(proposal.maximum_redshift - proposal.minimum_redshift),
        -jnp.inf,
    )
    if epsilon == 1.0:
        return uniform_logprob
    return jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) + uniform_logprob,
    )


def validate_matching_frequency_grids(
    injection_frequencies: ArrayLike, proposal_frequencies: ArrayLike
) -> None:
    """Require injection and proposal waveforms to share the exact frequency grid."""
    injection_frequencies = np.asarray(injection_frequencies)
    proposal_frequencies = np.asarray(proposal_frequencies)
    if not np.array_equal(injection_frequencies, proposal_frequencies):
        raise ValueError(
            "injection and proposal catalogs must have identical frequency grids"
        )


def compute_fiducial_injection_spectrum(
    polarization_power: ArrayLike,
    samples: Mapping[str, ArrayLike],
    *,
    fiducials: dict[str, float],
    redshift_grid: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Return the fiducial total rate and independently estimated spectrum."""
    polarization_power = jnp.asarray(polarization_power)
    samples_jax = {name: jnp.asarray(value) for name, value in samples.items()}
    redshift_grid = jnp.asarray(redshift_grid)
    total_rate, _, _ = compute_merger_rate_distance_and_logprob(
        fiducials,
        samples_jax,
        redshift_grid=redshift_grid,
    )
    weights = jnp.ones(polarization_power.shape[1])
    spectrum = spectral_density(
        polarization_power,
        weights,
        total_rate,
        average_mode="analytic_inclination",
    )
    return total_rate, spectrum
