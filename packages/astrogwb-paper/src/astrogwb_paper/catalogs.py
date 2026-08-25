"""Shared loading and validation for injection and proposal waveform catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb.waveform import apply_gw_distance_to_power, load_catalog
from numpy.typing import ArrayLike

from astrogwb_paper.config.mcmc import ProposalConfig


def load_propagated_catalog(path: Path, *, fiducials: dict[str, float]) -> xr.Dataset:
    """Load a catalog and apply fiducial GW propagation. Numpy-backed."""
    return apply_gw_distance_to_power(
        load_catalog(path),
        xi_0=float(fiducials["xi_0"]),
        xi_n=float(fiducials["xi_n"]),
    )


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
