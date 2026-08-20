"""Shared loading and validation for injection and proposal waveform catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb.waveform import apply_gw_distance_to_waveforms
from astrogwb.waveform import polarization_power as compute_polarization_power
from pluscross import load_catalog

from astrogwb_paper.config.mcmc import ProposalConfig


@dataclass(frozen=True)
class CatalogArrays:
    """JAX arrays reduced from one waveform catalog."""

    frequencies: Any
    polarization_power: Any
    samples: dict[str, Any]


def load_catalog_arrays(
    path: Path,
    *,
    fiducials: dict[str, float],
) -> CatalogArrays:
    """Load a catalog, apply fiducial GW propagation, and reduce its waveforms."""
    catalog = load_catalog(path)
    catalog = apply_gw_distance_to_waveforms(
        catalog,
        xi_0=float(fiducials["xi_0"]),
        xi_n=float(fiducials["xi_n"]),
    )
    arrays = CatalogArrays(
        frequencies=jnp.asarray(catalog.frequencies),
        polarization_power=jnp.asarray(compute_polarization_power(catalog)),
        samples={
            name: jnp.asarray(values)
            for name, values in catalog.source_parameters.items()
        },
    )
    return arrays


def validate_catalog_samples(
    catalog: CatalogArrays,
    *,
    label: str,
    z_min: float,
    z_max: float,
) -> None:
    """Validate required columns and redshift support."""
    required = {"redshift", "luminosity_distance"}
    missing = sorted(required - set(catalog.samples))
    if missing:
        raise ValueError(
            f"{label} catalog samples are missing required parameter(s): "
            + ", ".join(missing)
        )

    redshift = np.asarray(catalog.samples["redshift"])
    z_lo = float(np.min(redshift))
    z_hi = float(np.max(redshift))
    if z_lo < z_min or z_hi > z_max:
        raise ValueError(
            f"{label} catalog redshifts span [{z_lo:.4g}, {z_hi:.4g}] but "
            f"[z_min, z_max] is [{z_min:.4g}, {z_max:.4g}]"
        )


def compute_proposal_logprob(
    samples: dict[str, Any],
    proposal: ProposalConfig,
) -> Any:
    """Evaluate the fixed MD/uniform proposal at catalog redshifts."""
    redshift = jnp.asarray(samples["redshift"])
    proposal_grid = jnp.linspace(proposal.z_min, proposal.z_max, proposal.n_grid)
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

    in_support = (redshift >= proposal.z_min) & (redshift <= proposal.z_max)
    uniform_logprob = jnp.where(
        in_support,
        -jnp.log(proposal.z_max - proposal.z_min),
        -jnp.inf,
    )
    if epsilon == 1.0:
        return uniform_logprob
    return jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) + uniform_logprob,
    )


def validate_matching_frequency_grids(
    injection: CatalogArrays, proposal: CatalogArrays
) -> None:
    """Require injection and proposal waveforms to share the exact frequency grid."""
    injection_frequencies = np.asarray(injection.frequencies)
    proposal_frequencies = np.asarray(proposal.frequencies)
    if not np.array_equal(injection_frequencies, proposal_frequencies):
        raise ValueError(
            "injection and proposal catalogs must have identical frequency grids"
        )


def compute_fiducial_injection_spectrum(
    injection: CatalogArrays,
    *,
    fiducials: dict[str, float],
    redshift_grid: Any,
) -> tuple[Any, Any]:
    """Return the fiducial total rate and independently estimated spectrum."""
    total_rate, _, _ = compute_merger_rate_distance_and_logprob(
        fiducials,
        injection.samples,
        redshift_grid=redshift_grid,
    )
    weights = jnp.ones(injection.polarization_power.shape[1])
    spectrum = spectral_density(
        injection.polarization_power,
        weights,
        total_rate,
        average_mode="analytic_inclination",
    )
    return total_rate, spectrum
