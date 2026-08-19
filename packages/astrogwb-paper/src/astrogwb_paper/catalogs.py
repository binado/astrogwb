"""Shared loading and validation for injection and proposal waveform catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from astrogwb.gwb import spectral_density
from astrogwb.waveform import apply_gw_distance_to_waveforms
from astrogwb.waveform import polarization_power as compute_polarization_power
from pluscross import load_catalog

PROPOSAL_REDSHIFT_LOGPDF = "proposal_redshift_logpdf"


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
    require_proposal_density: bool,
) -> None:
    """Validate required columns, finite proposal density, and redshift support."""
    required = {"redshift", "luminosity_distance"}
    if require_proposal_density:
        required.add(PROPOSAL_REDSHIFT_LOGPDF)
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
    if require_proposal_density:
        logpdf = np.asarray(catalog.samples[PROPOSAL_REDSHIFT_LOGPDF])
        if logpdf.shape != redshift.shape:
            raise ValueError(
                f"{label} catalog {PROPOSAL_REDSHIFT_LOGPDF} shape "
                f"{logpdf.shape} does not match redshift shape {redshift.shape}"
            )
        if not np.all(np.isfinite(logpdf)):
            raise ValueError(
                f"{label} catalog {PROPOSAL_REDSHIFT_LOGPDF} must be finite"
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
    # gwmock_pop's Madau-Dickinson import initializes the XLA backend.
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        compute_merger_rate_distance_and_logprob,
    )

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
