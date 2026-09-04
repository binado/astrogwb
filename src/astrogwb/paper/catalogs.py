"""Loading, validation, and preparation of injection and proposal catalogs.

A catalog is a file: ``outputs/catalogs/<name>.h5``, built once by
``scripts/generate_catalog.py`` from ``config/catalogs/defs/<name>.toml``. A
run names one for each role, so there is nothing to compose here -- loading is
just reading the file, and the density its samples follow is read back off its
own recorded provenance rather than reassembled from the run config.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from astrogwb.catalog.io import open_catalog
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from astrogwb.paper.config.catalogs import (
    CatalogProvenance,
    check_fiducials_match,
    resolve_proposal,
)
from astrogwb.paper.config.mcmc import ProposalConfig, RunConfig
from astrogwb.waveform import apply_gw_distance_to_power


def load_run_catalog(path: Path | str, *, label: str) -> xr.Dataset:
    """Load one catalog file eagerly, validating its format.

    ``label`` is the role -- ``"injection"`` or ``"proposal"`` -- and is what
    identifies the catalog in error messages.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} catalog not found: {path}")
    return open_catalog(path).load()


def resolve_run_proposal(
    config: RunConfig, proposal_path: Path | str
) -> ProposalConfig:
    """Derive a run's importance-sampling density from its proposal catalog.

    This is called before runtime configuration. Reading HDF5 attributes does
    not initialize the JAX backend, so a fiducial or support mismatch still
    fails before a device is claimed. The file is authoritative because the
    population configs it was drawn from may have changed since generation.
    """
    label = str(proposal_path)
    provenance = CatalogProvenance.from_file(proposal_path)
    check_fiducials_match(provenance, config.fiducials, label=label)
    return resolve_proposal(
        provenance.redshift_proposal,
        minimum_redshift=config.cosmology.minimum_redshift,
        maximum_redshift=config.cosmology.maximum_redshift,
        label=label,
    )


def propagate_catalog(
    catalog: xr.Dataset, *, fiducials: dict[str, float]
) -> xr.Dataset:
    """Apply fiducial GW propagation to an in-memory catalog. Numpy-backed."""
    redshift = catalog.source_parameters.sel(parameter="redshift").values
    corrected_power = apply_gw_distance_to_power(
        catalog.polarization_power.values,
        redshift,
        xi_0=float(fiducials["xi_0"]),
        xi_n=float(fiducials["xi_n"]),
    )
    return catalog.assign(
        polarization_power=(catalog.polarization_power.dims, corrected_power)
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
    return catalog.isel(sample=np.flatnonzero(np.asarray(window)))


def compute_proposal_logprob(
    redshift: jax.typing.ArrayLike,
    proposal: ProposalConfig,
) -> jax.Array:
    """Evaluate the fixed MD/uniform proposal at catalog redshifts."""
    redshift = jnp.asarray(redshift)
    epsilon = proposal.uniform_mixing_fraction

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

    if epsilon == 0.0:
        return md_logprob
    return jnp.logaddexp(
        jnp.log1p(-epsilon) + md_logprob,
        jnp.log(epsilon) + uniform_logprob,
    )


def validate_matching_frequency_grids(
    injection_frequencies: ArrayLike,
    proposal_frequencies: ArrayLike,
    *,
    label: str = "injection and proposal",
) -> None:
    """Require two waveform catalogs to share the exact frequency grid."""
    injection_frequencies = np.asarray(injection_frequencies)
    proposal_frequencies = np.asarray(proposal_frequencies)
    if not np.array_equal(injection_frequencies, proposal_frequencies):
        raise ValueError(f"{label} catalogs must have identical frequency grids")


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
