"""Cosmology helpers for the importance-weighting reference models.

These functions are pure and JAX-traceable so they may run both at
catalog-build time (concrete arrays) and inside the jitted NUTS model
(traced arrays). They wrap ``gwmock_pop.cosmology.flat_lambda_cdm``.

A load-bearing contract: :func:`flat_lcdm_grid` takes ``max_redshift`` and
``n_grid`` as *static Python scalars*, never as tracers. Concretizing a tracer
(e.g. ``float(z_grid[-1])``) would raise ``ConcretizationTypeError`` once the
function runs under ``jax.jit`` inside the NUTS model. Callers must extract
these scalars eagerly at factory-build time; see
:func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.make_merger_rate_and_log_weights_fn`.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
from gwmock_pop.cosmology.flat_lambda_cdm import (
    SPEED_OF_LIGHT,
    build_distance_lookup,
    compute_normalized_hubble_parameter,
)


def log_gw_em_ratio(z, xi_0, xi_n):
    """Log of the modified-propagation GW-to-EM luminosity-distance ratio.

    Models a departure from the standard ``d_GW = d_EM`` propagation as
    ``xi_0 + (1 - xi_0) * (1 + z)^(-xi_n)`` and returns its natural log.
    With ``xi_0 = 1`` the ratio is identically 1 (GR propagation); the
    function returns 0 everywhere in that case.

    Parameters
    ----------
    z:
        Redshift array (JAX-traceable).
    xi_0, xi_n:
        Modified-propagation parameters.

    Returns
    -------
    jax.Array
        ``log(xi_0 + (1 - xi_0) * exp(-xi_n * log1p(z)))``, same shape as ``z``.
    """
    return jnp.log(xi_0 + (1.0 - xi_0) * jnp.exp(-xi_n * jnp.log1p(z)))


def flat_lcdm_grid(
    params: Mapping[str, Any],
    max_redshift: float,
    n_grid: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Luminosity distance and differential comoving volume on a redshift grid.

    Parameters
    ----------
    params:
        Mapping with keys ``"H0"`` (dimensionless Hubble constant) and
        ``"Omega_m"`` (matter density). May contain tracers during NUTS.
    max_redshift:
        Upper edge of the redshift grid. **Must be a static Python float** --
        not a tracer -- to avoid concretization errors under ``jax.jit``.
    n_grid:
        Number of grid points. **Must be a static Python int.**

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(luminosity_distance, differential_comoving_volume)`` on the grid,
        each of shape ``(n_grid,)``. The differential comoving volume is in
        ``Mpc^3 / sr`` (``SPEED_OF_LIGHT / 1000`` factor in the lookup).
    """
    h0 = params["H0"]
    omega_m = params["Omega_m"]

    # ``max_redshift`` / ``n_grid`` are passed as static Python scalars, not read
    # from a (possibly traced) ``z_grid`` -- ``float(tracer)`` would raise a
    # ConcretizationTypeError once this runs inside the jitted NUTS model.
    z, comoving_distance, luminosity_distance = build_distance_lookup(
        hubble_constant=h0,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
    e_z = compute_normalized_hubble_parameter(redshift=z, omega_m=omega_m)
    differential_comoving_volume = (
        4.0 * jnp.pi * comoving_distance**2 / (h0 * e_z) * SPEED_OF_LIGHT / 1000
    )
    return luminosity_distance, differential_comoving_volume
