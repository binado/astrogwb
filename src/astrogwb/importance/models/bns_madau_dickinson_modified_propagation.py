"""Reference importance-weights model: BNS + Madau-Dickinson rate + modified propagation.

This module packages one concrete realization of the
:class:`~astrogwb.importance.protocol.MergerRateAndLogWeightsFn` callback:
a binary-neutron-star population whose merger-rate density follows the
Madau-Dickinson (2017) shape, evolved on a flat-LambdaCDM cosmology, with a
phenomenological GW-to-EM luminosity-distance ratio (``xi_0``, ``xi_n``)
capturing a modified-gravity propagation effect.

It is a *reference implementation* -- the NumPyro model itself accepts any
callback satisfying the protocol, so callers may substitute their own.

Factory contract: :func:`make_merger_rate_and_log_weights_fn` must be called
with a concrete ``z_grid`` (not a tracer); it extracts ``max_redshift`` /
``n_grid`` eagerly so the returned closure is safe to trace inside ``jax.jit``
during NUTS.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_rate

from astrogwb.cosmology import distance_and_volume_grid, log_gw_em_ratio
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.utils import SECONDS_PER_YEAR


def make_merger_rate_and_log_weights_fn(
    *,
    z_grid: jnp.ndarray,
    proposal_log_pdf: jnp.ndarray,
    fiducial_xi_0: float,
    fiducial_xi_n: float,
) -> MergerRateAndLogWeightsFn:
    """Build the merger-rate + importance-log-weights callback.

    The returned closure reweights a fixed proposal catalog (drawn at the
    fiducial parameter point) to arbitrary sampled hyperparameters. It is
    JAX-traceable and intended to be passed (pre-built) to
    :func:`~astrogwb.sampling.numpyro_model.numpyro_model`.
    The callback's ``params`` mapping must include ``local_merger_rate`` in
    ``Gpc^-3 yr^-1`` alongside the cosmology and population parameters.

    Parameters
    ----------
    z_grid:
        Redshift grid used for the cosmology integrals and MD normalization.
        **Must be a concrete array** -- its extent and size are extracted
        eagerly below so the closure never calls ``float()`` on a tracer.
    proposal_log_pdf:
        Precomputed ``log`` of the proposal redshift PDF evaluated at the
        catalog redshifts, shape ``(N,)``.
    fiducial_xi_0, fiducial_xi_n:
        Modified-propagation parameters of the *proposal* catalog; the weight
        Jacobian includes their ratio against the sampled ``xi_0`` / ``xi_n``.

    Returns
    -------
    MergerRateAndLogWeightsFn
        Callable ``(params, samples) -> (total_merger_rate, log_weights)``.
        ``total_merger_rate`` is in mergers per second; ``log_weights`` has
        shape ``(N,)``.
    """
    # Extract the cosmology grid extent eagerly (z_grid is concrete here at
    # factory-build time) so the jitted closure never calls float() on a tracer.
    max_redshift = float(z_grid[-1])
    n_grid = int(z_grid.shape[0])

    def merger_rate_and_log_weights_fn(
        params: Mapping[str, Any],
        samples: Mapping[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = samples["redshift"]
        d_l_fid = samples["luminosity_distance"]

        luminosity_distance_grid, dvc_dz_grid = distance_and_volume_grid(
            params, max_redshift, n_grid
        )
        d_l_theta = jnp.interp(
            z,
            z_grid,
            luminosity_distance_grid,
            left=luminosity_distance_grid[0],
            right=luminosity_distance_grid[-1],
        )

        rate_shape_grid = madau_dickinson_rate(
            z_grid, params["gamma"], params["kappa"], params["z_peak"]
        )
        unnormalized_pdf_grid = rate_shape_grid / (1.0 + z_grid) * dvc_dz_grid
        integral_Mpc3 = jnp.trapezoid(unnormalized_pdf_grid, z_grid)

        rate_shape_samples = madau_dickinson_rate(
            z, params["gamma"], params["kappa"], params["z_peak"]
        )
        dvc_dz_samples = jnp.interp(
            z,
            z_grid,
            dvc_dz_grid,
            left=dvc_dz_grid[0],
            right=dvc_dz_grid[-1],
        )
        target_pdf = rate_shape_samples / (1.0 + z) * dvc_dz_samples / integral_Mpc3

        log_fiducial_gw_em_ratio = log_gw_em_ratio(z, fiducial_xi_0, fiducial_xi_n)
        log_target_gw_em_ratio = log_gw_em_ratio(z, params["xi_0"], params["xi_n"])
        log_weights = (
            jnp.log(target_pdf)
            - proposal_log_pdf
            + 2.0 * jnp.log(d_l_fid)
            - 2.0 * jnp.log(d_l_theta)
            + 2.0 * log_fiducial_gw_em_ratio
            - 2.0 * log_target_gw_em_ratio
        )

        total_merger_rate = (
            1e-9 * params["local_merger_rate"] * integral_Mpc3 / SECONDS_PER_YEAR
        )
        return total_merger_rate, log_weights

    return merger_rate_and_log_weights_fn
