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

The proposal and target share
:func:`compute_merger_rate_and_log_density`. Precompute
``proposal_logprob`` by evaluating that function at the fiducials; the
callback returns ``log_density(theta) - proposal_logprob``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_rate

from astrogwb.cosmology import distance_and_volume_grid, log_gw_em_ratio
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.utils import SECONDS_PER_YEAR


def compute_merger_rate_distance_and_logprob(
    params: Mapping[str, Any],
    samples: Mapping[str, jax.Array],
    *,
    z_grid: jax.Array,
    max_redshift: float | None = None,
    n_grid: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Merger rate and params-dependent log-density at catalog redshifts.

    Builds cosmology tables on ``z_grid`` via
    :func:`~astrogwb.cosmology.distance_and_volume_grid`, normalizes the
    Madau-Dickinson redshift weight by trapezoidal integration on that grid,
    and evaluates

    :math:`\mathrm{log\_density} = \log p(z|\theta) - 2 \log d_L(z|\theta)
    - 2 \log \Xi(z; \xi_0, \xi_n)`

    at ``samples["redshift"]`` by linearly interpolating ``dV_c/dz`` and
    ``d_L``. The same function is used for the proposal (at fiducials) and
    the target (at sampled ``params``), so their difference is identically
    zero at the fiducial point.

    Parameters
    ----------
    params:
        Hyperparameters. Must include ``H0``, ``Omega_m``, ``gamma``,
        ``kappa``, ``z_peak``, ``xi_0``, ``xi_n``, and ``local_merger_rate``
        (in ``Gpc^-3 yr^-1``).
    samples:
        Catalog arrays; must include ``redshift`` with leading dimension
        ``N``.
    z_grid:
        Concrete redshift grid for cosmology integrals and MD normalization.
    max_redshift, n_grid:
        Optional static Python scalars for the cosmology lookup. When omitted
        they are read from ``z_grid``. Callers under ``jax.jit`` should pass
        them explicitly so ``float(z_grid[-1])`` is never applied to a tracer.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(total_merger_rate, log_density)``. Rate is in mergers per second;
        ``log_density`` has shape ``(N,)``.
    """
    z = samples["redshift"]
    if max_redshift is None:
        max_redshift = float(z_grid[-1])
    if n_grid is None:
        n_grid = int(z_grid.shape[0])

    luminosity_distance_grid, dvc_dz_grid = distance_and_volume_grid(
        params, max_redshift, n_grid
    )
    rate_shape_grid = madau_dickinson_rate(
        z_grid, params["gamma"], params["kappa"], params["z_peak"]
    )
    unnormalized_pdf_grid = rate_shape_grid / (1.0 + z_grid) * dvc_dz_grid
    integral_mpc3 = jnp.trapezoid(unnormalized_pdf_grid, z_grid)

    rate_shape = madau_dickinson_rate(
        z, params["gamma"], params["kappa"], params["z_peak"]
    )
    dvc_dz = jnp.interp(
        z,
        z_grid,
        dvc_dz_grid,
        left=dvc_dz_grid[0],
        right=dvc_dz_grid[-1],
    )
    luminosity_distance = jnp.interp(
        z,
        z_grid,
        luminosity_distance_grid,
        left=luminosity_distance_grid[0],
        right=luminosity_distance_grid[-1],
    )
    pdf = rate_shape / (1.0 + z) * dvc_dz / integral_mpc3
    logpdf = jnp.log(pdf)
    total_merger_rate = (
        1e-9 * params["local_merger_rate"] * integral_mpc3 / SECONDS_PER_YEAR
    )
    return total_merger_rate, luminosity_distance, logpdf


def log_weights(
    logprob: jax.Array,
    proposal_logprob: jax.Array,
    luminosity_distance: jax.Array,
    parameters: Mapping[str, Any],
    samples: Mapping[str, jax.Array],
    fiducials: Mapping[str, Any],
) -> jax.Array:
    redshift = samples["redshift"]
    xi, n = parameters["xi_0"], parameters["xi_n"]
    xi_fid, n_fid = fiducials["xi_0"], fiducials["xi_n"]
    luminosity_distance_fid = samples["luminosity_distance"]
    logdiff_dl_em = jnp.log(luminosity_distance) - jnp.log(luminosity_distance_fid)
    logdiff_gw_em_ratio = log_gw_em_ratio(redshift, xi, n) - log_gw_em_ratio(
        redshift, xi_fid, n_fid
    )
    logdiff_dl_gw = logdiff_dl_em + logdiff_gw_em_ratio
    return logprob - proposal_logprob - logdiff_dl_gw * 2.0


def make_merger_rate_and_log_weights_fn(
    *,
    fiducials: Mapping[str, Any],
    z_grid: jax.Array,
    proposal_logprob: jax.Array,
) -> MergerRateAndLogWeightsFn:
    """Build the merger-rate + importance-log-weights callback.

    The returned closure reweights a fixed proposal catalog (drawn at the
    fiducial parameter point) to arbitrary sampled hyperparameters. It is
    JAX-traceable and intended to be passed (pre-built) to
    :func:`~astrogwb.sampling.numpyro_model.numpyro_model`.

    Precompute ``proposal_logprob`` with
    :func:`compute_merger_rate_and_log_density` at the fiducials::

        _, proposal_logprob = compute_merger_rate_and_log_density(
            fiducials, samples, z_grid=z_grid
        )

    Parameters
    ----------
    z_grid:
        Redshift grid used for the cosmology integrals and MD normalization.
        **Must be a concrete array** -- its extent and size are extracted
        eagerly below so the closure never calls ``float()`` on a tracer.
    proposal_logprob:
        Precomputed log-density at the fiducials for the catalog redshifts,
        shape ``(N,)``. Typically the second return value of
        :func:`compute_merger_rate_and_log_density`.

    Returns
    -------
    MergerRateAndLogWeightsFn
        Callable ``(params, samples) -> (total_merger_rate, log_weights)``.
        ``total_merger_rate`` is in mergers per second; ``log_weights`` has
        shape ``(N,)`` and equals ``log_density(params) - proposal_logprob``.
    """
    # Extract the cosmology grid extent eagerly (z_grid is concrete here at
    # factory-build time) so the jitted closure never calls float() on a tracer.
    max_redshift = float(z_grid[-1])
    n_grid = int(z_grid.shape[0])

    def merger_rate_and_log_weights_fn(
        params: Mapping[str, Any],
        samples: Mapping[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        total_merger_rate, luminosity_distance, logprob = (
            compute_merger_rate_distance_and_logprob(
                params,
                samples,
                z_grid=z_grid,
                max_redshift=max_redshift,
                n_grid=n_grid,
            )
        )
        logw = log_weights(
            logprob, proposal_logprob, luminosity_distance, params, samples, fiducials
        )
        return total_merger_rate, logw

    return merger_rate_and_log_weights_fn
