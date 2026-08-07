"""Reference importance-weights model: BNS + Madau-Dickinson rate + modified propagation.

This module packages one concrete realization of the
:class:`~astrogwb.importance.protocol.MergerRateAndLogWeightsFn` callback:
a binary-neutron-star population whose merger-rate density follows the
Madau-Dickinson (2017) shape, evolved on a flat-LambdaCDM cosmology, with a
phenomenological GW-to-EM luminosity-distance ratio (``xi_0``, ``xi_n``)
capturing a modified-gravity propagation effect.

It is a *reference implementation* -- the NumPyro model itself accepts any
callback satisfying the protocol, so callers may substitute their own.

Factory contract: :func:`make_merger_rate_and_log_weights_fn` takes a
``redshift_grid`` array that is captured by the returned closure, so the closure
never needs to extract static Python scalars from traced values and is safe
to trace inside ``jax.jit`` during NUTS.

The proposal and target share
:func:`compute_merger_rate_distance_and_logprob`. Precompute
``proposal_logprob`` by evaluating that function at the fiducials (third
return value); the callback returns importance log-weights via
:func:`log_weights`, which reweights the redshift PDF against the catalog
fiducial luminosity distances and the GW/EM ratio correction.
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

AMPLITUDE_PARAMETERS: tuple[str, ...] = ("H0", "local_merger_rate")
"""Parameters this callback supports marginalizing analytically."""


# Absolute scalings as module-level ``def``s (not closures over the fiducial)
# so they are singletons: ``AmplitudeConditional`` carries the amplitude
# function as pytree *aux* data, which JAX hashes into the jit cache key. A
# lambda (or a ``functools.partial`` over a float) is identity-hashed, so a
# fresh one per call would retrace the model on every construction. The
# consumer forms the ratio ``f(varphi)/f(varphi_fid)`` itself.
#
# The predicted spectrum factorizes as ``f = g_R * g_F``. ``local_merger_rate``
# enters only through ``total_merger_rate`` (linear; absent from ``log_weights``),
# so ``g_R = varphi``, ``g_F = 1``, ``f = varphi``. ``H0`` enters the rate via
# ``dV_c/dz ∝ h0^{-3}`` and the mean energy flux via
# ``exp(-2 log d_L) ∝ h0^2``, so ``g_R = varphi^{-3}``, ``g_F = varphi^2``,
# and ``f = varphi^{-1}``.
def merger_rate_H0_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Merger-rate scaling :math:`g_R(H_0) = H_0^{-3}`."""
    return marginalized_parameter**-3


def amplitude_H0_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Total amplitude scaling :math:`f(H_0) = H_0^{-1}` (:math:`g_R g_F`)."""
    return 1.0 / marginalized_parameter


def merger_rate_local_merger_rate_fn(
    marginalized_parameter: jax.Array,
) -> jax.Array:
    """Merger-rate scaling :math:`g_R(\\mathcal{R}_0) = \\mathcal{R}_0`."""
    return marginalized_parameter


def amplitude_local_merger_rate_fn(
    marginalized_parameter: jax.Array,
) -> jax.Array:
    """Total amplitude scaling :math:`f(\\mathcal{R}_0) = \\mathcal{R}_0`."""
    return marginalized_parameter


def compute_merger_rate_distance_and_logprob(
    params: Mapping[str, Any],
    samples: Mapping[str, jax.Array],
    *,
    redshift_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Merger rate, luminosity distance, and redshift log-pdf at catalog samples.

    Builds cosmology tables on ``redshift_grid`` via
    :func:`~astrogwb.cosmology.distance_and_volume_grid`, normalizes the
    Madau-Dickinson redshift weight by trapezoidal integration on that grid,
    and evaluates the redshift PDF

    :math:`\mathrm{logpdf} = \log p(z|\theta)`

    at ``samples["redshift"]`` by linearly interpolating the complete
    unnormalized redshift density. This makes the interpolated density's
    integral exactly equal to its trapezoidal normalization and avoids
    reevaluating the Madau-Dickinson rate at every catalog sample. Also returns
    the interpolated luminosity distance ``d_L(z|\theta)``. The same function
    is used for the proposal (at fiducials) and the target (at sampled
    ``params``); :func:`log_weights` combines these with the catalog fiducial
    distances and the GW/EM ratio correction.

    Parameters
    ----------
    params:
        Hyperparameters. Must include ``H0``, ``Omega_m``, ``gamma``,
        ``kappa``, ``z_peak``, ``xi_0``, ``xi_n``, and ``local_merger_rate``
        (in ``Gpc^-3 yr^-1``).
    samples:
        Catalog arrays; must include ``redshift`` with leading dimension
        ``N``.
    redshift_grid:
        Redshift grid for the cosmology integrals and MD normalization.
        ``rate_shape_grid`` and ``dvc_dz_grid`` are evaluated on this exact
        grid, so the two can never fall out of alignment.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(total_merger_rate, luminosity_distance, logpdf)``. Rate is in
        mergers per second; ``luminosity_distance`` and ``logpdf`` have shape
        ``(N,)``.
    """
    redshift = samples["redshift"]

    luminosity_distance_grid, dvc_dz_grid = distance_and_volume_grid(
        params, redshift_grid
    )
    rate_shape_grid = madau_dickinson_rate(
        redshift_grid, params["gamma"], params["kappa"], params["z_peak"]
    )
    unnormalized_pdf_grid = rate_shape_grid / (1.0 + redshift_grid) * dvc_dz_grid
    integral_mpc3 = jnp.trapezoid(unnormalized_pdf_grid, redshift_grid)

    unnormalized_pdf = jnp.interp(
        redshift,
        redshift_grid,
        unnormalized_pdf_grid,
        left=0.0,
        right=0.0,
    )
    luminosity_distance = jnp.interp(
        redshift,
        redshift_grid,
        luminosity_distance_grid,
        left=luminosity_distance_grid[0],
        right=luminosity_distance_grid[-1],
    )
    logpdf = jnp.log(unnormalized_pdf) - jnp.log(integral_mpc3)
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
    redshift_grid: jax.Array,
    proposal_logprob: jax.Array,
) -> MergerRateAndLogWeightsFn:
    """Build the merger-rate + importance-log-weights callback.

    The returned closure reweights a fixed proposal catalog (drawn at the
    fiducial parameter point) to arbitrary sampled hyperparameters. It is
    JAX-traceable and intended to be passed (pre-built) to
    :func:`~astrogwb.sampling.models.spectral_density_model`.

    Precompute ``proposal_logprob`` with
    :func:`compute_merger_rate_distance_and_logprob` at the fiducials::

        _, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
            fiducials, samples, redshift_grid=redshift_grid
        )

    Parameters
    ----------
    fiducials:
        Fiducial hyperparameters used for the GW/EM ratio correction inside
        :func:`log_weights`. Must include ``xi_0`` and ``xi_n``.
    redshift_grid:
        Redshift grid used for the cosmology integrals and MD normalization.
        Captured by the returned closure as a constant array.
    proposal_logprob:
        Precomputed redshift log-pdf at the fiducials for the catalog
        redshifts, shape ``(N,)``. Typically the third return value of
        :func:`compute_merger_rate_distance_and_logprob`.

    Returns
    -------
    MergerRateAndLogWeightsFn
        Callable ``(params, samples) -> (total_merger_rate, log_weights)``.
        ``total_merger_rate`` is in mergers per second; ``log_weights`` has
        shape ``(N,)``. ``samples`` must include ``redshift`` and
        ``luminosity_distance`` (fiducial EM distances from the catalog).
    """

    def merger_rate_and_log_weights_fn(
        params: Mapping[str, Any],
        samples: Mapping[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        total_merger_rate, luminosity_distance, logprob = (
            compute_merger_rate_distance_and_logprob(
                params,
                samples,
                redshift_grid=redshift_grid,
            )
        )
        logw = log_weights(
            logprob, proposal_logprob, luminosity_distance, params, samples, fiducials
        )
        return total_merger_rate, logw

    return merger_rate_and_log_weights_fn
