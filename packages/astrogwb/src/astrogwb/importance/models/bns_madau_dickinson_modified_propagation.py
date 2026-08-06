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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_rate

from astrogwb.cosmology import distance_and_volume_grid, log_gw_em_ratio
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.sampling.amplitude import MeanEnergyFluxAmplitudeFn, MergerRateAmplitudeFn
from astrogwb.utils import SECONDS_PER_YEAR

AMPLITUDE_PARAMETERS: tuple[str, ...] = ("H0", "local_merger_rate")
"""Parameters this callback supports marginalizing analytically."""


class AmplitudeScalings(NamedTuple):
    """The two independently-scaling factors that make up the amplitude, :math:`f = g_R \\cdot g_F`."""

    merger_rate: MergerRateAmplitudeFn
    mean_energy_flux: MeanEnergyFluxAmplitudeFn


def amplitude_scalings(parameter: str, fiducial: float) -> AmplitudeScalings:
    r"""Merger-rate and mean-energy-flux scalings for one amplitude parameter.

    ``local_merger_rate`` enters :func:`compute_merger_rate_distance_and_logprob`
    only through ``total_merger_rate = 1e-9 * local_merger_rate * integral_mpc3
    / SECONDS_PER_YEAR`` -- linear in the parameter and absent from
    ``logpdf`` and hence from :func:`log_weights` -- so
    :math:`g_R(\varphi) = \varphi/\varphi_{\mathrm{fid}}` and
    :math:`g_F(\varphi) = 1`.

    ``H0`` enters the same rate integral only through
    :func:`~astrogwb.cosmology.distance_and_volume_grid`'s differential
    comoving volume, ``4 pi * comoving_distance**2 * inv_e / h0 * c``, with
    ``comoving_distance = c/h0 * integral(inv_e)`` itself :math:`\propto 1/h_0`
    (:mod:`astrogwb.cosmology`, ``distance_and_volume_grid``): two powers from
    the squared distance plus one explicit ``1/h0`` give
    :math:`\mathrm{d}V_c/\mathrm{d}z \propto h_0^{-3}`, and since that factor
    divides out of the normalized redshift ``logpdf``, it is the only
    ``H0``-dependence of ``total_merger_rate``, so
    :math:`g_R(\varphi) = (\varphi_{\mathrm{fid}}/\varphi)^3`. The mean energy
    flux -- ``exp(log_weights)`` in the spectral density contraction -- picks
    up ``H0`` only through the ``-2 * logdiff_dl_gw`` term in
    :func:`log_weights`, via the target luminosity distance
    :math:`d_L \propto 1/h_0` against the fixed fiducial catalog distance:
    :math:`\exp(-2\log d_L(h_0)) \propto h_0^2`, so
    :math:`g_F(\varphi) = (\varphi/\varphi_{\mathrm{fid}})^2`.

    Parameters
    ----------
    parameter:
        One of :data:`AMPLITUDE_PARAMETERS`.
    fiducial:
        Reference value :math:`\varphi_{\mathrm{fid}}` that defines the
        template.

    Returns
    -------
    AmplitudeScalings
        The ``(merger_rate, mean_energy_flux)`` pair of scaling functions.

    Raises
    ------
    ValueError
        If ``parameter`` is not one of :data:`AMPLITUDE_PARAMETERS`.
    """
    if parameter == "local_merger_rate":
        return AmplitudeScalings(
            merger_rate=lambda marginalized_parameter: (
                marginalized_parameter / fiducial
            ),
            mean_energy_flux=lambda marginalized_parameter: jnp.ones_like(
                marginalized_parameter
            ),
        )
    if parameter == "H0":
        return AmplitudeScalings(
            merger_rate=lambda marginalized_parameter: (
                (fiducial / marginalized_parameter) ** 3
            ),
            mean_energy_flux=lambda marginalized_parameter: (
                (marginalized_parameter / fiducial) ** 2
            ),
        )
    raise ValueError(
        f"{parameter!r} is not one of the amplitude parameters {AMPLITUDE_PARAMETERS}"
    )


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
