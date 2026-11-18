r"""Health checks for importance-weighted quantities and catalog predictions.

:func:`relative_ess` answers the question *how many of my samples are
actually doing work?* for reweighting the fixed proposal catalog to sampled
hyperparameters. :func:`amplitude_cumulants` checks the Gaussian truncation of
the likelihood: the skewness a finite catalog gives the amplitude estimator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from astrogwb.inference import (
    DenseGaussianLikelihood,
    amplitude_template,
    response_templates,
)

if TYPE_CHECKING:
    # The model imports relative_ess from this module.
    from .model import CatalogModel

__all__ = ["AmplitudeCumulants", "amplitude_cumulants", "relative_ess"]


def relative_ess(log_weights: jax.Array) -> jax.Array:
    r"""Kish effective sample size as a fraction of the sample count.

    .. math::

        \frac{N_\mathrm{eff}}{N}
        = \frac{\left(\sum_i w_i\right)^2}{N \sum_i w_i^2}

    for :math:`w_i = e^{\ell_i}`. Ranges from ``1/N`` (one weight dominates) to
    ``1`` (all weights equal).

    Evaluated entirely in log space. The equivalent form that exponentiates
    first overflows whenever the log-weights are large in absolute terms, even
    though the ratio itself is invariant under a constant offset of all
    log-weights.

    Parameters
    ----------
    log_weights
        Log importance weights; the effective sample size is computed over the
        trailing axis, so leading batch dimensions are preserved.
    """
    log_count = jnp.log(log_weights.shape[-1])
    log_relative_ess = (
        2.0 * logsumexp(log_weights, axis=-1)
        - log_count
        - logsumexp(2.0 * log_weights, axis=-1)
    )
    return jnp.exp(log_relative_ess)


class AmplitudeCumulants(NamedTuple):
    r"""Cumulants of the amplitude estimator :math:`\hat a`.

    Attributes
    ----------
    noise_variance
        :math:`\rho^{-2}`, the detector noise's variance, ``()``.
    shot_variance
        The catalog's second cumulant, :math:`\sigma_{\rm shot}^2`, ``()``.
    third_cumulant
        The catalog's third cumulant; detector noise adds none, ``()``.
    """

    noise_variance: jax.Array
    shot_variance: jax.Array
    third_cumulant: jax.Array

    @property
    def skewness(self) -> jax.Array:
        r""":math:`\kappa_3 / \kappa_2^{3/2}` of :math:`\hat a` in detector noise."""
        return self.third_cumulant / (self.noise_variance + self.shot_variance) ** 1.5


def amplitude_cumulants(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    params: Mapping[str, ArrayLike],
    response_draws: Mapping[tuple[str, str], ArrayLike],
) -> AmplitudeCumulants:
    r"""The cumulants of :math:`\hat a` under per-source response draws.

    The draws are folded into templates on the frequency grid
    (:func:`~astrogwb.inference.response_templates`) and projected on the
    catalog
    (:meth:`~astrogwb.background.importance.model.CatalogModel.projected_cumulants`).
    The shot variance equals :func:`~astrogwb.inference.shot_noise_variance`
    when the draws sample the same inclination, sky and polarization
    distribution as the connected overlap. Costs ``L`` catalog contractions.

    Parameters
    ----------
    likelihood
        A likelihood of a catalog model; needs no data.
    params
        Model parameter values.
    response_draws
        Per-source pair responses keyed by the sorted detector-name pair, each
        ``(L, F)`` (:func:`~astrogwb.detector.sample_pair_response`).
    """
    _, squared_snr = amplitude_template(likelihood, params)
    templates = response_templates(likelihood, params, response_draws)
    second, third = likelihood.model.projected_cumulants(
        params,
        templates,
        observation_time=likelihood.observation_time,
    )
    return AmplitudeCumulants(1.0 / squared_snr, second, third)
