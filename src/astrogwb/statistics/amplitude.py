r"""The amplitude statistic of the dense Gaussian likelihood.

When the parameters only rescale the expected data, :math:`\boldsymbol\mu =
x\,\boldsymbol\mu_0` with a dimensionless amplitude :math:`x`. With the total
covariance :math:`\Sigma` held at the template's parameters, the Gaussian
likelihood gives the estimator

.. math::

    \hat x = \mathbf w^\top \bar{\mathbf C}, \qquad
    \mathbf w = \frac{\Sigma^{-1}\boldsymbol\mu_0}
        {\boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0}.

It is unbiased for any weights with :math:`\mathbf w^\top\boldsymbol\mu_0 = 1`,
:math:`\langle\hat x\rangle = x`, and under a covariance :math:`\Sigma'` its
relative variance is :math:`\mathbf w^\top\Sigma'\mathbf w`. With
:math:`\Sigma' = \Sigma` this is

.. math::

    \frac{\mathrm{Var}[\hat x]}{x^2}
        = \frac{1}{\boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0},

the inverse squared SNR :math:`\rho^{-2}` without shot noise. The variance is
linear in :math:`\Sigma'`, so each term :math:`\Sigma^X` of
:meth:`~astrogwb.inference.DenseGaussianLikelihood.covariance_components`
contributes :math:`\mathbf w^\top\Sigma^X\mathbf w`, and these sum to the
variance.

Only :func:`background_amplitude_weights` takes the likelihood; the estimate
and the variance take arrays.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.inference import DenseGaussianLikelihood, SpectralDensityModel

__all__ = [
    "background_amplitude_estimate",
    "background_amplitude_variance",
    "background_amplitude_weights",
]


def background_amplitude_weights[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> jax.Array:
    r"""The weights :math:`\mathbf w = \Sigma^{-1}\boldsymbol\mu_0 / \boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0`, ``(D,)``.

    Parameters
    ----------
    likelihood
        The likelihood; needs no data.
    params
        The template's parameter values, typically the fiducials.
    """
    mean = likelihood.mean(params)
    projected = likelihood.solve(mean, params)
    return projected / (projected @ mean)


def background_amplitude_estimate(weights: ArrayLike, observed: ArrayLike) -> jax.Array:
    r"""The estimate :math:`\hat x = \mathbf w^\top \bar{\mathbf C}` of each realization, ``(...)``.

    Parameters
    ----------
    weights
        The weights, ``(D,)``, from :func:`background_amplitude_weights`.
    observed
        The observed data, ``(..., D)``, on the same entries as ``weights``.
    """
    return jnp.asarray(observed) @ jnp.asarray(weights)


def background_amplitude_variance(
    weights: ArrayLike, covariance: ArrayLike
) -> jax.Array:
    r"""The relative variance :math:`\mathbf w^\top\Sigma'\mathbf w` of the estimator, ``()``.

    This is the variance of the unbiased estimator with these ``weights``
    when the data have covariance ``covariance``. It equals
    :math:`1 / \boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0` when the
    weights were built from that same covariance.

    Parameters
    ----------
    weights
        The weights, ``(D,)``, from :func:`background_amplitude_weights`.
    covariance
        The data's covariance, ``(D, D)``: the likelihood's
        :meth:`~astrogwb.inference.DenseGaussianLikelihood.covariance`, or one
        of its
        :meth:`~astrogwb.inference.DenseGaussianLikelihood.covariance_components`
        for that term's contribution.
    """
    weights = jnp.asarray(weights)
    return weights @ jnp.asarray(covariance) @ weights
