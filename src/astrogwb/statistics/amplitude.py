r"""The amplitude statistic of the dense Gaussian likelihood.

When the parameters only rescale the expected data, :math:`\boldsymbol\mu =
x\,\boldsymbol\mu_0` with a dimensionless amplitude :math:`x`. With the total
covariance :math:`\Sigma` held at the template's parameters, the Gaussian
likelihood gives the estimator

.. math::

    \hat x = \mathbf w^\top \bar{\mathbf C}, \qquad
    \mathbf w = \frac{\Sigma^{-1}\boldsymbol\mu_0}
        {\boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0},

unbiased, :math:`\langle\hat x\rangle = x`, with relative variance

.. math::

    \frac{\mathrm{Var}[\hat x]}{x^2}
        = \frac{1}{\boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0}.

Without shot noise :math:`\Sigma` is the detector noise and the variance is
:math:`\rho^{-2}`, the inverse squared SNR. Each term :math:`\Sigma^X` of the
covariance contributes :math:`\mathbf w^\top\Sigma^X\mathbf w`, and these sum
to the variance (:func:`amplitude_variance_budget`).

Every function takes a :class:`~astrogwb.inference.DenseGaussianLikelihood`,
which holds the model and the network's covariance, and the template's
parameters ``params`` (typically the fiducials).
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
from jax.typing import ArrayLike

from astrogwb.inference import DenseGaussianLikelihood, SpectralDensityModel

__all__ = [
    "amplitude",
    "amplitude_variance",
    "amplitude_variance_budget",
]


def amplitude_weights[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> jax.Array:
    r"""The weights :math:`\mathbf w = \Sigma^{-1}\boldsymbol\mu_0 / \boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0`, ``(D,)``.

    Parameters
    ----------
    likelihood
        The likelihood; needs no data.
    params
        The template's parameter values.
    """
    _, factor = likelihood.evaluate(params)
    projected = factor.solve(factor.mean)
    return projected / (projected @ factor.mean)


def amplitude_variance[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> jax.Array:
    r"""The relative variance :math:`1 / \boldsymbol\mu_0^\top\Sigma^{-1}\boldsymbol\mu_0` of :math:`\hat x`, ``()``.

    Parameters
    ----------
    likelihood
        The likelihood; needs no data.
    params
        The template's parameter values.
    """
    _, factor = likelihood.evaluate(params)
    whitened = factor.whiten(factor.mean)
    return 1.0 / (whitened @ whitened)


def amplitude[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> jax.Array:
    r"""The estimate :math:`\hat x = \mathbf w^\top \bar{\mathbf C}` of each realization, ``(...)``.

    Parameters
    ----------
    likelihood
        The likelihood, with its observed data ``(..., D)``.
    params
        The template's parameter values.

    Raises
    ------
    ValueError
        If the likelihood holds no data.
    """
    if likelihood.observed is None:
        raise ValueError("the likelihood was built without observed data")
    return likelihood.observed @ amplitude_weights(likelihood, params)


def amplitude_variance_budget[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> dict[str, jax.Array]:
    r"""The variance :math:`\mathbf w^\top\Sigma^X\mathbf w` of each covariance term, ``()`` each.

    Keyed as :meth:`~astrogwb.inference.DenseGaussianLikelihood.covariance_components`;
    the terms sum to :func:`amplitude_variance`. Each term's share of the sum
    is also the fractional change in the variance per fractional change in
    that term.

    Parameters
    ----------
    likelihood
        The likelihood, built with ``shot_noise`` and ``keep_generalized_orfs``;
        needs no data.
    params
        The template's parameter values.
    """
    weights = amplitude_weights(likelihood, params)
    return {
        name: weights @ covariance @ weights
        for name, covariance in likelihood.covariance_components(params).items()
    }
