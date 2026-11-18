r"""What the dense Gaussian likelihood says about the amplitude of the background.

With the detector noise :math:`\Sigma` and the expected data
:math:`\boldsymbol\mu` of a network, the noise-weighted template
:math:`\mathbf c = \Sigma^{-1}\boldsymbol\mu` gives the squared SNR
:math:`\rho^2 = \mathbf c^T \boldsymbol\mu` and the amplitude estimator
:math:`\hat a = \mathbf c^T \mathbf d / \rho^2`, unbiased with variance
:math:`\rho^{-2} + \sigma_{\rm shot}^2` under the catalog covariance
:math:`K`, the likelihood's connected covariance:

.. math::

    \sigma_{\rm shot}^2 = \frac{\mathbf c^T K \mathbf c}{\rho^4}.

The paper's App. ``fisher-amplitude`` whitens with :math:`\Sigma^{-1/2}`:
:math:`\tilde K = \Sigma^{-1/2} K \Sigma^{-1/2}`, the mean direction
:math:`\mathbf u = \Sigma^{-1/2}\boldsymbol\mu/\rho`, the restriction
:math:`\tilde K_\perp` of :math:`\tilde K` orthogonal to :math:`\mathbf u`,
and the cross component :math:`\mathbf v = (1 - \mathbf u\mathbf u^T)\tilde
K\mathbf u`. The amplitude information is reduced from its template value by

.. math::

    \delta = \mathbf v^T (1 + \tilde K_\perp)^{-1} \mathbf v, \qquad
    \boldsymbol\mu^T C^{-1} \boldsymbol\mu
        = \frac{\rho^2}{1 + \rho^2\sigma_{\rm shot}^2 - \delta}.

Every function takes the likelihood and a parameter point; the network is the
likelihood's. :func:`response_templates` folds per-source response draws into
templates on the frequency grid, for a model's projected cumulants
(:func:`~astrogwb.background.importance.diagnostics.amplitude_cumulants`).
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .gaussian import (
    DenseGaussianLikelihood,
    PairName,
    _catalog_covariance,
    _expected_data,
)
from .model import SpectralDensityModel

__all__ = [
    "amplitude_template",
    "response_templates",
    "shape_correction",
    "shot_noise_variance",
    "whitened_catalog_covariance",
]


def amplitude_template[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> tuple[jax.Array, jax.Array]:
    r"""The template :math:`\mathbf c = \Sigma^{-1}\boldsymbol\mu`, ``(D,)``, and :math:`\rho^2`.

    Parameters
    ----------
    likelihood
        The likelihood; needs no data.
    params
        Model parameter values.
    """
    network = likelihood._network
    mean = _expected_data(likelihood.model(params, shot_noise=False), network)
    projector = mean / network.scale**2
    return projector, projector @ mean


def shot_noise_variance[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> jax.Array:
    r""":math:`\sigma_{\rm shot}^2`, the catalog's variance of :math:`\hat a`, ``()``.

    Parameters
    ----------
    likelihood
        The likelihood, with shot noise; needs no data.
    params
        Model parameter values.
    """
    projector, squared_snr = amplitude_template(likelihood, params)
    covariance = _catalog_covariance(
        likelihood.model(params, shot_noise=True), likelihood._network
    )
    return projector @ covariance @ projector / squared_snr**2


def _whitened[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> tuple[jax.Array, jax.Array]:
    """:math:`\\tilde K` and the unit mean direction :math:`\\mathbf u`."""
    network = likelihood._network
    prediction = likelihood.model(params, shot_noise=True)
    covariance = _catalog_covariance(prediction, network)
    whitened = covariance / jnp.outer(network.scale, network.scale)
    direction = _expected_data(prediction, network) / network.scale
    return whitened, direction / jnp.linalg.norm(direction)


def whitened_catalog_covariance[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M],
    params: Mapping[str, ArrayLike],
    *,
    orthogonal: bool = False,
) -> jax.Array:
    r""":math:`\tilde K`, or with ``orthogonal`` its projection orthogonal to the mean, ``(D, D)``.

    The projection :math:`(1 - \mathbf u\mathbf u^T)\tilde K(1 - \mathbf
    u\mathbf u^T)` has :math:`\mathbf u` as a null vector and, on its
    complement, the eigenvalues of :math:`\tilde K_\perp`.

    Parameters
    ----------
    likelihood
        The likelihood, with shot noise; needs no data.
    params
        Model parameter values.
    orthogonal
        Whether to project out the mean direction.
    """
    whitened, direction = _whitened(likelihood, params)
    if not orthogonal:
        return whitened
    projection = jnp.eye(direction.shape[0]) - jnp.outer(direction, direction)
    return projection @ whitened @ projection


def shape_correction[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M], params: Mapping[str, ArrayLike]
) -> jax.Array:
    r""":math:`\delta`, the amplitude information recovered from the spectral shape, ``()``.

    :math:`(1 + \tilde K_\perp)^{-1}\mathbf v` is solved in the full space:
    :math:`1 + P\tilde K P` maps :math:`\mathbf u` to itself and the
    complement to itself, and :math:`\mathbf v` lies in the complement.

    Parameters
    ----------
    likelihood
        The likelihood, with shot noise; needs no data.
    params
        Model parameter values.
    """
    whitened, direction = _whitened(likelihood, params)
    projection = jnp.eye(direction.shape[0]) - jnp.outer(direction, direction)
    cross = projection @ whitened @ direction
    system = jnp.eye(direction.shape[0]) + projection @ whitened @ projection
    return cross @ jnp.linalg.solve(system, cross)


def response_templates[M: SpectralDensityModel](
    likelihood: DenseGaussianLikelihood[M],
    params: Mapping[str, ArrayLike],
    response_draws: Mapping[PairName, ArrayLike],
) -> jax.Array:
    r"""Per-draw templates of :math:`\hat a` on the frequency grid, ``(L, F)``.

    Each source contributes :math:`\mathbf c^T \mathbf p_i / \rho^2`, with
    :math:`\mathbf p_i` its per-pair cross powers, which under draw :math:`l`
    are its intensity times :math:`r_{l,ab}(f_k)`. Summing the template over
    the pairs folds every draw into one template on the frequency grid,

    .. math::

        v_{l,k} = \frac{1}{\rho^2} \sum_{ab} c_{ab,k}\, r_{l,ab,k}.

    Parameters
    ----------
    likelihood
        The likelihood; needs no data.
    params
        Model parameter values.
    response_draws
        Per-source pair responses keyed by the sorted detector-name pair, each
        ``(L, F)`` (:func:`~astrogwb.detector.sample_pair_response` and
        :func:`~astrogwb.detector.pair_names`).
    """
    network = likelihood._network
    projector, squared_snr = amplitude_template(likelihood, params)
    draws = network.compress(network.stack(response_draws))
    templates = jax.vmap(
        lambda draw: jnp.zeros(network.num_bins).at[network.bins].add(projector * draw)
    )(draws)
    return templates / squared_snr
