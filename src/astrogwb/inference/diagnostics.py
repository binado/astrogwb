r"""What the dense Gaussian likelihood says about the amplitude of the background.

With the detector noise :math:`\Sigma` and the expected data
:math:`\boldsymbol\mu` of a network, the noise-weighted template
:math:`\mathbf c = \Sigma^{-1}\boldsymbol\mu` gives the squared SNR
:math:`\rho^2 = \mathbf c^T \boldsymbol\mu` and the amplitude estimator
:math:`\hat a = \mathbf c^T \mathbf d / \rho^2`, unbiased with variance
:math:`\rho^{-2} + \sigma_{\rm shot}^2` under the catalog covariance
:math:`K` (:func:`~astrogwb.inference.gaussian.catalog_covariance`):

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

Every function takes a :class:`~astrogwb.inference.model.Prediction` and its
:class:`~astrogwb.inference.gaussian.Network`, so one prediction serves every
network; :func:`amplitude_cumulants` reaches into the catalog itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .gaussian import Network, catalog_covariance, expected_data
from .model import CatalogModel, Prediction

__all__ = [
    "AmplitudeCumulants",
    "amplitude_cumulants",
    "amplitude_template",
    "shape_correction",
    "shot_noise_variance",
    "whitened_catalog_covariance",
]


def amplitude_template(
    prediction: Prediction, network: Network
) -> tuple[jax.Array, jax.Array]:
    r"""The template :math:`\mathbf c = \Sigma^{-1}\boldsymbol\mu`, ``(D,)``, and :math:`\rho^2`.

    Parameters
    ----------
    prediction
        The model's prediction.
    network
        The detector network.
    """
    mean = expected_data(prediction, network)
    projector = mean / network.scale**2
    return projector, projector @ mean


def shot_noise_variance(prediction: Prediction, network: Network) -> jax.Array:
    r""":math:`\sigma_{\rm shot}^2`, the catalog's variance of :math:`\hat a`, ``()``.

    Parameters
    ----------
    prediction
        The model's prediction, with a power moment.
    network
        The detector network, with a response moment.
    """
    projector, squared_snr = amplitude_template(prediction, network)
    covariance = catalog_covariance(prediction, network)
    return projector @ covariance @ projector / squared_snr**2


def _whitened(prediction: Prediction, network: Network) -> tuple[jax.Array, jax.Array]:
    """:math:`\\tilde K` and the unit mean direction :math:`\\mathbf u`."""
    covariance = catalog_covariance(prediction, network)
    whitened = covariance / jnp.outer(network.scale, network.scale)
    direction = expected_data(prediction, network) / network.scale
    return whitened, direction / jnp.linalg.norm(direction)


def whitened_catalog_covariance(
    prediction: Prediction, network: Network, *, orthogonal: bool = False
) -> jax.Array:
    r""":math:`\tilde K`, or with ``orthogonal`` its projection orthogonal to the mean, ``(D, D)``.

    The projection :math:`(1 - \mathbf u\mathbf u^T)\tilde K(1 - \mathbf
    u\mathbf u^T)` has :math:`\mathbf u` as a null vector and, on its
    complement, the eigenvalues of :math:`\tilde K_\perp`.

    Parameters
    ----------
    prediction
        The model's prediction, with a power moment.
    network
        The detector network, with a response moment.
    orthogonal
        Whether to project out the mean direction.
    """
    whitened, direction = _whitened(prediction, network)
    if not orthogonal:
        return whitened
    projection = jnp.eye(direction.shape[0]) - jnp.outer(direction, direction)
    return projection @ whitened @ projection


def shape_correction(prediction: Prediction, network: Network) -> jax.Array:
    r""":math:`\delta`, the amplitude information recovered from the spectral shape, ``()``.

    :math:`(1 + \tilde K_\perp)^{-1}\mathbf v` is solved in the full space:
    :math:`1 + P\tilde K P` maps :math:`\mathbf u` to itself and the
    complement to itself, and :math:`\mathbf v` lies in the complement.

    Parameters
    ----------
    prediction
        The model's prediction, with a power moment.
    network
        The detector network, with a response moment.
    """
    whitened, direction = _whitened(prediction, network)
    projection = jnp.eye(direction.shape[0]) - jnp.outer(direction, direction)
    cross = projection @ whitened @ direction
    system = jnp.eye(direction.shape[0]) + projection @ whitened @ projection
    return cross @ jnp.linalg.solve(system, cross)


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
    model: CatalogModel,
    params: Mapping[str, ArrayLike],
    network: Network,
    response_draws: ArrayLike,
) -> AmplitudeCumulants:
    r"""The cumulants of :math:`\hat a` under per-source response draws.

    Each source contributes :math:`\mathbf c^T \mathbf p_i / \rho^2`, with
    :math:`\mathbf p_i` its per-pair cross powers, which under draw :math:`l`
    are its intensity times :math:`(2/5)\, r_{l,ab}(f_k)`. Summing the
    template over the pairs folds every draw into one template on the
    frequency grid, :math:`v_{l,k} = \sum_{ab} c_{ab,k}\, r_{l,ab,k} /
    \rho^2` (:meth:`~astrogwb.inference.model.CatalogModel.projected_cumulants`).
    The shot variance equals :func:`shot_noise_variance` with the response
    moment of the same draws. Costs ``L`` catalog contractions.

    Parameters
    ----------
    model
        The catalog model.
    params
        Model parameter values.
    network
        The detector network.
    response_draws
        Per-pair response draws, ``(L, P, F)``
        (:func:`~astrogwb.detector.response.sample_pair_response`).
    """
    prediction = model.predict(params, shot_noise=False)
    projector, squared_snr = amplitude_template(prediction, network)
    draws = network.compress(response_draws)
    num_bins = prediction.mean.shape[0]
    templates = jax.vmap(
        lambda draw: jnp.zeros(num_bins).at[network.bins].add(projector * draw)
    )(draws)
    second, third = model.projected_cumulants(params, templates / squared_snr)
    return AmplitudeCumulants(1.0 / squared_snr, second, third)
