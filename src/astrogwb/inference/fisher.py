r"""Fisher information of the Gaussian spectrum likelihoods.

A diagonal Gaussian compares a predicted spectrum to data bin by bin,
:math:`d_i \sim \mathcal{N}(S_i(\theta), \sigma_i)`, with a scale that does
not depend on :math:`\theta`. Its Fisher matrix is therefore a sum of per-bin
terms,

.. math::

    F_{ab} = \sum_i F_{ab,i}, \qquad
    F_{ab,i} = \frac{\partial_a S_i \, \partial_b S_i}{\sigma_i^2},

the matrix generalization of
:func:`~astrogwb.background.snr.spectral_snr_squared_per_bin`.
:func:`fisher_matrix_per_bin` returns the terms unsummed so the information
can be attributed to frequency bands; ``.sum(axis=0)`` is the total.

Each :math:`F_{\cdot\cdot,i}` is an outer product, hence rank one: a single bin
constrains one direction in parameter space. Per-bin diagonals, or a
cumulative sum over bins, are meaningful band diagnostics; the inverse of one
bin's matrix is not.

Any differentiable :data:`SpectrumFn` works. For the importance spectrum
(:meth:`~astrogwb.inference.model.CatalogModel.spectrum`) the derivative is a score-function Monte Carlo estimate whose noise grows as
:func:`~astrogwb.importance.diagnostics.relative_ess` drops, so evaluate near
the catalog's fiducials. A hyperparameter that moves a support edge misses the
boundary term.

:func:`dense_fisher_matrix` is the Fisher matrix of the dense Gaussian
likelihood (:mod:`astrogwb.inference.gaussian`), whose covariance also depends
on the parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.inference.gaussian import (
    DenseGaussianLikelihood,
    catalog_covariance,
    expected_data,
)

__all__ = [
    "DenseFisher",
    "SpectrumFn",
    "dense_fisher_matrix",
    "fisher_matrix_per_bin",
    "spectral_density_jacobian",
]

#: A spectrum on the observed grid: ``params -> (F,)``.
type SpectrumFn = Callable[[Mapping[str, ArrayLike]], jax.Array]


def spectral_density_jacobian(
    spectral_density_fn: SpectrumFn,
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
) -> jax.Array:
    r""":math:`\partial S_i / \partial \theta_a` at ``params``, shape ``(F, P)``.

    Columns follow ``parameter_names``; every other entry of ``params`` is held
    fixed. Forward mode, since there are few parameters and many bins.

    Parameters
    ----------
    spectral_density_fn
        Function returning strain spectral density.
    params
        Model parameter values.
    parameter_names
        Names of parameters to differentiate.
    """
    return _jacobian(
        lambda full: (spectral_density_fn(full),), params, parameter_names
    )[0]


def fisher_matrix_per_bin(
    spectral_density_fn: SpectrumFn,
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
    *,
    scale: jax.Array,
    frequency_mask: jax.Array | None = None,
) -> jax.Array:
    r"""Per-bin Fisher matrices :math:`F_{ab,i}`, shape ``(F, P, P)``.

    ``scale`` is the per-bin standard deviation the likelihood uses, normally
    ``gaussian_bin_scale(psd, time_years, frequencies)``. ``frequency_mask``
    selects bins: an excluded bin contributes exactly zero, even where its
    scale is infinite.

    Parameters
    ----------
    spectral_density_fn
        Function returning strain spectral density.
    params
        Model parameter values.
    parameter_names
        Names of parameters to differentiate.
    scale
        Noise standard deviation.
    frequency_mask
        Mask of frequency bins to include.
    """
    jacobian = spectral_density_jacobian(spectral_density_fn, params, parameter_names)
    inverse_variance = jnp.asarray(scale) ** -2
    if frequency_mask is not None:
        inverse_variance = jnp.where(jnp.asarray(frequency_mask), inverse_variance, 0.0)
    return inverse_variance[:, None, None] * jacobian[:, :, None] * jacobian[:, None, :]


class DenseFisher(NamedTuple):
    r"""The two terms of a dense Gaussian's Fisher matrix, each ``(P, P)``.

    Attributes
    ----------
    mean
        :math:`\partial_a \boldsymbol\mu^T C^{-1} \partial_b \boldsymbol\mu`.
    covariance
        :math:`\tfrac12 \mathrm{Tr}[C^{-1} \partial_a C\, C^{-1} \partial_b C]`.
    """

    mean: jax.Array
    covariance: jax.Array

    @property
    def total(self) -> jax.Array:
        """The Fisher matrix, ``(P, P)``."""
        return self.mean + self.covariance


def dense_fisher_matrix(
    likelihood: DenseGaussianLikelihood,
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
) -> DenseFisher:
    r"""The Fisher matrix of ``likelihood`` at ``params``, split into its two terms.

    For a Gaussian whose mean and covariance both depend on the parameters,

    .. math::

        F_{ab} = \partial_a \boldsymbol\mu^T C^{-1} \partial_b \boldsymbol\mu
            + \tfrac12 \mathrm{Tr}\!\left[C^{-1} \partial_a C\,
              C^{-1} \partial_b C\right],

    keeping the terms apart so the covariance term can be checked against the
    mean term. Forward mode, holding ``P`` covariance derivatives of shape
    ``(D, D)``. The derivative caveats of the module docstring apply.

    Parameters
    ----------
    likelihood
        The likelihood; its data are not used.
    params
        Model parameter values.
    parameter_names
        Names of parameters to differentiate.
    """
    network = likelihood.network

    def moments(full: Mapping[str, ArrayLike]) -> tuple[jax.Array, jax.Array]:
        prediction = likelihood.model.predict(full, shot_noise=likelihood.shot_noise)
        covariance = jnp.diag(network.scale**2)
        if likelihood.shot_noise:
            covariance = covariance + catalog_covariance(prediction, network)
        return expected_data(prediction, network), covariance

    mean_derivative, covariance_derivative = _jacobian(moments, params, parameter_names)
    _, covariance = moments(params)
    factor = jax.scipy.linalg.cho_factor(covariance, lower=True)
    whitened_mean = jax.scipy.linalg.cho_solve(factor, mean_derivative)
    # C^{-1} dC_a for every parameter, (P, D, D).
    whitened_covariance = jax.vmap(
        lambda derivative: jax.scipy.linalg.cho_solve(factor, derivative),
        in_axes=-1,
    )(covariance_derivative)
    return DenseFisher(
        mean_derivative.T @ whitened_mean,
        0.5 * jnp.einsum("aij,bji->ab", whitened_covariance, whitened_covariance),
    )


def _jacobian(
    fn: Callable[[Mapping[str, ArrayLike]], tuple[jax.Array, ...]],
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
) -> tuple[jax.Array, ...]:
    """Forward-mode derivatives of each of ``fn``'s outputs, parameters on a trailing axis."""
    missing = sorted(set(parameter_names) - params.keys())
    if missing:
        raise KeyError(f"parameters {missing} are not in params")
    free = {name: jnp.asarray(params[name], dtype=float) for name in parameter_names}
    fixed = {
        name: value for name, value in params.items() if name not in parameter_names
    }
    columns = jax.jacfwd(lambda free_params: fn({**fixed, **free_params}))(free)
    return tuple(
        jnp.stack([output[name] for name in parameter_names], axis=-1)
        for output in columns
    )
