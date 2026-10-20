r"""Physical shot noise as a low-rank term of the spectrum's covariance.

The observed spectrum is one catalog realization: a finite Poisson sum over
mergers, which scatters about the expected spectrum :math:`\mathbf{S}` on top
of the detector noise :math:`D = \mathrm{diag}(\sigma^2)`. Modelling that
scatter as one Gaussian mode :math:`\mathbf{u}` with unit variance gives the
covariance :math:`C = D + \mathbf{u}\mathbf{u}^T`, whose inverse and
determinant follow from the Sherman-Morrison identity. With the whitened
residual :math:`x = (\mathbf{d} - \mathbf{S})/\sigma` and direction
:math:`v = \mathbf{u}/\sigma`,

.. math::

    \chi^2 = x^T x - \frac{(x^T v)^2}{1 + v^T v}, \qquad
    \ln\det C = \sum_f \ln\sigma_f^2 + \ln(1 + v^T v).

Two directions are offered:

- *Amplitude* (:func:`amplitude_direction`): :math:`\mathbf{u} = s\,\mathbf{S}`,
  a multiplicative nuisance :math:`1 + \epsilon` with an informative prior
  :math:`\epsilon \sim \mathcal{N}(0, s^2)`, marginalized. Along the template
  this is a Gaussian in the amplitude estimator with variance
  :math:`\rho^{-2} + A^2 s^2`; the shape residual is unchanged. The likelihood
  takes :math:`s^2` as the exact variance of the catalog projected on the
  template
  (:meth:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood.relative_variance`);
  :func:`amplitude_shot_noise_variance` is its upper bound from the per-bin
  variances alone.
- *Per frequency* (:func:`per_frequency_direction`):
  :math:`u_f = \sqrt{V_f}`, the right variance in every bin and still perfectly
  correlated across bins. It equals the amplitude direction wherever
  :math:`\sqrt{V_f}/S_f` is flat, and is a cross-check where a parameter tilts
  the template.

:math:`V_f` is the per-bin variance a
:class:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood` predicts.

The rank-one term generalizes to :math:`k` modes. With an orthonormal basis
:math:`U` of the whitened catalog covariance's leading eigenvectors
(:func:`shot_noise_modes`) and the shot noise :math:`B` in that basis, the
whitened covariance is :math:`I + U B U^T`, and Woodbury gives

.. math::

    \chi^2 = x^T x - y^T B (I + B)^{-1} y, \qquad
    \ln\det C = \sum_f \ln\sigma_f^2 + \ln\det(I + B), \qquad
    y = U^T x,

at the cost of a :math:`k \times k` Cholesky factor
(:func:`low_rank_gaussian_log_likelihood`). The amplitude mode is :math:`k = 1`
with :math:`U` the whitened template direction and :math:`B = \rho^2 s^2`.

Everything broadcasts over leading dimensions and contracts over the trailing
frequency axis.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "amplitude_direction",
    "amplitude_shot_noise_variance",
    "low_rank_gaussian_log_likelihood",
    "pad_modes",
    "per_frequency_direction",
    "rank_one_gaussian_log_likelihood",
    "shot_noise_modes",
]


def _keep(frequency_mask: ArrayLike | None, like: jax.Array) -> jax.Array:
    """The bins counted, broadcast against ``like``."""
    if frequency_mask is None:
        return jnp.ones(jnp.shape(like), dtype=bool)
    return jnp.broadcast_to(jnp.asarray(frequency_mask), jnp.shape(like))


def amplitude_shot_noise_variance(
    spectrum: ArrayLike,
    variance: ArrayLike,
    scale: ArrayLike,
    frequency_mask: ArrayLike | None = None,
) -> jax.Array:
    r"""The relative amplitude variance :math:`s^2` of the shot noise.

    .. math::

        s = \frac{\sum_f w_f \sqrt{V_f} / S_f}{\sum_f w_f}, \qquad
        w_f = \frac{S_f^2}{\sigma_f^2},

    the SNR-weighted mean relative standard deviation over the counted bins
    with signal. It is exact when the shot noise is perfectly correlated
    across bins, and otherwise an upper bound on the variance of the
    amplitude estimator (Cauchy-Schwarz), which
    :meth:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood.relative_variance`
    computes exactly from the catalog.

    Parameters
    ----------
    spectrum, variance:
        The expected spectrum :math:`S` and its per-bin shot-noise variance
        :math:`V`, shape ``(..., F)``.
    scale:
        Per-bin detector standard deviation :math:`\sigma`, ``(F,)``.
    frequency_mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all.

    Returns
    -------
    jax.Array
        :math:`s^2`, shape ``(...)``.
    """
    spectrum = jnp.asarray(spectrum)
    variance = jnp.asarray(variance)
    usable = _keep(frequency_mask, spectrum) & (spectrum > 0.0)
    safe_spectrum = jnp.where(usable, spectrum, 1.0)
    weight = jnp.where(usable, (safe_spectrum / jnp.asarray(scale)) ** 2, 0.0)
    ratio = jnp.sqrt(jnp.where(usable, variance, 0.0)) / safe_spectrum
    scatter = jnp.sum(weight * ratio, axis=-1) / jnp.sum(weight, axis=-1)
    return scatter**2


def amplitude_direction(spectrum: ArrayLike, relative_variance: ArrayLike) -> jax.Array:
    r""":math:`\mathbf{u} = s\,\mathbf{S}`, the shot noise as an amplitude mode.

    ``relative_variance`` is :math:`s^2`, shape ``(...)``, from
    :func:`amplitude_shot_noise_variance` at the same point or fixed.
    """
    return jnp.sqrt(jnp.asarray(relative_variance))[..., None] * jnp.asarray(spectrum)


def per_frequency_direction(variance: ArrayLike) -> jax.Array:
    r""":math:`u_f = \sqrt{V_f}`, the shot noise with its per-bin variance."""
    return jnp.sqrt(jnp.asarray(variance))


def rank_one_gaussian_log_likelihood(
    observed: ArrayLike,
    prediction: ArrayLike,
    scale: ArrayLike,
    direction: ArrayLike,
    frequency_mask: ArrayLike | None = None,
) -> jax.Array:
    r"""Gaussian log-likelihood with covariance :math:`D + \mathbf{u}\mathbf{u}^T`.

    Evaluated with the Sherman-Morrison identity on whitened quantities, so it
    stays finite for per-bin scales many orders of magnitude from one. A zero
    ``direction`` gives the diagonal Gaussian exactly.

    Parameters
    ----------
    observed, prediction:
        Data and expected spectrum, ``(..., F)``.
    scale:
        Per-bin detector standard deviation, ``(F,)``.
    direction:
        The shot-noise mode :math:`\mathbf{u}`, ``(..., F)``.
    frequency_mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all. Excluded
        bins contribute exactly zero, so the result equals evaluating on the
        arrays compressed to the selection, and a masked bin's ``scale`` may
        be infinite.

    Returns
    -------
    jax.Array
        Shape ``(...)``.
    """
    scale = jnp.asarray(scale)
    residual = jnp.asarray(observed) - jnp.asarray(prediction)
    direction = jnp.asarray(direction)
    residual, direction = jnp.broadcast_arrays(residual, direction)
    keep = _keep(frequency_mask, residual)
    safe_scale = jnp.where(keep, scale, 1.0)
    whitened = jnp.where(keep, residual / safe_scale, 0.0)
    mode = jnp.where(keep, direction / safe_scale, 0.0)
    mode_norm = jnp.sum(mode * mode, axis=-1)
    projection = jnp.sum(whitened * mode, axis=-1)
    chi_squared = jnp.sum(whitened * whitened, axis=-1) - projection**2 / (
        1.0 + mode_norm
    )
    log_scale = jnp.where(keep, jnp.log(safe_scale) + 0.5 * jnp.log(2.0 * jnp.pi), 0.0)
    return -0.5 * chi_squared - jnp.sum(log_scale, axis=-1) - 0.5 * jnp.log1p(mode_norm)


def shot_noise_modes(
    covariance: ArrayLike,
    scale: ArrayLike,
    frequency_mask: ArrayLike | None = None,
    *,
    tolerance: float = 1e-3,
    max_modes: int = 32,
) -> jax.Array:
    r"""The leading eigenvectors of a whitened catalog covariance, ``(F, k)``.

    The whitened covariance :math:`K = \sigma^{-1} \mathrm{Cov}\, \sigma^{-1}`
    over the counted bins has eigenvalues :math:`\lambda_1 \geq \lambda_2
    \geq \dots` in units of the detector-noise variance. The fewest leading
    eigenvectors are kept such that the dropped ones satisfy

    .. math::

        \sum_{i > k} \max(\lambda_i, 0) < \epsilon,

    with :math:`\epsilon` = ``tolerance``. Dropping a mode of eigenvalue
    :math:`\lambda` changes the log-likelihood by about :math:`\lambda / 2` (its
    log-determinant and its expected :math:`\chi^2`), so the expected
    :math:`|\Delta \ln \mathcal{L}|` from the truncation is about
    :math:`\epsilon / 2`, whatever :math:`\lambda_1`. At least one mode is kept,
    and at most ``max_modes``; if the cap stops the selection before the
    budget is met, a :class:`UserWarning` reports the dropped sum and the
    :math:`k` the budget needs. The columns are orthonormal over the counted
    bins and zero elsewhere, ordered by decreasing eigenvalue. Call outside JAX
    transformations; :math:`k` is a host integer.

    Parameters
    ----------
    covariance:
        The catalog covariance, ``(F, F)``, typically
        :meth:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood.catalog_covariance`
        at the fiducials under the network.
    scale:
        Per-bin detector standard deviation :math:`\sigma`, ``(F,)``.
    frequency_mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all.
    tolerance:
        The budget :math:`\epsilon` on the dropped eigenvalues, in units of the
        detector-noise variance.
    max_modes:
        The cap on :math:`k`.

    Returns
    -------
    jax.Array
        The basis :math:`U`, ``(F, k)``.
    """
    covariance = np.asarray(covariance, dtype=float)
    scale = np.asarray(scale, dtype=float)
    keep = np.isfinite(scale) & (scale > 0.0)
    if frequency_mask is not None:
        keep &= np.asarray(frequency_mask, dtype=bool)
    kept_scale = scale[keep]
    whitened = covariance[np.ix_(keep, keep)] / np.outer(kept_scale, kept_scale)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    # dropped[k]: the sum of the positive eigenvalues past the first k.
    positive = np.clip(eigenvalues, 0.0, None)
    dropped = np.concatenate([np.cumsum(positive[::-1])[::-1], [0.0]])
    within = np.flatnonzero(dropped < tolerance)
    needed = max(int(within[0]) if within.size else positive.size, 1)
    num_modes = min(needed, max_modes)
    if needed > max_modes:
        warnings.warn(
            f"max_modes = {max_modes} drops a whitened eigenvalue sum of "
            f"{dropped[num_modes]:.3e} >= tolerance = {tolerance:.3e}; "
            f"the budget needs k = {needed}",
            stacklevel=2,
        )
    basis = np.zeros((scale.size, num_modes))
    basis[keep] = eigenvectors[:, :num_modes]
    return jnp.asarray(basis)


def pad_modes(modes: ArrayLike, num_modes: int) -> jax.Array:
    """Pad a basis with zero columns to ``num_modes``, ``(F, num_modes)``.

    Zero columns are exactly inert in :func:`low_rank_gaussian_log_likelihood`,
    so bases of several networks can be stacked at one :math:`k`.
    """
    modes = jnp.asarray(modes)
    return jnp.pad(modes, ((0, 0), (0, num_modes - modes.shape[-1])))


def low_rank_gaussian_log_likelihood(
    observed: ArrayLike,
    prediction: ArrayLike,
    scale: ArrayLike,
    modes: ArrayLike,
    mode_covariance: ArrayLike,
    frequency_mask: ArrayLike | None = None,
) -> jax.Array:
    r"""Gaussian log-likelihood with whitened covariance :math:`I + U B U^T`.

    The covariance is :math:`\sigma (I + U B U^T) \sigma`, with :math:`U` =
    ``modes`` orthonormal over the counted bins (:func:`shot_noise_modes`) and
    :math:`B` = ``mode_covariance`` positive semidefinite. Evaluated with
    Woodbury and a Cholesky factor of :math:`I + B` on whitened quantities, as
    :func:`rank_one_gaussian_log_likelihood` is. A zero column of :math:`U`
    with its zero row and column of :math:`B` contributes exactly nothing.

    Parameters
    ----------
    observed, prediction:
        Data and expected spectrum, ``(F,)``.
    scale:
        Per-bin detector standard deviation, ``(F,)``.
    modes:
        The basis :math:`U`, ``(F, k)``.
    mode_covariance:
        The whitened shot-noise covariance in the basis, :math:`B`, ``(k, k)``.
    frequency_mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all. Excluded
        bins contribute exactly zero, and their ``scale`` may be infinite.

    Returns
    -------
    jax.Array
        Shape ``()``.
    """
    scale = jnp.asarray(scale)
    residual = jnp.asarray(observed) - jnp.asarray(prediction)
    keep = _keep(frequency_mask, residual)
    safe_scale = jnp.where(keep, scale, 1.0)
    whitened = jnp.where(keep, residual / safe_scale, 0.0)
    modes = jnp.where(keep[:, None], jnp.asarray(modes), 0.0)
    mode_covariance = jnp.asarray(mode_covariance)
    projection = modes.T @ whitened
    factor = jnp.linalg.cholesky(jnp.eye(mode_covariance.shape[-1]) + mode_covariance)
    solved = jax.scipy.linalg.solve_triangular(factor, projection, lower=True)
    chi_squared = whitened @ whitened - (projection @ projection - solved @ solved)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(factor)))
    log_scale = jnp.where(keep, jnp.log(safe_scale) + 0.5 * jnp.log(2.0 * jnp.pi), 0.0)
    return -0.5 * chi_squared - jnp.sum(log_scale) - 0.5 * log_det
