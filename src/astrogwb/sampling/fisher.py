r"""Per-frequency Fisher information of the diagonal Gaussian spectrum likelihood.

:func:`~astrogwb.sampling.models.gwb_spectral_density_model` compares a
predicted spectrum to data bin by bin, :math:`d_i \sim \mathcal{N}(S_i(\theta),
\sigma_i)`, with a scale that does not depend on :math:`\theta`. Its Fisher
matrix is therefore a sum of per-bin terms,

.. math::

    F_{ab} = \sum_i F_{ab,i}, \qquad
    F_{ab,i} = \frac{\partial_a S_i \, \partial_b S_i}{\sigma_i^2},

the matrix generalization of
:func:`~astrogwb.gwb.snr.spectral_snr_squared_per_bin`.
:func:`fisher_matrix_per_bin` returns the terms unsummed so the information
can be attributed to frequency bands; ``.sum(axis=0)`` is the total.

Each :math:`F_{\cdot\cdot,i}` is an outer product, hence rank one: a single bin
constrains one direction in parameter space. Per-bin diagonals, or a
cumulative sum over bins, are meaningful band diagnostics; the inverse of one
bin's matrix is not.

Any :class:`~astrogwb.sampling.SpectralDensityFn` works, as long as it is
differentiable in the requested parameters. The importance spectrum from
:func:`~astrogwb.importance.spectral.build_importance_spectrum` is: it
reweights a fixed catalog without drawing anything. Its derivative is still a
Monte Carlo estimate -- at the catalog's own fiducials it reduces to the
score-function estimator, with noise that grows as
:func:`~astrogwb.importance.diagnostics.relative_ess` drops -- so evaluate near
those fiducials. A hyperparameter that moves a support edge (a hard mass
cutoff, say) differentiates only the density's normalization, not the sources
crossing the edge, so its derivative is missing that boundary term.
:func:`~astrogwb.sampling.forward_model.gwb_forward_model` is not a valid input:
its Poisson count is discrete, and differentiating through its draws gives one
realization's derivative rather than the mean spectrum's.

Two views of the same matrix expose its degeneracies.
:func:`whitened_jacobian` returns the per-bin derivatives in noise units,
:math:`w_{ia} = \partial_a S_i / \sigma_i`. :math:`F_{ab} = \sum_i w_{ia} w_{ib}`,
so :func:`derivative_cosine_matrix` -- the cosine between two of those curves
-- is exactly how alike two parameters bend the spectrum; a cosine near
:math:`\pm 1` is a degeneracy. :func:`fisher_eigenmodes` diagonalizes the
matrix after rescaling each parameter by a reference width, giving the
combinations the likelihood constrains and how well; without that rescaling
the eigenvectors would depend on the parameters' units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from astrogwb.sampling.protocol import SpectralDensityFn

__all__ = [
    "FisherEigenmodes",
    "derivative_cosine_matrix",
    "fisher_eigenmodes",
    "fisher_from_whitened_jacobian",
    "fisher_matrix_per_bin",
    "prior_sigma_along_modes",
    "spectral_density_jacobian",
    "whitened_jacobian",
]


def spectral_density_jacobian(
    spectral_density_fn: SpectralDensityFn,
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
) -> jax.Array:
    r""":math:`\partial S_i / \partial \theta_a` at ``params``, shape ``(F, P)``.

    Columns follow ``parameter_names``; every other entry of ``params`` is held
    fixed. Forward mode, since there are few parameters and many bins.
    """
    missing = sorted(set(parameter_names) - params.keys())
    if missing:
        raise KeyError(f"parameters {missing} are not in params")
    free = {name: jnp.asarray(params[name], dtype=float) for name in parameter_names}
    fixed = {
        name: value for name, value in params.items() if name not in parameter_names
    }

    def spectrum(free_params: dict[str, jax.Array]) -> jax.Array:
        return spectral_density_fn({**fixed, **free_params})[0]

    columns = jax.jacfwd(spectrum)(free)
    return jnp.stack([columns[name] for name in parameter_names], axis=-1)


def whitened_jacobian(
    spectral_density_fn: SpectralDensityFn,
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
    *,
    scale: jax.Array,
    frequency_mask: jax.Array | None = None,
) -> jax.Array:
    r"""Derivatives in noise units, :math:`\partial_a S_i / \sigma_i`, shape ``(F, P)``.

    ``scale`` and ``frequency_mask`` are as for :func:`fisher_matrix_per_bin`;
    an excluded bin is exactly zero, even where its scale is infinite. These
    are the vectors whose inner products are the Fisher matrix, so their
    shapes in frequency show *why* two parameters are degenerate, and their
    squares show *where* the information sits.
    """
    jacobian = spectral_density_jacobian(spectral_density_fn, params, parameter_names)
    whitened = jacobian / jnp.asarray(scale)[:, None]
    if frequency_mask is None:
        return whitened
    return jnp.where(jnp.asarray(frequency_mask)[:, None], whitened, 0.0)


def fisher_from_whitened_jacobian(whitened: ArrayLike) -> jax.Array:
    r"""Per-bin Fisher matrices :math:`w_{ia} w_{ib}` from :func:`whitened_jacobian`.

    Lets a caller that needs both the whitened derivatives and the Fisher
    matrix differentiate once.
    """
    whitened = jnp.asarray(whitened)
    return whitened[:, :, None] * whitened[:, None, :]


def fisher_matrix_per_bin(
    spectral_density_fn: SpectralDensityFn,
    params: Mapping[str, ArrayLike],
    parameter_names: Sequence[str],
    *,
    scale: jax.Array,
    frequency_mask: jax.Array | None = None,
) -> jax.Array:
    r"""Per-bin Fisher matrices :math:`F_{ab,i}`, shape ``(F, P, P)``.

    ``scale`` is the per-bin standard deviation the likelihood uses, normally
    ``gaussian_bin_scale(psd, time_years, df)``. ``frequency_mask`` selects
    bins as in :func:`~astrogwb.sampling.models.gwb_spectral_density_model`:
    an excluded bin contributes exactly zero, even where its scale is
    infinite. Sum over the leading axis for the total Fisher matrix.
    """
    return fisher_from_whitened_jacobian(
        whitened_jacobian(
            spectral_density_fn,
            params,
            parameter_names,
            scale=scale,
            frequency_mask=frequency_mask,
        )
    )


def derivative_cosine_matrix(fisher: ArrayLike) -> np.ndarray:
    r"""Cosine between whitened derivative curves, :math:`F_{ab} / \sqrt{F_{aa} F_{bb}}`.

    ``fisher`` is a summed ``(P, P)`` matrix. An entry near :math:`\pm 1`
    means the two parameters change the spectrum in the same shape, so the
    likelihood cannot tell them apart. For two parameters it is minus the
    forecast correlation. A parameter with no information (a zero diagonal)
    gets NaN in its row and column rather than a division warning.
    """
    fisher = np.asarray(fisher, dtype=float)
    diagonal = np.diag(fisher)
    norms = np.sqrt(np.where(diagonal > 0.0, diagonal, np.nan))
    return fisher / np.outer(norms, norms)


class FisherEigenmodes(NamedTuple):
    r"""Principal axes of a Fisher matrix in rescaled parameters.

    Mode ``k`` is the combination
    :math:`\sum_a \mathrm{directions}_{ak}\,(\theta_a - \bar\theta_a) / s_a`,
    measured to ``sigmas[k]``; the modes are sorted best-constrained first.
    """

    parameter_names: tuple[str, ...]
    #: The reference width :math:`s_a` each parameter was divided by.
    parameter_scales: np.ndarray
    #: Standard deviation along each mode, in units of the scales; ``inf`` for
    #: a direction the matrix carries no information about.
    sigmas: np.ndarray
    #: Unit eigenvectors as columns, ``(P, P)``, each signed so its largest
    #: component is positive.
    directions: np.ndarray


def fisher_eigenmodes(
    fisher: ArrayLike,
    parameter_names: Sequence[str],
    *,
    parameter_scales: Sequence[float] | None = None,
    rcond: float | None = None,
) -> FisherEigenmodes:
    r"""Diagonalize a summed ``(P, P)`` Fisher matrix in rescaled parameters.

    Each parameter is divided by ``parameter_scales`` (default one) before
    the decomposition, :math:`\tilde F_{ab} = s_a s_b F_{ab}`. Choose widths
    that make the parameters commensurate, such as their prior standard
    deviations: the eigenvectors of the raw matrix depend on its units, so
    ``H0`` in km/s/Mpc would otherwise dominate every mode.

    An eigenvalue at or below ``rcond`` times the largest, by default
    ``P * eps`` as :func:`numpy.linalg.matrix_rank` uses, is an unconstrained
    direction and gets ``sigma = inf``.
    """
    names = tuple(parameter_names)
    fisher = np.asarray(fisher, dtype=float)
    if fisher.shape != (len(names), len(names)):
        raise ValueError(
            f"fisher has shape {fisher.shape}, expected {(len(names), len(names))} "
            f"for parameters {names}"
        )
    scales = (
        np.ones(len(names))
        if parameter_scales is None
        else np.asarray(parameter_scales, dtype=float)
    )
    if scales.shape != (len(names),) or np.any(~np.isfinite(scales) | (scales <= 0)):
        raise ValueError(
            f"parameter_scales must be {len(names)} positive finite widths, got {scales}"
        )
    rescaled = fisher * np.outer(scales, scales)
    eigenvalues, directions = np.linalg.eigh(0.5 * (rescaled + rescaled.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, directions = eigenvalues[order], directions[:, order]
    tolerance = (np.finfo(float).eps * len(names) if rcond is None else rcond) * max(
        float(eigenvalues[0]), 0.0
    )
    constrained = eigenvalues > tolerance
    sigmas = np.full(len(names), np.inf)
    sigmas[constrained] = 1.0 / np.sqrt(eigenvalues[constrained])
    leading = np.argmax(np.abs(directions), axis=0)
    signs = np.sign(directions[leading, np.arange(len(names))])
    return FisherEigenmodes(names, scales, sigmas, directions * signs)


def prior_sigma_along_modes(
    modes: FisherEigenmodes, prior_sigmas: Mapping[str, float]
) -> np.ndarray:
    r"""Width of independent Gaussian priors along each mode, in the modes' units.

    :math:`1 / \sqrt{v_k^\top \tilde P v_k}`, with :math:`\tilde P` the
    prior precision in rescaled parameters. Compare with ``modes.sigmas``: a
    mode whose likelihood width exceeds this is prior-dominated. A parameter
    missing from ``prior_sigmas`` has no prior, and a mode no prior touches
    gets ``inf``.
    """
    unknown = sorted(set(prior_sigmas) - set(modes.parameter_names))
    if unknown:
        raise KeyError(f"priors on {unknown} are not in {modes.parameter_names}")
    precision = np.array(
        [
            (scale / float(prior_sigmas[name])) ** 2 if name in prior_sigmas else 0.0
            for name, scale in zip(
                modes.parameter_names, modes.parameter_scales, strict=True
            )
        ]
    )
    projected = np.einsum("ak,a,ak->k", modes.directions, precision, modes.directions)
    with np.errstate(divide="ignore"):
        return np.where(projected > 0.0, 1.0 / np.sqrt(projected), np.inf)
