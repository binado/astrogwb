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

:func:`fisher_svd` is the same decomposition taken on the rescaled whitened
Jacobian :math:`\tilde W = U \Sigma V^\top` instead of on
:math:`\tilde F = \tilde W^\top \tilde W = V \Sigma^2 V^\top`. It works at
the square root of the Fisher matrix's condition number, so a weak mode that
the eigen-decomposition rounds to zero stays finite, and it also returns
:math:`U`: the unit spectral template, in noise units, through which the data
see each mode. :func:`post_newtonian_templates` and
:func:`cumulative_template_fractions` name those templates, by measuring how
much of each a nested basis -- the spectrum itself, then its low-order
frequency corrections -- explains.
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
    "FisherModes",
    "cumulative_template_fractions",
    "derivative_cosine_matrix",
    "fisher_eigenmodes",
    "fisher_from_whitened_jacobian",
    "fisher_matrix_per_bin",
    "fisher_svd",
    "post_newtonian_templates",
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
    modes: FisherEigenmodes | FisherModes, prior_sigmas: Mapping[str, float]
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


class FisherModes(NamedTuple):
    r"""Singular modes of a rescaled whitened Jacobian.

    Mode ``k`` is the combination
    :math:`\sum_a \mathrm{directions}_{ak}\,(\theta_a - \bar\theta_a) / s_a`,
    measured to ``sigmas[k]``, and it changes the whitened spectrum along
    ``templates[:, k]``. The modes are sorted best-constrained first.
    """

    parameter_names: tuple[str, ...]
    #: The reference width :math:`s_a` each parameter was divided by.
    parameter_scales: np.ndarray
    #: Singular values :math:`\Sigma_k`, the square roots of the Fisher
    #: eigenvalues in rescaled units.
    singular_values: np.ndarray
    #: Standard deviation along each mode, :math:`1 / \Sigma_k`; ``inf`` for a
    #: direction the Jacobian carries no information about.
    sigmas: np.ndarray
    #: Unit parameter directions as columns, ``(P, P)``, each signed so its
    #: largest component is positive.
    directions: np.ndarray
    #: Unit spectral templates as columns, ``(F, P)``, signed with their mode.
    templates: np.ndarray


def fisher_svd(
    whitened: ArrayLike,
    parameter_names: Sequence[str],
    *,
    parameter_scales: Sequence[float] | ArrayLike | None = None,
    rcond: float | None = None,
) -> FisherModes:
    r"""Singular value decomposition of a rescaled whitened Jacobian.

    ``whitened`` is ``(F, P)`` from :func:`whitened_jacobian`, with the bins a
    band excludes set to zero. Each column is multiplied by its
    ``parameter_scales`` entry (default one) before the decomposition, for the
    reason :func:`fisher_eigenmodes` gives. The widths and directions agree
    with :func:`fisher_eigenmodes` of the summed Fisher matrix wherever that is
    numerically resolved.

    A singular value at or below ``rcond`` times the largest, by default
    ``max(F, P) * eps`` as :func:`numpy.linalg.matrix_rank` uses, is an
    unconstrained direction and gets ``sigma = inf``. Because this works on
    the Jacobian, that threshold sits at the square root of the Fisher
    matrix's.
    """
    names = tuple(parameter_names)
    whitened = np.asarray(whitened, dtype=float)
    if whitened.ndim != 2 or whitened.shape[1] != len(names):
        raise ValueError(
            f"whitened has shape {whitened.shape}, expected (F, {len(names)}) "
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
    templates, singular_values, directions_t = np.linalg.svd(
        whitened * scales, full_matrices=False
    )
    directions = directions_t.T
    tolerance = (
        np.finfo(float).eps * max(whitened.shape) if rcond is None else rcond
    ) * (float(singular_values[0]) if singular_values.size else 0.0)
    constrained = singular_values > tolerance
    sigmas = np.full(len(names), np.inf)
    sigmas[constrained] = 1.0 / singular_values[constrained]
    leading = np.argmax(np.abs(directions), axis=0)
    signs = np.sign(directions[leading, np.arange(len(names))])
    return FisherModes(
        names,
        scales,
        singular_values,
        sigmas,
        directions * signs,
        templates * signs,
    )


def post_newtonian_templates(
    frequencies: ArrayLike,
    whitened_spectrum: ArrayLike,
    exponents: Sequence[float],
    *,
    reference_frequency: float,
) -> np.ndarray:
    r"""Whitened spectrum times powers of frequency, shape ``(F, len(exponents))``.

    Column ``j`` is :math:`(S_i / \sigma_i)\,(f_i / f_{\mathrm{ref}})^{e_j}`.
    Exponent ``0`` is the spectrum itself -- an overall rescaling -- and
    ``2/3``, ``1``, ``4/3`` are the relative 1PN, 1.5PN and 2PN corrections to
    an inspiral energy spectrum, each proportional to a power of
    :math:`\pi M (1+z) f`. ``reference_frequency`` only keeps the columns
    commensurate; the span does not depend on it.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    spectrum = np.asarray(whitened_spectrum, dtype=float)
    if frequencies.shape != spectrum.shape or frequencies.ndim != 1:
        raise ValueError(
            f"frequencies {frequencies.shape} and whitened_spectrum "
            f"{spectrum.shape} must be matching 1-D arrays"
        )
    ratio = frequencies / float(reference_frequency)
    return np.stack([spectrum * ratio**exponent for exponent in exponents], axis=-1)


def cumulative_template_fractions(vectors: ArrayLike, basis: ArrayLike) -> np.ndarray:
    r"""Fraction of each vector's squared norm spanned by the first ``j`` basis columns.

    ``vectors`` is ``(F, K)`` and ``basis`` ``(F, B)``. Entry ``[k, j]`` is
    :math:`\lVert \Pi_{j} v_k \rVert^2 / \lVert v_k \rVert^2`, with
    :math:`\Pi_j` the orthogonal projector onto the span of basis columns
    ``0 .. j``. Each row is non-decreasing and bounded by one; what the full
    basis leaves unexplained is ``1 - fractions[:, -1]``. Nested, because the
    columns of a frequency expansion are far from orthogonal and a separate
    fraction per column would double-count.
    """
    vectors = np.asarray(vectors, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    if basis.ndim != 2 or basis.shape[0] != vectors.shape[0]:
        raise ValueError(
            f"basis has shape {basis.shape}, expected ({vectors.shape[0]}, B)"
        )
    # Gram-Schmidt through QR: column j of q spans what basis column j adds.
    q, _ = np.linalg.qr(basis)
    norms = np.sum(vectors**2, axis=0)
    captured = np.cumsum((q.T @ vectors) ** 2, axis=0)
    return (captured / np.where(norms > 0.0, norms, np.nan)).T
