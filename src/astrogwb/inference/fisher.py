r"""Per-frequency Fisher information of the diagonal Gaussian spectrum likelihood.

:func:`~astrogwb.inference.models.gaussian_gwb_model.gwb_likelihood_model`
compares a predicted spectrum to data bin by bin,
:math:`d_i \sim \mathcal{N}(S_i(\theta),
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

Any :data:`~astrogwb.inference.protocol.SpectrumFn` works, as long as it is
differentiable in the requested parameters. The importance spectrum from
:meth:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood.spectrum` is: it
reweights a fixed catalog without drawing anything. Its derivative is still a
Monte Carlo estimate over the intrinsic draws -- at the catalog's own fiducials
it reduces to the score-function estimator, with noise that grows as
:func:`~astrogwb.importance.diagnostics.relative_ess` drops -- so evaluate near
those fiducials. A hyperparameter that moves a support edge (a hard mass
cutoff, say) differentiates only the density's normalization, not the sources
crossing the edge, so its derivative is missing that boundary term.
:func:`~astrogwb.simulators.spectra.poisson_counts.poisson_counts_forward_model` is not a valid
input: its Poisson count is discrete, and differentiating through its draws
gives one realization's derivative rather than the mean spectrum's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.inference.protocol import SpectrumFn

__all__ = ["fisher_matrix_per_bin", "spectral_density_jacobian"]


def spectral_density_jacobian(
    spectral_density_fn: SpectrumFn,
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
        return spectral_density_fn({**fixed, **free_params})

    columns = jax.jacfwd(spectrum)(free)
    return jnp.stack([columns[name] for name in parameter_names], axis=-1)


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
        ``gaussian_bin_scale(psd, time_years, frequencies)``. ``frequency_mask`` selects
        bins as in
    :class:`~astrogwb.inference.likelihood.Network`:
        an excluded bin contributes exactly zero, even where its scale is
        infinite. Sum over the leading axis for the total Fisher matrix.
    """
    jacobian = spectral_density_jacobian(spectral_density_fn, params, parameter_names)
    inverse_variance = jnp.asarray(scale) ** -2
    if frequency_mask is not None:
        inverse_variance = jnp.where(jnp.asarray(frequency_mask), inverse_variance, 0.0)
    return inverse_variance[:, None, None] * jacobian[:, :, None] * jacobian[:, None, :]
