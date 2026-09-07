r"""Health checks for importance-weighted quantities.

Answers the question *how many of my samples are actually doing work?* for
reweighting the fixed proposal catalog to sampled hyperparameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


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
    log_weights:
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


def power_weighted_relative_ess(
    log_weights: jax.Array, polarization_power: jax.Array
) -> jax.Array:
    r"""Effective sample size of the Monte-Carlo power sum, not of the weights alone.

    ``spectral_density`` estimates each frequency bin as a weighted mean
    ``mean_i(w_i P_{f,i})``, not merely a mean of ``w_i``. Whenever
    ``polarization_power`` is itself heavy-tailed -- e.g. :math:`1/d_L(z)^2`
    blowing up as a source approaches the redshift window's inner edge -- a
    handful of high-power samples can dominate that sum even when the
    importance weights :math:`w_i` are close to uniform. ``relative_ess``
    alone is blind to this: it never sees ``polarization_power``. Folding it
    in by treating :math:`w_i P_{f,i}` as the effective weight recovers the
    quantity that actually governs the sum's Monte-Carlo noise.

    ``log_weights`` broadcasts against ``polarization_power``'s trailing
    (sample) axis, so passing the full ``(frequency, sample)`` array returns
    one effective sample size per frequency bin, and passing a single
    frequency slice (shape ``(sample,)``) returns a scalar.
    """
    return relative_ess(log_weights + jnp.log(polarization_power))
