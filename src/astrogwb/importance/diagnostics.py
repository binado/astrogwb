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
