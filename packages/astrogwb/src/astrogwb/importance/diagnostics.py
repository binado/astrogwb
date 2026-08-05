r"""Health checks for importance-weighted quantities.

Both helpers here answer the same question -- *how many of my samples are
actually doing work?* -- for the two places importance weights enter this
pipeline: reweighting the fixed proposal catalog to sampled hyperparameters,
and reweighting a run-time amplitude prior to the scientific one in
post-processing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
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


def log_prior_reweighting(
    amplitude: jax.Array,
    *,
    used: dist.Distribution,
    target: dist.Distribution,
) -> jax.Array:
    r"""Log weights that swap the amplitude prior a chain was run under.

    The amplitude prior passed to
    :func:`~astrogwb.sampling.models.amplitude_marginalized_model` is a
    proposal, chosen so the closed-form marginalization applies. When the
    scientific prior differs -- most notably because a uniform prior on a
    multiplicative amplitude :math:`A` is *not* uniform on a physical parameter
    that enters as :math:`1/A` -- the difference is corrected afterwards by
    reweighting with

    .. math:: \ell_i = \ln \pi_\mathrm{target}(A_i) - \ln \pi_\mathrm{used}(A_i).

    Compose with :func:`relative_ess` for the corresponding health check.

    Parameters
    ----------
    amplitude:
        Reconstructed amplitude draws, e.g. from
        :func:`astrogwb.sampling.amplitude.draw_amplitude`.
    used:
        Prior the chain actually ran under.
    target:
        Prior the posterior should reflect.
    """
    return jnp.asarray(target.log_prob(amplitude)) - jnp.asarray(
        used.log_prob(amplitude)
    )
