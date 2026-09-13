r"""The importance weight itself, over plain arrays.

Separated from :mod:`astrogwb.importance.spectral` so the spectrum and the
diagnostic figures that want raw per-source weights share one implementation of
the arithmetic rather than two that agree by inspection. Identical evaluated
densities and distances give *exactly* zero log weights, which is the sanity
check the whole importance scheme is legible through -- and an arithmetic that
exists twice cannot promise it.
"""

from __future__ import annotations

import jax

__all__ = ["importance_log_weights"]


def importance_log_weights(
    *,
    target_log_prob: jax.Array,
    proposal_log_prob: jax.Array,
    log_luminosity_distance: jax.Array,
    log_reference_distance: jax.Array,
) -> jax.Array:
    r"""Source density ratio times inverse-square distance rescaling, in log space.

    .. math::

        \log w_i = \left[\log p(x_i \mid \theta) - \log q(x_i)\right]
            - 2\left[\log d_L(z_i \mid \theta) - \log d_i^{\mathrm{ref}}\right]

    The reference distance is the effective distance the stored polarization
    power was generated at; power scales as the inverse square of it. No
    proposal merger rate is needed -- a proposal is a density, not an
    observation.

    Every argument has shape ``(N,)``. Evaluate only where the proposal has
    support: subtracting two negative-infinite log densities gives ``nan``.
    """
    log_prob_ratio = target_log_prob - proposal_log_prob
    log_distance_ratio = log_luminosity_distance - log_reference_distance
    return log_prob_ratio - 2.0 * log_distance_ratio
