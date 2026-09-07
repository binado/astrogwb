r"""Population importance weights evaluated at fixed catalog samples.

The target density and distance are evaluated at fixed catalog samples. The
proposal density and reference distance are cached by ``ImportanceCatalog``.
The reference distance must correspond to the stored polarization power,
which scales as distance to the power minus two. It must not be reconstructed
from a cosmology table: interpolation differences would bias every weight.
"""

from __future__ import annotations

import jax

from astrogwb.population import PopulationTerms


def importance_log_weights(
    target: PopulationTerms,
    *,
    proposal_log_prob: jax.Array,
    log_reference_distance: jax.Array,
) -> jax.Array:
    """Source density ratio times inverse-square distance rescaling in log space.

    The reference distance corresponds to the stored polarization power. No
    proposal merger rate is needed. Identical evaluated densities and distances
    give exactly zero log weights.
    """
    log_prob_ratio = target.log_prob - proposal_log_prob
    log_distance_ratio = target.log_luminosity_distance - log_reference_distance
    return log_prob_ratio - 2.0 * log_distance_ratio


__all__ = ["importance_log_weights"]
