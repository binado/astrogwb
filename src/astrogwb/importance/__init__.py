"""Importance-sampling reference models and cosmology helpers for the SGWB.

The :mod:`astrogwb.sampling.numpyro_model` is callback-driven: it accepts any
``(params, samples) -> (total_merger_rate, log_weights)`` callable. This
subpackage packages reference realizations of that callback together with the
cosmology primitives they share, so scripts and notebooks import one canonical,
tested implementation rather than duplicating it.
"""

from .cosmology import flat_lcdm_grid, log_gw_em_ratio
from .models import make_merger_rate_and_log_weights_fn

__all__ = [
    "flat_lcdm_grid",
    "log_gw_em_ratio",
    "make_merger_rate_and_log_weights_fn",
]
