"""Importance-sampling reference models and cosmology helpers for the SGWB.

The :mod:`astrogwb.sampling.numpyro_model` is callback-driven: it accepts any
``(params, samples) -> (total_merger_rate, log_weights)`` callable. This
subpackage packages reference realizations of that callback together with the
cosmology primitives they share.

Import from explicit submodules rather than this package root:

- :mod:`astrogwb.importance.cosmology` — shared cosmology helpers
- :mod:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation` —
  the BNS + Madau-Dickinson reference callback factory
"""

__all__: list[str] = []
