"""Importance-sampling reference models and cosmology helpers for the SGWB.

The :mod:`astrogwb.sampling.models` models are callback-driven: they accept any
``(params, samples) -> (total_merger_rate, log_weights)`` callable. This
subpackage packages reference realizations of that callback together with the
cosmology primitives they share.

Import from explicit submodules rather than this package root:

- :mod:`astrogwb.cosmology` — shared cosmology helpers
- :mod:`astrogwb.importance.protocol` — the callback protocol
- :mod:`astrogwb.importance.diagnostics` — effective sample size and
  prior-reweighting helpers
- :mod:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation` —
  the BNS + Madau-Dickinson reference callback factory
"""

__all__: list[str] = []
