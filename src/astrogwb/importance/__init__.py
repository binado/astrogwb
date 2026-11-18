"""Importance weighting of a fixed reference catalog against a target population.

The package is split into four submodules, imported directly:

- :mod:`astrogwb.importance.catalog` holds the importance proposal: fixed-size
  reference draws of a population and their per-source intensity.
- :mod:`astrogwb.importance.spectrum` holds the estimator: the background
  spectrum as a redshift quadrature over the reweighted intrinsic draws.
- :mod:`astrogwb.importance.model` holds :class:`~astrogwb.importance.model.CatalogModel`,
  the catalog as a :class:`~astrogwb.inference.model.SpectralDensityModel`.
- :mod:`astrogwb.importance.diagnostics` holds the effective-sample-size
  diagnostics of the weights and the amplitude cumulants of a catalog.
"""

__all__: list[str] = []
