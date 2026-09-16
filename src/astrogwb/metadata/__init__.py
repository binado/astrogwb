"""The provenance every persisted artifact carries, as validated records.

A catalog file describes the density that drew it and the waveform backend
that produced its power. Those two declarations are what this package owns:
they are small, flat, JSON-shaped, and they are the only part of an artifact
that travels as HDF5 attributes rather than as arrays.

They live in their own top-level package, not beside the code that consumes
them, because importing a submodule runs its parent package first:
:mod:`astrogwb.populations` imports the population models and
:mod:`astrogwb.waveform` imports the generators, and both reach JAX. A record
under either would drag JAX in behind it, which is exactly the constraint
``astrogwb.paper.config`` and the ``Snakefile``'s DAG construction cannot
afford. :mod:`astrogwb` itself is empty, so a top-level package here is
importable with nothing but ``numpy`` and ``pydantic`` behind it.

The edges back into the layers these records describe --
:meth:`PopulationMetadata.build` and the generators -- are taken inside method
bodies, so nothing here imports JAX at module scope.
``tests/core/test_metadata_imports.py`` asserts that directly.
"""

from astrogwb.metadata.population import (
    DENSITY_SITES_ATTR,
    MODEL_KWARGS_ATTR,
    MODEL_NAME_ATTR,
    POPULATION_ATTRS,
    SEED_ATTR,
    PopulationMetadata,
)

__all__ = [
    "DENSITY_SITES_ATTR",
    "MODEL_KWARGS_ATTR",
    "MODEL_NAME_ATTR",
    "POPULATION_ATTRS",
    "SEED_ATTR",
    "PopulationMetadata",
]
