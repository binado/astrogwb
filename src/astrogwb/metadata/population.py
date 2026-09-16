"""The population declaration an artifact carries, as one record.

Every artifact this package persists -- a polarization-power catalog, a
spectral-density catalog -- records the density that produced it: the
registered population name, the flat construction kwargs it was built with,
the density factors included in importance weighting, and the seed the draw
used. That record was previously spelled out field by field on each artifact
and re-encoded attribute by attribute in each writer, which is how the two
formats drifted into naming the same thing differently.

:meth:`to_attrs` and :meth:`from_attrs` trade plain dictionaries, so this
module never imports h5py. It does not import the population registry at
module scope either: populating the registry means importing the models, which
reaches JAX, and :mod:`astrogwb.metadata` is deliberately importable without
it. :meth:`build` and :meth:`check_registered` take that import in their own
bodies, which is the only edge from here back into the population layer.

The record is deliberately not a cross-check: nothing here compares the
declaration against the arrays it travels with. It is the single statement of
what drew them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from astrogwb._attrs import int_attr, json_array_attr, json_object_attr

if TYPE_CHECKING:
    from astrogwb.populations.registry import Population

__all__ = [
    "DENSITY_SITES_ATTR",
    "MODEL_KWARGS_ATTR",
    "MODEL_NAME_ATTR",
    "POPULATION_ATTRS",
    "SEED_ATTR",
    "PopulationMetadata",
]

#: The canonical HDF5 attribute names for a persisted population record. One
#: definition, shared by every format that carries one, so two writers cannot
#: spell the same field differently.
MODEL_NAME_ATTR = "population_model"
MODEL_KWARGS_ATTR = "population_model_kwargs"
DENSITY_SITES_ATTR = "population_density_sites"
SEED_ATTR = "population_seed"

#: Every attribute :meth:`PopulationMetadata.to_attrs` writes, in write order.
POPULATION_ATTRS: tuple[str, ...] = (
    MODEL_NAME_ATTR,
    MODEL_KWARGS_ATTR,
    DENSITY_SITES_ATTR,
    SEED_ATTR,
)


@dataclass(frozen=True, slots=True)
class PopulationMetadata:
    """The registered population an artifact was drawn from.

    ``model_kwargs`` is the flat construction mapping the population is built
    with, passed whole to
    :func:`~astrogwb.populations.build_population`. Keeping it flat is what
    lets a narrowed redshift window reach the rebuilt population from a single
    rewrite.

    ``density_sites`` is ordered and load-bearing: the two mass sites form one
    conceptual ordered-pair density contribution, and a proposal density
    computed with either excluded gives silently wrong importance weights with
    no shape error anywhere.
    """

    model_name: str
    model_kwargs: Mapping[str, Any]
    density_sites: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a str")
        object.__setattr__(self, "model_kwargs", dict(self.model_kwargs))
        object.__setattr__(self, "density_sites", tuple(self.density_sites))

    def build(self) -> Population:
        """Reconstruct the generating population with its kwargs bound.

        Returns both callables from one call rather than a getter each: they
        hash by identity, so two getters would hand a caller a fresh,
        equal-but-not-identical pair on every call and force a jit recompile.
        Call once and reuse the result. An unknown name fails here, listing
        what is registered; ``merger_rate_fn`` is ``None`` for a population
        that declares no physical rate.

        Imports the registry in its own body: populating it means importing
        the population models, which reaches JAX, and this module is
        deliberately free of it.
        """
        from astrogwb.populations import build_population

        return build_population(self.model_name, **self.model_kwargs)

    def check_registered(self) -> None:
        """Raise if the recorded name is no longer registered.

        Building the population and discarding it is the whole check: an
        unknown name raises ``KeyError`` listing the registered populations,
        and a construction setting the population does not take raises
        ``TypeError``. It verifies the *name*, not the mathematics --
        re-pointing a registered key at a different density would be invisible
        here.
        """
        self.build()

    def to_attrs(self) -> dict[str, str | int]:
        """Encode the record as HDF5-writable scalar attributes.

        Mappings and sequences travel as JSON strings with sorted keys, so a
        file written twice from the same record is byte-identical.
        """
        return {
            MODEL_NAME_ATTR: self.model_name,
            MODEL_KWARGS_ATTR: json.dumps(dict(self.model_kwargs), sort_keys=True),
            DENSITY_SITES_ATTR: json.dumps(list(self.density_sites)),
            SEED_ATTR: self.seed,
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], *, label: str) -> Self:
        """Rebuild a record from decoded file attributes.

        ``attrs`` holds values already reduced to ``str``/``int``/``float``
        scalars; ``label`` names the file in error messages. Format-level
        concerns -- which attributes a given format requires, and any
        compatibility tier that supplies a missing one -- belong to the reader
        that calls this, not here.
        """
        return cls(
            model_name=str(attrs[MODEL_NAME_ATTR]),
            model_kwargs=json_object_attr(
                attrs[MODEL_KWARGS_ATTR], label=label, name=MODEL_KWARGS_ATTR
            ),
            density_sites=tuple(
                json_array_attr(
                    attrs[DENSITY_SITES_ATTR], label=label, name=DENSITY_SITES_ATTR
                )
            ),
            seed=int_attr(attrs[SEED_ATTR], label=label, name=SEED_ATTR),
        )
