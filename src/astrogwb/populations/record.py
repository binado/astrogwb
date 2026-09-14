"""The population declaration an artifact carries, as one record.

Every artifact this package persists -- a polarization-power catalog, a
spectral-density catalog -- records the density that produced it: the
registered source and rate names, the flat construction settings both were
built with, the density factors included in importance weighting, and the seed
the draw used. That record was previously spelled out field by field on each
artifact and re-encoded attribute by attribute in each writer, which is how the
two formats drifted into naming the same thing differently.

It lives beside the registry rather than beside the files because what it
*means* is "the arguments that rebuild :data:`SourceFn` and
:data:`MergerRateFn`". :meth:`to_attrs` and :meth:`from_attrs` trade plain
dictionaries, so this module never imports h5py and the population layer stays
free of a serialization dependency.

The record is deliberately not a cross-check: nothing here compares the
declaration against the arrays it travels with. It is the single statement of
what drew them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from astrogwb._attrs import int_attr, json_array_attr, json_object_attr
from astrogwb.populations.registry import (
    MergerRateFn,
    SourceFn,
    build_merger_rate_fn,
    build_source_model,
)

__all__ = [
    "DENSITY_SITES_ATTR",
    "MODEL_KWARGS_ATTR",
    "POPULATION_ATTRS",
    "RATE_MODEL_NAME_ATTR",
    "SEED_ATTR",
    "SOURCE_MODEL_NAME_ATTR",
    "PopulationRecord",
]

#: The canonical HDF5 attribute names for a persisted population record. One
#: definition, shared by every format that carries one, so two writers cannot
#: spell the same field differently.
SOURCE_MODEL_NAME_ATTR = "population_source_model"
RATE_MODEL_NAME_ATTR = "population_rate_model"
MODEL_KWARGS_ATTR = "population_model_kwargs"
DENSITY_SITES_ATTR = "population_density_sites"
SEED_ATTR = "population_seed"

#: Every attribute :meth:`PopulationRecord.to_attrs` writes, in write order.
POPULATION_ATTRS: tuple[str, ...] = (
    SOURCE_MODEL_NAME_ATTR,
    RATE_MODEL_NAME_ATTR,
    MODEL_KWARGS_ATTR,
    DENSITY_SITES_ATTR,
    SEED_ATTR,
)


@dataclass(frozen=True, slots=True)
class PopulationRecord:
    """The registered source and rate models an artifact was drawn from.

    ``model_kwargs`` is the one flat construction mapping both callables are
    built with: :func:`~astrogwb.populations.build_source_model` binds all of
    it, :func:`~astrogwb.populations.build_merger_rate_fn` binds the
    :data:`~astrogwb.populations.SHARED_MODEL_KWARGS` subset. Keeping it flat
    is what lets a narrowed redshift window reach both from a single rewrite.

    ``density_sites`` is ordered and load-bearing: the two mass sites form one
    conceptual ordered-pair density contribution, and a proposal density
    computed with either excluded gives silently wrong importance weights with
    no shape error anywhere.
    """

    source_model_name: str
    rate_model_name: str
    model_kwargs: Mapping[str, Any]
    density_sites: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if not isinstance(self.source_model_name, str):
            raise TypeError("source_model_name must be a str")
        if not isinstance(self.rate_model_name, str):
            raise TypeError("rate_model_name must be a str")
        object.__setattr__(self, "model_kwargs", dict(self.model_kwargs))
        object.__setattr__(self, "density_sites", tuple(self.density_sites))

    def get_source_model(self) -> SourceFn:
        """Reconstruct the generating source model with its settings bound.

        Returns the callable rather than a ``(model, kwargs)`` pair so every
        consumer sees the one ``source_model(params)`` interface. An unknown
        name fails here, listing what is registered. Each call builds a new
        :func:`functools.partial`; call once and reuse the result.
        """
        return build_source_model(self.source_model_name, settings=self.model_kwargs)

    def get_merger_rate_fn(self) -> MergerRateFn:
        """Reconstruct the merger-rate function with its shared settings bound.

        Reads the same flat construction settings as :meth:`get_source_model`,
        so a narrowed window reaches both.
        """
        return build_merger_rate_fn(self.rate_model_name, settings=self.model_kwargs)

    def check_registered(self) -> None:
        """Raise if either recorded name is no longer registered.

        Building both callables and discarding them is the whole check: an
        unknown name raises ``KeyError`` listing the registered models. It
        verifies the *names*, not the mathematics -- re-pointing a registered
        key at a different density would be invisible here.
        """
        self.get_source_model()
        self.get_merger_rate_fn()

    def to_attrs(self) -> dict[str, str | int]:
        """Encode the record as HDF5-writable scalar attributes.

        Mappings and sequences travel as JSON strings with sorted keys, so a
        file written twice from the same record is byte-identical.
        """
        return {
            SOURCE_MODEL_NAME_ATTR: self.source_model_name,
            RATE_MODEL_NAME_ATTR: self.rate_model_name,
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
            source_model_name=str(attrs[SOURCE_MODEL_NAME_ATTR]),
            rate_model_name=str(attrs[RATE_MODEL_NAME_ATTR]),
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
