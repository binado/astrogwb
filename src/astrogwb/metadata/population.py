"""The population declaration an artifact carries, as one record.

Every artifact this package persists -- a polarization-power catalog, a
spectral-density catalog -- records the density that produced it: the
registered population name, the flat construction kwargs it was built with,
and the seed the draw used. That record was previously spelled out field by
field on each artifact
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
what drew them -- and only of that. Which of the population's density factors
enter an importance weight is not part of it: that choice changes no sample,
is made by the analysis that reweights the draw, and is declared there.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from astrogwb._attrs import int_attr, json_object_attr

if TYPE_CHECKING:
    from astrogwb.populations.registry import Population

__all__ = [
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
SEED_ATTR = "population_seed"

#: Every attribute :meth:`PopulationMetadata.to_attrs` writes, in write order.
POPULATION_ATTRS: tuple[str, ...] = (
    MODEL_NAME_ATTR,
    MODEL_KWARGS_ATTR,
    SEED_ATTR,
)


#: Construction kwargs travel as HDF5 attributes via JSON, so they must be
#: JSON scalars. Declaring that here rather than as prose in the config layer
#: is what makes the round trip type-stable: an ``int`` stays an ``int`` and a
#: ``float`` stays a ``float``, so a file written from a loaded record is
#: byte-identical to the one it was read from.
ModelKwargs = dict[str, float | int]


class PopulationMetadata(BaseModel):
    """The registered population an artifact was drawn from.

    ``model_kwargs`` is the flat construction mapping the population is built
    with, passed whole to
    :func:`~astrogwb.populations.build_population`. Keeping it flat is what
    lets a narrowed redshift window reach the rebuilt population from a single
    rewrite.

    Validation is strict. That is not fussiness: in pydantic's default lax mode
    a ``seed`` of ``True`` validates as ``1``, which would silently undo the
    bool rejection this record has always had, and a ``"42"`` from a
    hand-edited config would be accepted as an integer. Strict mode still
    promotes ``int`` to ``float``, so a setting written ``2`` rather than
    ``2.0`` keeps validating.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_name: str
    model_kwargs: ModelKwargs = Field(default_factory=dict)
    seed: int

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

    def with_model_kwargs(self, **updates: float) -> Self:
        """A re-validated copy with construction kwargs overridden.

        Constructs rather than using ``model_copy(update=...)``, which writes
        the field and skips every validator -- so a narrowed redshift window
        would reach a rebuilt population unchecked. Every field is named rather
        than splatted, so a field added later is a type error here instead of
        something silently dropped from the copy.
        """
        return type(self)(
            model_name=self.model_name,
            model_kwargs={**self.model_kwargs, **updates},
            seed=self.seed,
        )

    def to_attrs(self) -> dict[str, str | int]:
        """Encode the record as HDF5-writable scalar attributes.

        ``model_kwargs`` travels as a JSON string with sorted keys, so a file
        written twice from the same record is byte-identical.

        Hand-written rather than ``model_dump_json``: pydantic serializes in
        field-declaration order, not sorted-key order, so switching to it would
        quietly cost the byte-identity this guarantees.
        """
        return {
            MODEL_NAME_ATTR: self.model_name,
            MODEL_KWARGS_ATTR: json.dumps(dict(self.model_kwargs), sort_keys=True),
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

        Only the named keys are read, so a file written before the weighting
        choice moved to the analysis config still loads: its leftover
        ``population_density_sites`` attribute is simply not one of them.
        """
        try:
            return cls(
                model_name=str(attrs[MODEL_NAME_ATTR]),
                model_kwargs=json_object_attr(
                    attrs[MODEL_KWARGS_ATTR], label=label, name=MODEL_KWARGS_ATTR
                ),
                seed=int_attr(attrs[SEED_ATTR], label=label, name=SEED_ATTR),
            )
        except ValidationError as error:
            raise ValueError(
                f"{label}: invalid population metadata: {error}"
            ) from error
