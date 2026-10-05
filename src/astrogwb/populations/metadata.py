"""The population declaration an artifact carries, as one record.

Every artifact this package persists -- a polarization-power catalog, a
spectral-density catalog -- records the density that produced it: the
registered population name, the flat construction kwargs it was built with,
and the seed the draw used. That record was previously spelled out field by
field on each artifact and re-encoded attribute by attribute in each writer, which is how the two
formats drifted into naming the same thing differently.

This module never imports h5py or the population registry at
module scope: populating the registry means importing the models, which
reaches JAX, and nothing here may *initialize* the XLA backend. :meth:`build`
and :meth:`check_registered` take that import in their own bodies, which is the
only edge from here back into the registry.

The record is deliberately not a cross-check: nothing here compares the
declaration against the arrays it travels with. It is the single statement of
what drew them -- and only of that. Which of the population's density factors
enter an importance weight is not part of it: that choice changes no sample,
is made by the analysis that reweights the draw, and is declared there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from astrogwb.populations.registry import Population

__all__ = ["ModelKwargs", "PopulationMetadata", "widen_model_kwargs"]


#: Construction kwargs travel inside the metadata JSON, so they must be
#: JSON scalars. Declaring that here rather than as prose in the config layer
#: is what makes the round trip type-stable: an ``int`` stays an ``int`` and a
#: ``float`` stays a ``float`` and the inclination choice stays a ``bool``.
type ModelKwargs = dict[str, float | int | bool]


def widen_model_kwargs(population: dict[str, Any]) -> None:
    """Widen numeric construction kwargs to ``float``, preserving booleans.

    A setting spelled ``2`` in one config and ``2.0`` in another names the
    same draw, so both must hash alike.
    """
    population["model_kwargs"] = {
        name: value if isinstance(value, bool) else float(value)
        for name, value in population["model_kwargs"].items()
    }


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

    @field_validator("model_kwargs")
    @classmethod
    def _validate_model_kwargs(cls, kwargs: ModelKwargs) -> ModelKwargs:
        for name, value in kwargs.items():
            if name == "sample_inclination":
                if not isinstance(value, bool):
                    raise ValueError("sample_inclination must be a bool")
            elif isinstance(value, bool):
                raise ValueError(f"{name} must be a number, not a bool")
        return kwargs

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

    def with_model_kwargs(self, **updates: float | bool) -> Self:
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
