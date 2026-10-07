from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from astrogwb.populations.registry import Population

__all__ = ["ModelKwargs", "PopulationMetadata", "widen_model_kwargs"]


#: Construction kwargs travel inside the metadata JSON, so they must be
#: JSON scalars. Declaring that here rather than as prose in the config layer
#: is what makes the round trip type-stable: an ``int`` stays an ``int``, a
#: ``float`` stays a ``float``, and a flag stays a ``bool`` or a ``str``.
type ModelKwargs = dict[str, float | int | bool | str]


def widen_model_kwargs(population: dict[str, Any]) -> None:
    """Widen numeric construction kwargs to ``float``, preserving booleans and strings.

    A setting spelled ``2`` in one config and ``2.0`` in another names the
    same draw, so both must hash alike.
    """
    population["model_kwargs"] = {
        name: value if isinstance(value, bool | str) else float(value)
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
    a ``True`` setting would validate as ``1``, silently undoing the bool
    rejection this record has always had. Strict mode still promotes ``int`` to
    ``float``, so a setting written ``2`` rather than ``2.0`` keeps validating.

    There is no seed: a seed picks one realization of the density this record
    describes, so it is an input to the simulator, not part of the record.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_name: str
    model_kwargs: ModelKwargs = Field(default_factory=dict)

    def build(self) -> Population:
        """Reconstruct the generating population with its kwargs bound.

        The result hashes by identity, so a getter called twice would hand a
        caller an equal-but-not-identical callable and force a jit recompile.
        Call once and reuse the result. An unknown name fails here, listing
        what is registered.

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

    def with_model_kwargs(self, **updates: float | bool | str) -> Self:
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
        )
