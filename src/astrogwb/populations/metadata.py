from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from astrogwb.populations.registry import Population, build_population

__all__ = ["ModelKwargs", "PopulationMetadata"]


#: Construction kwargs travel inside the metadata JSON, so they must be JSON
#: scalars; the union keeps the round trip type-stable.
type ModelKwargs = dict[str, float | int | bool | str]


class PopulationMetadata(BaseModel):
    """The registered population an artifact was drawn from.

    ``model_kwargs`` is the flat construction mapping the population is built
    with, passed whole to :func:`~astrogwb.populations.build_population`.

    Validation is strict, so ``True`` does not validate as ``1``; ``int`` still
    promotes to ``float``. The seed is a simulator input, not part of the record.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_name: str
    model_kwargs: ModelKwargs = Field(default_factory=dict)

    def build(self) -> Population:
        """Reconstruct the generating population with its kwargs bound.

        The result hashes by identity, so each call forces a fresh jit trace:
        call once and reuse it. An unknown name raises, listing what is
        registered.
        """
        return build_population(self.model_name, **self.model_kwargs)

    def check_registered(self) -> None:
        """Raise unless the population builds from the recorded name and kwargs.

        An unknown name raises ``KeyError``; an unknown kwarg ``TypeError``.
        Only the name is checked, not the density it now points at.
        """
        self.build()

    def with_model_kwargs(self, **updates: float | bool | str) -> Self:
        """A re-validated copy with construction kwargs overridden.

        Constructs rather than ``model_copy(update=...)``, which skips
        validation.

        Parameters
        ----------
        **updates
            Construction keyword arguments to override.
        """
        return type(self)(
            model_name=self.model_name,
            model_kwargs={**self.model_kwargs, **updates},
        )
