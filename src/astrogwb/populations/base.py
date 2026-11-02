"""The typed record a population is built from.

Every population is a frozen :class:`PopulationMetadata` subclass: its fields
are its construction settings, its ``model_name`` literal the tag that
:data:`~astrogwb.populations.AnyPopulation` deserializes it by, and
:meth:`PopulationMetadata.build` the
:data:`~astrogwb.populations.Population` it describes.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import cache
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.populations._types import Population

__all__ = ["PopulationMetadata"]


class PopulationMetadata(BaseModel):
    """The construction settings of a population, its redshift window included.

    Every population samples ``redshift`` on
    ``[minimum_redshift, maximum_redshift]``, so the window is declared here;
    subclasses add their own settings and a ``model_name`` literal.

    Validation is strict, so ``True`` does not validate as ``1``; ``int`` still
    promotes to ``float``. The hyperparameters a population is called with are
    not part of the record.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    minimum_redshift: Annotated[float, Field(ge=0.0)]
    maximum_redshift: float

    @model_validator(mode="after")
    def _ordered_window(self) -> Self:
        if self.maximum_redshift <= self.minimum_redshift:
            raise ValueError(
                f"maximum_redshift {self.maximum_redshift} must exceed "
                f"minimum_redshift {self.minimum_redshift}"
            )
        return self

    @abstractmethod
    def _build(self) -> Population:
        """The population these settings describe, built fresh."""

    def build(self) -> Population:
        """The population these settings describe.

        Equal records return the same callable, so a jitted function closing
        over it compiles once.
        """
        return _build(self)

    def replace(self, **updates: Any) -> Self:
        """A re-validated copy with fields overridden.

        Validates rather than ``model_copy(update=...)``, which skips
        validation.

        Parameters
        ----------
        **updates
            Field values to override.
        """
        return type(self).model_validate({**self.model_dump(), **updates})


@cache
def _build(population: PopulationMetadata) -> Population:
    return population._build()
