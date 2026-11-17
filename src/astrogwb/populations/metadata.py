"""Every population record, as one type that deserializes by ``model_name``."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from astrogwb.populations.base import PopulationMetadata
from astrogwb.populations.bns_coba import BNSCobaPopulation

__all__ = ["AnyPopulation", "PopulationMetadata"]

#: A field typed :data:`AnyPopulation` validates to the subclass its
#: ``model_name`` names.
type AnyPopulation = Annotated[BNSCobaPopulation, Field(discriminator="model_name")]
