"""Everything that determines a set of population draws, before any waveform."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations.metadata import PopulationMetadata, widen_model_kwargs
from astrogwb.simulators.core.keys import content_key

__all__ = ["Hyperparameter", "PopulationDrawMetadata"]

#: One hyperparameter's declaration: the value it is fixed at, or the prior it
#: is sampled from once per draw. Which one is decided by the type alone, so a
#: parameter cannot be both fixed and sampled.
type Hyperparameter = float | DistributionConfig


class PopulationDrawMetadata(BaseModel):
    """The record a set of population draws is generated from and cached under.

    A *draw* is one realization of the whole population model: hyperparameters
    drawn (or fixed), a source count, and that many sources. It is both the
    request and the provenance, as
    :class:`~astrogwb.simulators.polarization_power.CatalogMetadata` is for a
    single catalog: :func:`~astrogwb.simulators.population.population` turns it
    and one seed per draw into arrays, and :meth:`key` is the middle part of the
    file name they are cached under.

    ``hyperparameters`` holds, per name, either a number -- every draw is made
    at that value -- or a :class:`~astrogwb.distributions.config.DistributionConfig`
    it is drawn from, independently per draw. Priors are data rather than code,
    so editing a bound re-keys the artifact without a version bump.

    ``count`` selects Poisson counts, ``Poisson(R(theta) * observation_time)``,
    or exactly ``num_events`` sources per draw. Counts are exact: there is no
    padded capacity, so nothing is truncated and no count depends on the other
    draws in the call. ``observation_time`` is positive in both modes; in fixed
    mode it only matters to a consumer that normalizes by it.

    Neither a seed nor a draw count is part of it: the seeds are the
    simulator's input, one per draw, and a draw depends on its own seed alone.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    population: PopulationMetadata
    hyperparameters: dict[str, Hyperparameter]
    observation_time: Annotated[float, Field(gt=0.0)]
    count: Literal["poisson", "fixed"] = "poisson"
    num_events: Annotated[int, Field(gt=0)] | None = None
    version: str = __version__

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.count == "fixed" and self.num_events is None:
            raise ValueError("num_events is required for fixed counts")
        if self.count == "poisson" and self.num_events is not None:
            raise ValueError("num_events is only valid for fixed counts")
        return self

    @property
    def fixed(self) -> dict[str, float]:
        """The hyperparameters every draw shares, by name."""
        return {
            name: float(value)
            for name, value in self.hyperparameters.items()
            if not isinstance(value, DistributionConfig)
        }

    @property
    def sampled(self) -> dict[str, DistributionConfig]:
        """The hyperparameters drawn once per draw, with their priors."""
        return {
            name: value
            for name, value in self.hyperparameters.items()
            if isinstance(value, DistributionConfig)
        }

    def key(self) -> str:
        """The content hash these draws are cached under.

        Numeric construction kwargs, fixed values and prior kwargs are widened
        to ``float``, so ``2`` and ``2.0`` name the same draws. Boolean
        construction choices retain their type.
        """
        return content_key(self.payload())

    def payload(self) -> dict[str, object]:
        """The canonical JSON-ready form :meth:`key` hashes.

        Exposed so a record that nests this one hashes the same widened form
        rather than a second spelling of it.
        """
        payload = self.model_dump(mode="json")
        widen_model_kwargs(payload["population"])
        payload["hyperparameters"] = {
            name: (
                {**value, "kwargs": {k: float(v) for k, v in value["kwargs"].items()}}
                if isinstance(value, dict)
                else float(value)
            )
            for name, value in payload["hyperparameters"].items()
        }
        return payload
