"""Everything that determines a set of forward-model spectral-density draws."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.metadata import PriorSpec
from astrogwb.populations.metadata import PopulationMetadata, widen_model_kwargs
from astrogwb.simulators.core.keys import content_key
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["Hyperparameter", "SpectraMetadata"]

#: One hyperparameter's declaration: the value it is fixed at, or the prior it
#: is sampled from once per draw. Which one is decided by the type alone, so a
#: parameter cannot be both fixed and sampled.
type Hyperparameter = float | PriorSpec


class SpectraMetadata(BaseModel):
    """The record a spectral-density artifact is generated from and cached under.

    It is both the request and the provenance: a generator turns it into
    draws, the artifact carries it, and :meth:`key` is the file name the draws
    are cached under. Nothing outside it changes what the draws contain.

    ``hyperparameters`` holds, per name, either a number -- every draw is made
    at that value -- or a :class:`~astrogwb.metadata.PriorSpec` it is drawn
    from, independently per draw. Draws at fixed hyperparameters are the
    all-numbers case. Priors are data rather than code, so editing a bound
    re-keys the artifact without a version bump.

    Neither a seed nor a draw count is part of it: the seeds are the
    simulator's input, one per draw. ``count`` selects Poisson counts or exactly
    ``num_events`` sources per realization. ``observation_time`` is positive
    in both modes: it determines Poisson counts, but cancels from fixed-count
    spectrum normalization. ``n_max_sigma`` sizes the static Poisson event
    plate (default five sigma); fixed counts have no padding. These normalized
    fields are all part of the key. ``chunk_size`` is not: it only chunks the
    waveform reduction and consumes no randomness, so it is the simulator's.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata
    hyperparameters: dict[str, Hyperparameter]
    observation_time: Annotated[float, Field(gt=0.0)]
    count: Literal["poisson", "fixed"] = "poisson"
    num_events: Annotated[int, Field(gt=0)] | None = None
    n_max_sigma: Annotated[float, Field(ge=0.0)] | None = None
    version: str = __version__

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.count == "fixed":
            if self.num_events is None:
                raise ValueError("num_events is required for fixed counts")
            if self.n_max_sigma is not None:
                raise ValueError("n_max_sigma is only valid for Poisson counts")
        else:
            if self.num_events is not None:
                raise ValueError("num_events is only valid for fixed counts")
            if self.n_max_sigma is None:
                object.__setattr__(self, "n_max_sigma", 5.0)
        return self

    @property
    def fixed(self) -> dict[str, float]:
        """The hyperparameters every draw shares, by name."""
        return {
            name: float(value)
            for name, value in self.hyperparameters.items()
            if not isinstance(value, PriorSpec)
        }

    @property
    def sampled(self) -> dict[str, PriorSpec]:
        """The hyperparameters drawn once per draw, with their priors."""
        return {
            name: value
            for name, value in self.hyperparameters.items()
            if isinstance(value, PriorSpec)
        }

    def key(self) -> str:
        """The content hash these draws are cached under.

        Numeric construction kwargs, fixed values and prior kwargs are widened
        to ``float``, so ``2`` and ``2.0`` name the same draws. Boolean
        construction choices retain their type.
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
        return content_key(payload)
