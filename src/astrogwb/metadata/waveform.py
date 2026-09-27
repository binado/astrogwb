"""Validated, JAX-free waveform provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.constants import ISCO_ALPHA

ANALYTIC_APPROXIMANT = "AnalyticInspiral"
_CONFUSABLE = frozenset(
    {"analytical", "analytic", "Analytic", "AnalyticalInspiral", "analytic_inspiral"}
)


class WaveformMetadata(BaseModel):
    """The complete settings that determine a waveform generator."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    approximant: str
    minimum_frequency: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_frequency: float = Field(gt=0.0, allow_inf_nan=False)
    reference_frequency: float = Field(gt=0.0, allow_inf_nan=False)
    sampling_frequency: float = Field(gt=0.0, allow_inf_nan=False)
    frequency_resolution: float = Field(gt=0.0, allow_inf_nan=False)
    alpha: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    use_taper_in_tidal_corrections: bool = True

    @model_validator(mode="before")
    @classmethod
    def _default_alpha(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            data = dict(value)
            if (
                data.get("approximant") == ANALYTIC_APPROXIMANT
                and data.get("alpha") is None
            ):
                data["alpha"] = ISCO_ALPHA
            return data
        return value

    @model_validator(mode="after")
    def _validate_settings(self) -> Self:
        if self.maximum_frequency < self.minimum_frequency:
            raise ValueError(
                "maximum_frequency must be greater than or equal to minimum_frequency"
            )
        if self.approximant in _CONFUSABLE:
            raise ValueError(
                f"approximant {self.approximant!r} is not a Ripple approximant; "
                f"the closed-form inspiral is spelled {ANALYTIC_APPROXIMANT!r}"
            )
        if self.alpha is not None and self.approximant != ANALYTIC_APPROXIMANT:
            raise ValueError("alpha is only valid for the AnalyticInspiral approximant")
        if (
            not self.use_taper_in_tidal_corrections
            and self.approximant == ANALYTIC_APPROXIMANT
        ):
            raise ValueError(
                "use_taper_in_tidal_corrections=False is only valid for Ripple "
                "tidal approximants"
            )
        return self

    def build(self) -> Any:
        """Build the concrete generator described by this record."""
        from astrogwb.waveform import AnalyticInspiralGenerator, RippleGenerator

        if self.approximant == ANALYTIC_APPROXIMANT:
            return AnalyticInspiralGenerator(self)
        return RippleGenerator(self)
