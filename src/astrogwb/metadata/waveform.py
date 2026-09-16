"""Validated, JAX-free waveform provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from astrogwb._attrs import require_attrs, scalar_attr
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
        return self

    def build(self) -> Any:
        """Build the concrete generator described by this record."""
        from astrogwb.waveform import AnalyticInspiralGenerator, RippleGenerator

        if self.approximant == ANALYTIC_APPROXIMANT:
            return AnalyticInspiralGenerator(self)
        return RippleGenerator(self)

    def to_attrs(self) -> dict[str, str | float]:
        """Encode fields as scalar HDF5 attributes, omitting an unset alpha."""
        return {
            name: value
            for name in WAVEFORM_ATTRS
            if (value := getattr(self, name)) is not None
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], *, label: str) -> Self:
        """Decode waveform attributes, including pre-alpha catalogs."""
        require_attrs(attrs, WAVEFORM_ATTRS[:-1], label=label, kind="waveform metadata")
        try:
            return cls(
                approximant=str(scalar_attr(attrs["approximant"], name="approximant")),
                minimum_frequency=float(
                    scalar_attr(attrs["minimum_frequency"], name="minimum_frequency")
                ),
                maximum_frequency=float(
                    scalar_attr(attrs["maximum_frequency"], name="maximum_frequency")
                ),
                reference_frequency=float(
                    scalar_attr(
                        attrs["reference_frequency"], name="reference_frequency"
                    )
                ),
                sampling_frequency=float(
                    scalar_attr(attrs["sampling_frequency"], name="sampling_frequency")
                ),
                frequency_resolution=float(
                    scalar_attr(
                        attrs["frequency_resolution"], name="frequency_resolution"
                    )
                ),
                alpha=(
                    float(scalar_attr(attrs["alpha"], name="alpha"))
                    if "alpha" in attrs
                    else None
                ),
            )
        except (TypeError, ValidationError) as error:
            raise ValueError(f"{label}: invalid waveform metadata: {error}") from error


WAVEFORM_ATTRS: tuple[str, ...] = tuple(WaveformMetadata.model_fields)
