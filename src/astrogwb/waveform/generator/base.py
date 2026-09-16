"""Common descriptor and interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from astrogwb._attrs import require_attrs, scalar_attr

__all__ = ["WAVEFORM_ATTRS", "PolarizationPowerGenerator"]


@dataclass(frozen=True, slots=True)
class PolarizationPowerGenerator:
    """Frequency-domain waveform descriptor and power-generation interface.

    Concrete subclasses turn source parameters into a frequency axis and
    frequency-first polarization power. :meth:`generate` is one source;
    :meth:`generate_batch` is a 1-D catalog. ``__call__`` returns
    ``(frequencies, polarization_power)`` so catalog construction records the
    axis the backend actually produced. The base class is also used as a
    metadata-only descriptor when a persisted catalog is loaded.

    ``frequency_resolution`` records what was *requested*; it is not
    necessarily the realized bin width. The generating backend chooses the
    actual grid, so the realized spacing belongs to the catalog it produces
    (see ``PolarizationPowerCatalog.df``), not to this descriptor.
    """

    approximant: str
    minimum_frequency: float
    maximum_frequency: float
    reference_frequency: float
    sampling_frequency: float
    frequency_resolution: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_frequency", float(self.minimum_frequency))
        object.__setattr__(self, "maximum_frequency", float(self.maximum_frequency))
        object.__setattr__(self, "reference_frequency", float(self.reference_frequency))
        object.__setattr__(self, "sampling_frequency", float(self.sampling_frequency))
        object.__setattr__(
            self, "frequency_resolution", float(self.frequency_resolution)
        )

        settings = (
            self.minimum_frequency,
            self.maximum_frequency,
            self.reference_frequency,
            self.sampling_frequency,
        )
        if not all(np.isfinite(settings)):
            raise ValueError("waveform frequency settings must be finite")
        if self.maximum_frequency < self.minimum_frequency:
            raise ValueError(
                "maximum_frequency must be greater than or equal to minimum_frequency"
            )
        if self.sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive")
        if (
            not np.isfinite(self.frequency_resolution)
            or self.frequency_resolution <= 0.0
        ):
            raise ValueError("frequency_resolution must be a finite positive scalar")

    @property
    def frequencies(self) -> ArrayLike:
        """The frequency axis this generator produces power on.

        Concrete generators know their grid from their configuration alone, so
        it is available before the first generation -- which is what lets a
        caller size a reduction over an empty catalog.
        """
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )

    def check_sources(self, source_parameters: Mapping[str, ArrayLike]) -> None:
        """Check source *values* against what this generator can represent.

        Eager only, and never called during generation: a traced array cannot
        drive a Python exception, and converting one to decide would sync the
        host on every batch. Generation therefore trusts its parameter arrays,
        and a caller establishes that trust once by calling this -- or
        :func:`~astrogwb.sampling.validate_source_model` -- on concrete values
        before handing a model to inference.

        The base implementation accepts everything: a generator constrains its
        inputs only where its waveform family does.
        """
        del source_parameters

    def generate(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for a single source, shape ``(F,)``.

        The default implementation wraps :meth:`generate_batch` around a
        length-1 catalog. Concrete generators that cannot form a batch of one
        should override this.
        """
        power = self.generate_batch(
            {
                name: jnp.atleast_1d(jnp.asarray(values))
                for name, values in source_parameters.items()
            }
        )
        if power.shape[-1] != 1:
            raise ValueError(
                f"generate expects a single source; received {power.shape[-1]} events"
            )
        return power[:, 0]

    def generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for a 1-D catalog, shape ``(F, N)``."""
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(frequencies, polarization_power)`` for ``source_parameters``.

        Polarization power is frequency-first, shape ``(F, N)``. The base
        implementation exists so it can describe a loaded catalog; only
        concrete generator subclasses are intended to generate.
        """
        del source_parameters
        raise NotImplementedError(
            "PolarizationPowerGenerator is a metadata-only descriptor; "
            "use a concrete generator subclass"
        )

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def to_attrs(self) -> dict[str, str | int | float]:
        """Encode the descriptor as the scalar attributes every artifact stamps.

        Only the base class's own fields travel, which is what
        :data:`WAVEFORM_ATTRS` means. A subclass field -- ``alpha`` on
        :class:`~astrogwb.waveform.AnalyticInspiralGenerator` -- is deliberately
        not persisted: :meth:`from_attrs` rebuilds the base descriptor either
        way, so an attribute no reader could restore would be dead weight in
        the file.
        """
        return {name: getattr(self, name) for name in WAVEFORM_ATTRS}

    @classmethod
    def from_attrs(
        cls, attrs: Mapping[str, Any], *, label: str
    ) -> PolarizationPowerGenerator:
        """Rebuild the metadata-only descriptor from decoded file attributes.

        Always returns the *base* descriptor, never ``cls``: the concrete
        subclass is not recoverable from these six fields, and rebuilding one
        would need construction settings the file does not carry. A loaded
        catalog therefore describes its waveform backend without being able to
        re-run it, which is the standing contract (see the class docstring).

        ``attrs`` holds values already reduced to ``str``/``int``/``float``
        scalars; ``label`` names the file in error messages.
        """
        require_attrs(attrs, WAVEFORM_ATTRS, label=label, kind="waveform metadata")
        try:
            return PolarizationPowerGenerator(
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
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}: invalid waveform metadata: {error}") from error


#: The descriptor, as attribute names -- derived from the base class's own
#: fields rather than restated, so a field added here cannot drift from what
#: the readers require. Both catalog formats stamp all six.
WAVEFORM_ATTRS: tuple[str, ...] = tuple(
    field.name for field in fields(PolarizationPowerGenerator)
)
