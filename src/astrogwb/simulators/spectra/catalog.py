"""A catalog of forward-model spectral density, drawn without materializing power.

The sibling of :mod:`astrogwb.simulators.polarization_power.catalog`. That catalog
persists ``(F, N)`` power for ``N`` sources and leaves the contraction to the
consumer; this one persists the contraction itself -- one spectral density per
draw -- and never holds a catalog's worth of waveforms at once. Both describe
the density that produced them with the same
:class:`~astrogwb.populations.PopulationMetadata`, so a consumer reconstructs the
generating source model the same way from either.

The two differ in what a row is, and that is the whole difference. A
polarization-power catalog's sample axis indexes *sources* drawn once at one
set of hyperparameters, which it records as scalar fiducials. A
spectral-density catalog's row axis indexes *draws* of the whole forward
model -- each with a Poisson or fixed event count and total merger rate -- and
its hyperparameters are a column per name, because a hyperparameter may be
drawn from a prior once per row.

Everything that determined the draws is one
:class:`~astrogwb.simulators.spectra.SpectraMetadata`: the waveform and population, each
hyperparameter's fixed value or prior, the observation time, the count mode,
fixed source count or Poisson padding, and the ``astrogwb`` version. Its
:meth:`~astrogwb.simulators.spectra.SpectraMetadata.key` names the cache file
with the seeds input of :func:`~astrogwb.simulators.spectra.spectra`; the draw
count is the length of that input. The columns are what
the record produced: a fixed hyperparameter's column must repeat its value, and
a sampled one's holds the value each row was drawn at.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from astrogwb.frequency import bin_widths, validate_frequency_grid
from astrogwb.populations import Population
from astrogwb.populations.metadata import PopulationMetadata
from astrogwb.simulators.spectra.metadata import SpectraMetadata
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["SpectralDensityCatalog"]


@dataclass(frozen=True, slots=True)
class SpectralDensityCatalog:
    """Forward-model spectral-density draws and the population that produced them.

    ``spectral_density`` is draw-first, shape ``(draws, F)`` -- the forward
    models' Predictive output, kept so the simulator writes what they produced.
    ``n_events`` and ``total_merger_rate`` have shape ``(draws,)``, and every
    hyperparameter column has shape ``(draws,)``. In fixed mode every count
    equals the recorded ``num_events``.

    The record is private and read through :attr:`metadata` and the
    properties below, so a caller cannot rebind it away from the arrays it
    describes; what a caller needs from the population is
    :meth:`get_population`, not the strings it was rebuilt from.
    """

    spectral_density: NDArray[Any]
    frequencies: NDArray[np.floating[Any]]
    n_events: NDArray[Any]
    total_merger_rate: NDArray[Any]
    hyperparameters: Mapping[str, NDArray[Any]]
    _metadata: SpectraMetadata

    def __post_init__(self) -> None:
        # The same grid invariant a polarization-power catalog checks against
        # itself: finite and strictly increasing, with the spacing free to vary.
        frequencies = validate_frequency_grid(self.frequencies)

        spectra = np.asarray(self.spectral_density)
        if spectra.ndim != 2 or spectra.shape[1] != frequencies.size:
            raise ValueError("spectral_density must have shape (draws, frequency)")
        draws = spectra.shape[0]

        n_events = np.asarray(self.n_events)
        if n_events.ndim != 1 or n_events.shape[0] != draws:
            raise ValueError("n_events must have shape (draws,)")
        if self._metadata.count == "fixed" and not np.all(
            n_events == self._metadata.num_events
        ):
            raise ValueError("fixed-mode n_events must match the metadata's num_events")
        rates = np.asarray(self.total_merger_rate)
        if rates.ndim != 1 or rates.shape[0] != draws:
            raise ValueError("total_merger_rate must have shape (draws,)")

        columns: dict[str, NDArray[Any]] = {}
        for name, values in self.hyperparameters.items():
            if not isinstance(name, str):
                raise TypeError("hyperparameter names must be strings")
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(
                    f"hyperparameter {name!r} must be one-dimensional, got "
                    f"shape {array.shape}"
                )
            if array.shape[0] != draws:
                raise ValueError(
                    f"hyperparameter {name!r} has {array.shape[0]} draws; "
                    f"expected {draws}"
                )
            columns[name] = array

        declared = set(self._metadata.hyperparameters)
        if set(columns) != declared:
            raise ValueError(
                f"hyperparameter columns {sorted(columns)} do not match the "
                f"metadata's {sorted(declared)}"
            )
        for name, value in self._metadata.fixed.items():
            if not np.all(columns[name] == value):
                raise ValueError(
                    f"hyperparameter {name!r} is fixed at {value} in the "
                    "metadata, but its column holds other values"
                )

        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "spectral_density", spectra)
        object.__setattr__(self, "n_events", n_events)
        object.__setattr__(self, "total_merger_rate", rates)
        object.__setattr__(self, "hyperparameters", columns)

    @classmethod
    def from_arrays(cls, outputs: Mapping[str, Any], metadata: SpectraMetadata) -> Self:
        """Wrap the arrays :func:`~astrogwb.simulators.spectra.spectra` returns.

        The metadata is supplied rather than inferred: the caller ran the node
        from it, so it is the only place that knows what drew these arrays.
        Construction checks that every column agrees on the draw count and that
        the fixed source count and every fixed hyperparameter match the record.
        """
        return cls(
            spectral_density=outputs["spectral_density"],
            frequencies=outputs["frequencies"],
            n_events=outputs["n_events"],
            total_merger_rate=outputs["total_merger_rate"],
            hyperparameters=outputs["hyperparameters"],
            _metadata=metadata,
        )

    # ----------------------------------------------------------------- #
    # The recorded population
    # ----------------------------------------------------------------- #
    @property
    def metadata(self) -> SpectraMetadata:
        """Everything that determined these draws; its key names their file."""
        return self._metadata

    @property
    def version(self) -> str:
        """The ``astrogwb`` version that generated these draws."""
        return self._metadata.version

    @property
    def count(self) -> Literal["poisson", "fixed"]:
        """Whether each realization has a Poisson or fixed source count."""
        return self._metadata.count

    @property
    def num_events(self) -> int | None:
        """Sources per realization in fixed mode; ``None`` for Poisson mode."""
        return self._metadata.num_events

    @property
    def n_max_sigma(self) -> float | None:
        """Poisson plate padding in standard deviations; ``None`` for fixed counts."""
        return self._metadata.n_max_sigma

    @property
    def observation_time(self) -> float:
        """Positive observation time in years; cancels for fixed-count spectra."""
        return self._metadata.observation_time

    @property
    def population(self) -> PopulationMetadata:
        """The population declaration these draws were produced from."""
        return self._metadata.population

    @property
    def waveform_metadata(self) -> WaveformMetadata:
        """The waveform settings that produced these draws."""
        return self._metadata.waveform

    @property
    def population_model_name(self) -> str:
        """The registry key of the population these draws used."""
        return self.population.model_name

    @property
    def population_model_kwargs(self) -> Mapping[str, Any]:
        """The model's construction kwargs, as persisted."""
        return dict(self.population.model_kwargs)

    def get_population(self) -> Population:
        """Reconstruct the generating population with its kwargs bound.

        ``merger_rate_fn`` is ``None`` only for a proposal density, which the
        simulator that writes this format refuses to draw from.
        """
        return self.population.build()

    @property
    def num_draws(self) -> int:
        """The number of forward-model draws in this catalog."""
        return int(self.spectral_density.shape[0])

    @property
    def bin_widths(self) -> NDArray[np.float64]:
        """Each frequency bin's width, derived from the grid rather than recorded.

        The same distinction the polarization-power catalog draws: what was
        *asked for* lives on ``waveform_metadata.frequency_resolution``, and
        the backend's realized grid can differ. Raises on a one-bin catalog,
        which is a supported shape -- there is no bin width to report.
        """
        return np.asarray(bin_widths(self.frequencies))
