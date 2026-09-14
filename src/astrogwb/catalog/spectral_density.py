"""A catalog of forward-model spectral density, drawn without materializing power.

The sibling of :mod:`astrogwb.catalog.polarization_power`. That catalog
persists ``(F, N)`` power for ``N`` sources and leaves the contraction to the
consumer; this one persists the contraction itself -- one spectral density per
draw -- and never holds a catalog's worth of waveforms at once. Both describe
the density that produced them with the same
:class:`~astrogwb.populations.PopulationRecord`, so a consumer reconstructs the
generating source model the same way from either.

The two differ in what a row is, and that is the whole difference. A
polarization-power catalog's sample axis indexes *sources* drawn once at one
set of hyperparameters, which it records as scalar fiducials. A
spectral-density catalog's row axis indexes *draws* of the whole forward
model -- each with its own Poisson event count and total merger rate -- so its
hyperparameters are a column per name, free to vary from row to row even though
the simulator that writes them today holds them fixed.

``n_max_sigma``, ``average_mode`` and ``observation_time`` are recorded because
none of them is recoverable from the arrays: the first sized the static plate
the Poisson count was capped against, the second fixed the inclination
convention the contraction assumed, and the third set the Poisson mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from astrogwb.frequency import uniform_grid_spacing
from astrogwb.gwb.spectral import AverageMode
from astrogwb.populations import MergerRateFn, PopulationRecord, SourceFn
from astrogwb.waveform import PolarizationPowerGenerator

__all__ = ["AVERAGE_MODES", "SpectralDensityCatalog", "average_mode_from"]

#: Every inclination convention ``gwb_forward_model`` accepts, as a runtime
#: tuple. ``AverageMode`` is a ``Literal`` and so cannot be checked against a
#: value read back from a file; this is the same set, spelled once.
AVERAGE_MODES: tuple[AverageMode, ...] = (
    "analytic_inclination",
    "catalog_inclination",
)


def average_mode_from(value: str) -> AverageMode:
    """Narrow a string read back from a file to an :data:`AverageMode`."""
    for mode in AVERAGE_MODES:
        if value == mode:
            return mode
    raise ValueError(f"average_mode is {value!r}, expected one of {AVERAGE_MODES}")


@dataclass(frozen=True, slots=True)
class SpectralDensityCatalog:
    """Forward-model spectral-density draws and the population that produced them.

    ``spectral_density`` is draw-first, shape ``(draws, F)`` -- the orientation
    :func:`~astrogwb.sampling.gwb_forward_model` returns, kept rather than
    transposed so the simulator writes what the model produced.
    ``n_events`` and ``total_merger_rate`` have shape ``(draws,)``, and every
    hyperparameter column has shape ``(draws,)``.

    Like its sibling, the population record is private: what callers need is
    :meth:`get_source_model` and :meth:`get_merger_rate_fn`, not the strings
    they were rebuilt from.
    """

    spectral_density: NDArray[Any]
    frequencies: NDArray[np.floating[Any]]
    n_events: NDArray[Any]
    total_merger_rate: NDArray[Any]
    hyperparameters: Mapping[str, NDArray[Any]]
    waveform_metadata: PolarizationPowerGenerator
    _population: PopulationRecord
    n_max_sigma: float
    average_mode: AverageMode
    observation_time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_max_sigma", float(self.n_max_sigma))
        object.__setattr__(self, "observation_time", float(self.observation_time))
        object.__setattr__(
            self, "average_mode", average_mode_from(str(self.average_mode))
        )
        if self.n_max_sigma < 0.0:
            raise ValueError("n_max_sigma must be non-negative")
        if self.observation_time <= 0.0:
            raise ValueError("observation_time must be positive")

        frequencies = np.asarray(self.frequencies)
        if frequencies.ndim != 1:
            raise ValueError("catalog frequencies must be one-dimensional")
        if frequencies.size >= 2:
            # Validation only, result discarded -- the same uniform-grid
            # invariant a polarization-power catalog checks against itself.
            uniform_grid_spacing(frequencies)

        spectra = np.asarray(self.spectral_density)
        if spectra.ndim != 2 or spectra.shape[1] != frequencies.size:
            raise ValueError("spectral_density must have shape (draws, frequency)")
        draws = spectra.shape[0]
        if draws <= 0:
            raise ValueError("catalog must contain at least one draw")

        n_events = np.asarray(self.n_events)
        if n_events.ndim != 1 or n_events.shape[0] != draws:
            raise ValueError("n_events must have shape (draws,)")
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

        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "spectral_density", spectra)
        object.__setattr__(self, "n_events", n_events)
        object.__setattr__(self, "total_merger_rate", rates)
        object.__setattr__(self, "hyperparameters", columns)

    # ----------------------------------------------------------------- #
    # The recorded population
    # ----------------------------------------------------------------- #
    @property
    def population(self) -> PopulationRecord:
        """The population declaration these draws were produced from."""
        return self._population

    @property
    def seed(self) -> int:
        """The seed the draws were folded from."""
        return self._population.seed

    @property
    def population_source_model_name(self) -> str:
        """The registry key of the source model these draws used."""
        return self._population.source_model_name

    @property
    def population_rate_model_name(self) -> str:
        """The registry key of the merger-rate function these draws used."""
        return self._population.rate_model_name

    @property
    def population_model_kwargs(self) -> Mapping[str, Any]:
        """The model's construction settings, as persisted."""
        return dict(self._population.model_kwargs)

    @property
    def density_sites(self) -> tuple[str, ...]:
        """Ordered source-density factors included in importance weighting."""
        return self._population.density_sites

    def get_source_model(self) -> SourceFn:
        """Reconstruct the generating source model with its settings bound."""
        return self._population.get_source_model()

    def get_merger_rate_fn(self) -> MergerRateFn:
        """Reconstruct the merger-rate function with its shared settings bound."""
        return self._population.get_merger_rate_fn()

    @property
    def num_draws(self) -> int:
        """The number of forward-model draws in this catalog."""
        return int(self.spectral_density.shape[0])

    @property
    def df(self) -> float:
        """The frequency bin width, measured from the grid rather than recorded.

        The same distinction the polarization-power catalog draws: what was
        *asked for* lives on ``waveform_metadata.frequency_resolution``, and
        the backend's realized spacing can differ. Raises on a one-bin
        catalog, which is a supported shape -- there is no bin width to report.
        """
        return uniform_grid_spacing(self.frequencies)

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a spectra file, reconstructing and validating its population record.

        Loading calls
        :meth:`~astrogwb.populations.PopulationRecord.check_registered` to
        verify the recorded source and rate model names are still registered;
        it does not re-run the forward model or compare the stored spectra
        against it.
        """
        from astrogwb.catalog import _io

        return _io.load_spectral_density_catalog(cls, path)

    def save(self, path: str | Path, *, compression: str | None = None) -> None:
        """Write the draws, waveform metadata, and the population record."""
        from astrogwb.catalog import _io

        _io.save_spectral_density_catalog(self, path, compression=compression)
