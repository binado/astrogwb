"""A catalog that describes the density that drew it.

Before this, three partial records described one run and none was sufficient:
the merged run TOML, a catalog attribute naming only the *shape* of the
redshift proposal, and a config object derived from the two. They were
reconciled by exact float equality over five hard-coded parameter names, which
left ``xi_0``, ``xi_n`` and ``local_merger_rate`` checked by nothing at all.

A catalog now records its complete population declaration: the registered model
name, the model's construction settings, the hyperparameters it was drawn at,
and the density factors excluded from importance weighting. That is enough to
reconstruct the exact map from hyperparameters to source density, so the run
config no longer restates any of it and nothing has to be cross-checked.

The excluded-factor set is part of that record for a reason that is easy to
miss: a catalog whose proposal density was computed with the mass factors
excluded, reweighted against a target that includes them, gives silently wrong
weights with no shape error anywhere.

Immutability is a contract, not a language guarantee. The dataclass is frozen
and transformations such as :meth:`Catalog.restrict_redshift` return new
catalogs, but the underlying arrays are ordinary NumPy arrays and nothing stops
a caller writing through them.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.catalog.metadata import PopulationMetadata
from astrogwb.populations import (
    REDSHIFT_SITE,
    PopulationModel,
    population_model,
)
from astrogwb.waveform import PolarizationPowerGenerator

__all__ = ["Catalog"]

#: Construction settings a population model must take for a catalog drawn from
#: it to support :meth:`Catalog.restrict_redshift`. Narrowing the window changes
#: the *normalization* of the generating density, so the arrays and the model
#: kwargs have to move together or the recorded density stops describing the
#: samples.
REDSHIFT_WINDOW_KWARGS = ("z_min", "z_max")


@dataclass(frozen=True, slots=True)
class Catalog:
    """Source parameters, polarization power, and the population that drew them.

    ``polarization_power`` is frequency-first, shape ``(F, N)``; every source
    parameter has shape ``(N,)``.

    The population fields are private because their public interface is
    :meth:`get_population_model`: what callers need is the reconstructed
    ``model(params)`` callable, not the strings it was rebuilt from. They are
    persisted as HDF5 attributes; the callable itself never is.
    """

    source_parameters: Mapping[str, NDArray[Any]]
    polarization_power: NDArray[Any]
    waveform_metadata: PolarizationPowerGenerator
    population_metadata: PopulationMetadata
    _model_name: str
    _model_kwargs: Mapping[str, Any]
    _population_params: Mapping[str, float]
    _hidden_sites: frozenset[str]

    def __post_init__(self) -> None:
        power = np.asarray(self.polarization_power)
        if power.ndim != 2 or not np.issubdtype(power.dtype, np.number):
            raise ValueError(
                "polarization_power must be a real-valued two-dimensional array"
            )
        if np.issubdtype(power.dtype, np.complexfloating):
            raise ValueError(
                "polarization_power must be real-valued; pass |h+|^2 + |hx|^2, "
                "not raw complex polarizations"
            )
        num_frequencies, num_samples = power.shape
        if num_frequencies != self.waveform_metadata.frequencies.size:
            raise ValueError(
                "polarization_power frequency axis does not match waveform frequencies"
            )
        if num_samples != self.population_metadata.num_samples:
            raise ValueError(
                "population num_samples does not match polarization_power sample axis"
            )

        parameters: dict[str, NDArray[Any]] = {}
        for name, values in self.source_parameters.items():
            if not isinstance(name, str):
                raise TypeError("source parameter names must be strings")
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(
                    f"source parameter {name!r} must be one-dimensional, got "
                    f"shape {array.shape}"
                )
            if array.shape[0] != num_samples:
                raise ValueError(
                    f"source parameter {name!r} has {array.shape[0]} samples; "
                    f"expected {num_samples}"
                )
            parameters[name] = array
        if REDSHIFT_SITE not in parameters:
            raise ValueError(
                f"a catalog must store a {REDSHIFT_SITE!r} column; it is the one "
                "source parameter whose density never cancels in an importance "
                "weight"
            )

        if not isinstance(self._hidden_sites, frozenset):
            object.__setattr__(self, "_hidden_sites", frozenset(self._hidden_sites))
        object.__setattr__(self, "polarization_power", power)
        object.__setattr__(self, "source_parameters", parameters)
        object.__setattr__(self, "_model_kwargs", dict(self._model_kwargs))
        object.__setattr__(
            self,
            "_population_params",
            {name: float(value) for name, value in self._population_params.items()},
        )

    @classmethod
    def from_generator(
        cls,
        source_parameters: Mapping[str, ArrayLike],
        *,
        generator: PolarizationPowerGenerator,
        population_metadata: PopulationMetadata,
        model_name: str,
        model_kwargs: Mapping[str, Any],
        population_params: Mapping[str, float],
        hidden_sites: frozenset[str],
    ) -> Self:
        """Generate polarization power and return a validated catalog.

        The population record is supplied rather than inferred: the caller ran
        the model to draw ``source_parameters``, so it is the only place that
        knows which model and settings produced them.
        """
        parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        power = np.asarray(generator(source_parameters))
        return cls(
            source_parameters=parameters,
            polarization_power=power,
            waveform_metadata=generator,
            population_metadata=population_metadata,
            _model_name=model_name,
            _model_kwargs=model_kwargs,
            _population_params=population_params,
            _hidden_sites=hidden_sites,
        )

    # ----------------------------------------------------------------- #
    # The recorded population
    # ----------------------------------------------------------------- #
    @property
    def population_model_name(self) -> str:
        """The registry key of the model this catalog was drawn from."""
        return self._model_name

    @property
    def population_model_kwargs(self) -> Mapping[str, Any]:
        """The model's construction settings, as persisted."""
        return dict(self._model_kwargs)

    @property
    def population_params(self) -> Mapping[str, float]:
        """The hyperparameters the samples were drawn at.

        These are *generating* parameters. They are not bound into the callable
        :meth:`get_population_model` returns: a target evaluation supplies its
        own, and binding these would silently pin them.
        """
        return dict(self._population_params)

    @property
    def hidden_sites(self) -> frozenset[str]:
        """Source-density factors excluded from importance weighting."""
        return self._hidden_sites

    def get_population_model(self) -> PopulationModel:
        """Reconstruct the generating model with its construction settings bound.

        Returns the callable rather than a ``(model, kwargs)`` pair so every
        consumer sees the one ``model(params)`` interface. An unknown name
        fails here, listing what is registered.
        """
        return functools.partial(
            population_model(self._model_name), **self._model_kwargs
        )

    # ----------------------------------------------------------------- #
    # Transformations
    # ----------------------------------------------------------------- #
    def restrict_redshift(self, z_min: float, z_max: float) -> Self:
        """Restrict to sources inside a redshift window, narrowing the population.

        Both halves move together, which is the whole reason this is one
        method: dropping samples without narrowing the generating model would
        leave the recorded density normalized over a window the samples no
        longer span, and every importance weight would be off by that
        normalization. Draws truncated to a sub-window follow the same law as
        draws made directly from it, so only the support changes.

        Returns a new catalog; the original is untouched.
        """
        missing = [
            name for name in REDSHIFT_WINDOW_KWARGS if name not in self._model_kwargs
        ]
        if missing:
            raise ValueError(
                f"population model {self._model_name!r} takes no {missing} "
                "construction setting(s), so its redshift window cannot be narrowed"
            )
        generated_min = float(self._model_kwargs["z_min"])
        generated_max = float(self._model_kwargs["z_max"])
        if not generated_min <= z_min < z_max <= generated_max:
            raise ValueError(
                f"analysis redshift support [{z_min:.4g}, {z_max:.4g}] must lie "
                f"within the catalog generation support [{generated_min:.4g}, "
                f"{generated_max:.4g}]"
            )

        redshift = self.source_parameters[REDSHIFT_SITE]
        keep = np.flatnonzero((redshift >= z_min) & (redshift <= z_max))
        if keep.size == 0:
            raise ValueError(
                f"catalog {self.population_metadata.name!r} has no samples in the "
                f"redshift window [{z_min:.4g}, {z_max:.4g}]"
            )
        return replace(
            self,
            source_parameters={
                name: values[keep] for name, values in self.source_parameters.items()
            },
            polarization_power=self.polarization_power[:, keep],
            population_metadata=replace(
                self.population_metadata, num_samples=int(keep.size)
            ),
            _model_kwargs={
                **self._model_kwargs,
                "z_min": float(z_min),
                "z_max": float(z_max),
            },
        )

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a catalog file, validating its population record against its arrays.

        Loading re-executes the recorded population at the stored stochastic
        values and compares every derived column it declares -- detector-frame
        masses, luminosity distance -- against what the file holds. A catalog
        whose columns drifted from its declared population fails here rather
        than producing a plausible, wrong spectrum.

        Files written before the population record existed are rejected with a
        regeneration message; there is no reconstruction path for them.
        """
        from astrogwb.catalog import _io

        return _io.load_catalog(cls, path)

    def save(self, path: str | Path, *, compression: str | None = None) -> None:
        """Write source arrays, power, waveform metadata, and the population record.

        The population travels as ``(registry name, construction kwargs,
        generating parameters, excluded sites)``. Neither this nor
        :meth:`load` serializes a Python callable.
        """
        from astrogwb.catalog import _io

        _io.save_catalog(self, path, compression=compression)
