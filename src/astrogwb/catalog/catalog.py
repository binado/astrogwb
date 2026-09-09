"""A catalog that describes the density that drew it.

Before this, three partial records described one run and none was sufficient:
the merged run TOML, a catalog attribute naming only the *shape* of the
redshift proposal, and a config object derived from the two. They were
reconciled by exact float equality over five hard-coded parameter names, which
left ``xi_0``, ``xi_n`` and ``local_merger_rate`` checked by nothing at all.

A catalog now records its complete population declaration: the registered model
name, the model's construction settings, the hyperparameters it was drawn at,
and the density factors included in importance weighting. That is enough to
reconstruct the exact map from hyperparameters to source density, so the run
config no longer restates any of it and nothing has to be cross-checked.

The included-factor tuple is part of that record for a reason that is easy to
miss: the two mass sites form one conceptual ordered-pair density contribution.
A catalog whose proposal density was computed with either mass factor excluded
gives silently wrong weights with no shape error anywhere.

Immutability is a contract, not a language guarantee. The dataclass is frozen
and transformations such as :meth:`Catalog.restrict_redshift` return new
catalogs, but the underlying arrays are ordinary NumPy arrays and nothing stops
a caller writing through them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.populations import Population, build_population
from astrogwb.waveform import PolarizationPowerGenerator

__all__ = ["REDSHIFT_SITE", "Catalog"]

#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

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
    _model_name: str
    _model_kwargs: Mapping[str, Any]
    _fiducials: Mapping[str, float]
    _density_sites: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")

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
        if num_samples <= 0:
            raise ValueError("catalog must contain at least one sample")
        if num_frequencies != self.waveform_metadata.frequencies.size:
            raise ValueError(
                "polarization_power frequency axis does not match waveform frequencies"
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

        if not isinstance(self._density_sites, tuple):
            object.__setattr__(self, "_density_sites", tuple(self._density_sites))
        object.__setattr__(self, "polarization_power", power)
        object.__setattr__(self, "source_parameters", parameters)
        object.__setattr__(self, "_model_kwargs", dict(self._model_kwargs))
        object.__setattr__(
            self,
            "_fiducials",
            {name: float(value) for name, value in self._fiducials.items()},
        )

    @classmethod
    def from_generator(
        cls,
        source_parameters: Mapping[str, ArrayLike],
        *,
        generator: PolarizationPowerGenerator,
        model_name: str,
        model_kwargs: Mapping[str, Any],
        fiducials: Mapping[str, float],
        density_sites: tuple[str, ...],
        seed: int,
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
            _model_name=model_name,
            _model_kwargs=model_kwargs,
            _fiducials=fiducials,
            _density_sites=density_sites,
            seed=seed,
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
    def fiducials(self) -> Mapping[str, float]:
        """The hyperparameters the samples were drawn at.

        These are *generating* parameters. They are not bound into the callable
        :meth:`get_population_model` returns: a target evaluation supplies its
        own, and binding these would silently pin them.
        """
        return dict(self._fiducials)

    @property
    def density_sites(self) -> tuple[str, ...]:
        """Ordered source-density factors included in importance weighting."""
        return self._density_sites

    def get_population_model(self) -> Population:
        """Reconstruct the generating model with its construction settings bound.

        Returns the callable rather than a ``(model, kwargs)`` pair so every
        consumer sees the one ``model(params)`` interface. An unknown name
        fails here, listing what is registered.
        """
        return build_population(
            self._model_name,
            settings=self._model_kwargs,
            density_sites=self._density_sites,
        )

    @property
    def num_samples(self) -> int:
        """The number of source samples in this catalog."""
        return int(self.polarization_power.shape[1])

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
                f"catalog has no samples in the "
                f"redshift window [{z_min:.4g}, {z_max:.4g}]"
            )
        return replace(
            self,
            source_parameters={
                name: values[keep] for name, values in self.source_parameters.items()
            },
            polarization_power=self.polarization_power[:, keep],
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

        Files written in older catalog formats are rejected; there is no
        compatibility reader.
        """
        from astrogwb.catalog import _io

        return _io.load_catalog(cls, path)

    def save(self, path: str | Path, *, compression: str | None = None) -> None:
        """Write source arrays, power, waveform metadata, and the population record.

        The population travels as ``(registry name, construction kwargs,
        generating parameters, density sites)``. Neither this nor
        :meth:`load` serializes a Python callable.
        """
        from astrogwb.catalog import _io

        _io.save_catalog(self, path, compression=compression)
