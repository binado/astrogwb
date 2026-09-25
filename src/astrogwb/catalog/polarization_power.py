"""A catalog of polarization power that describes the density that drew it.

Before this, three partial records described one run and none was sufficient:
the merged run TOML, a catalog attribute naming only the *shape* of the
redshift proposal, and a config object derived from the two. They were
reconciled by exact float equality over five hard-coded parameter names, which
left ``xi_0``, ``xi_n`` and ``local_merger_rate`` checked by nothing at all.

A catalog now records its complete population declaration -- the registered
population name, the construction kwargs, and the density factors
included in importance weighting, carried as one
:class:`~astrogwb.metadata.PopulationMetadata` -- plus the hyperparameters it
was drawn at. That is enough to reconstruct the exact map from hyperparameters
to source density, so the run config no longer restates any of it and nothing
has to be cross-checked.

Immutability is a contract, not a language guarantee. The dataclass is frozen
and transformations such as
:meth:`PolarizationPowerCatalog.restrict_redshift` return new catalogs, but the
underlying arrays are ordinary NumPy arrays and nothing stops a caller writing
through them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb import __version__
from astrogwb.frequency import uniform_grid_spacing
from astrogwb.metadata import CatalogMetadata, PopulationMetadata, WaveformMetadata
from astrogwb.populations import Population
from astrogwb.waveform import PolarizationPowerGenerator

__all__ = ["REDSHIFT_SITE", "PolarizationPowerCatalog"]

#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

#: Construction kwargs a population model must take for a catalog drawn from
#: it to support :meth:`PolarizationPowerCatalog.restrict_redshift`. Narrowing
#: the window changes the *normalization* of the generating density, so the
#: arrays and the model kwargs have to move together or the recorded density
#: stops describing the samples.
REDSHIFT_WINDOW_KWARGS = ("minimum_redshift", "maximum_redshift")


@dataclass(frozen=True, slots=True)
class PolarizationPowerCatalog:
    """Source parameters, polarization power, and the population that drew them.

    ``polarization_power`` is frequency-first, shape ``(F, N)``; every source
    parameter has shape ``(N,)``.

    The population record is private because its public interface is
    :meth:`get_population`: what callers need is the reconstructed
    ``fn(params)`` callables, not the strings they were rebuilt from. It is
    persisted as HDF5 attributes; the callables themselves never are.
    """

    source_parameters: Mapping[str, NDArray[Any]]
    polarization_power: NDArray[Any]
    frequencies: NDArray[np.floating[Any]]
    _metadata: CatalogMetadata
    _fiducials: Mapping[str, float]
    #: The ``astrogwb`` version that generated the arrays. A catalog built in
    #: memory is stamped with the running version; a loaded one keeps the
    #: version its file records, which is what its cache key was taken over.
    _version: str = field(default=__version__)

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
        if num_samples <= 0:
            raise ValueError("catalog must contain at least one sample")
        frequencies = np.asarray(self.frequencies)
        if frequencies.ndim != 1:
            raise ValueError("catalog frequencies must be one-dimensional")
        if num_frequencies != frequencies.size:
            raise ValueError(
                "polarization_power frequency axis does not match catalog frequencies"
            )
        if frequencies.size >= 2:
            # Validation only, result discarded: a uniform grid is a catalog
            # invariant, checked against itself rather than against a second
            # record. Runs once per catalog, including every `load`.
            uniform_grid_spacing(frequencies)

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

        object.__setattr__(self, "polarization_power", power)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "source_parameters", parameters)
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
        population: PopulationMetadata,
        fiducials: Mapping[str, float],
    ) -> Self:
        """Generate polarization power and return a validated catalog.

        The population record is supplied rather than inferred: the caller ran
        the model to draw ``source_parameters``, so it is the only place that
        knows which population and settings produced them. It arrives as one
        :class:`~astrogwb.metadata.PopulationMetadata` rather than as its four
        parts, so a caller cannot pair a model name with another draw's seed or
        density sites -- the record is the unit that has to stay consistent.

        ``fiducials`` stays separate because it is not part of the record: the
        hyperparameters a draw was made *at* describe the samples, while the
        record describes the density that produced them.

        Frequencies come from the generator rather than from metadata: that is
        the axis the backend actually produces.

        Generation runs under :func:`jax.jit`. Generators are deliberately
        jit-free so they compose inside NumPyro models, which inference jits
        anyway -- but this constructor is the opposite case. It is the eager,
        whole-catalog entry point (it materializes NumPy immediately, so it
        can never run under a trace), and at production sizes an unfused graph
        would hold every waveform intermediate at once. The jit belongs here,
        at the call site, rather than hidden inside the generator.
        """
        parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        frequencies = generator.frequencies
        power = jax.jit(generator.generate_batch)(source_parameters)
        return cls(
            source_parameters=parameters,
            polarization_power=np.asarray(power),
            frequencies=np.asarray(frequencies),
            _metadata=CatalogMetadata(
                waveform=generator.metadata, population=population
            ),
            _fiducials=fiducials,
        )

    # ----------------------------------------------------------------- #
    # The recorded population
    # ----------------------------------------------------------------- #
    @property
    def population(self) -> PopulationMetadata:
        """The population declaration this catalog was drawn from."""
        return self._metadata.population

    @property
    def waveform_metadata(self) -> WaveformMetadata:
        """The waveform settings that produced this catalog."""
        return self._metadata.waveform

    @property
    def seed(self) -> int:
        """The seed the population draw used."""
        return self.population.seed

    @property
    def population_model_name(self) -> str:
        """The registry key of the population this catalog was drawn from."""
        return self.population.model_name

    @property
    def population_model_kwargs(self) -> Mapping[str, Any]:
        """The model's construction kwargs, as persisted."""
        return dict(self.population.model_kwargs)

    @property
    def fiducials(self) -> Mapping[str, float]:
        """The hyperparameters the samples were drawn at.

        These are *generating* parameters. They are not bound into the
        callables :meth:`get_population` returns: a target evaluation supplies
        its own, and binding these would silently pin them.
        """
        return dict(self._fiducials)

    def get_population(self) -> Population:
        """Reconstruct the generating population with its kwargs bound.

        Returns the callables rather than the ``(name, kwargs)`` pair they were
        rebuilt from, so every consumer sees the one ``fn(params)`` interface.
        An unknown name fails here, listing what is registered. A window
        narrowed by :meth:`restrict_redshift` reaches both callables, because
        both are built from the one flat kwargs mapping it rewrites.

        ``merger_rate_fn`` is ``None`` for a catalog drawn from a proposal
        density that declares no physical rate. Each call builds fresh
        partials; call once and reuse the result.
        """
        return self.population.build()

    @property
    def version(self) -> str:
        """The ``astrogwb`` version that generated this catalog."""
        return self._version

    @property
    def num_samples(self) -> int:
        """The number of source samples in this catalog."""
        return int(self.polarization_power.shape[1])

    @property
    def df(self) -> float:
        """The catalog's frequency bin width, measured from its own grid.

        Measured, not recorded: the generating backend chooses the actual
        grid (Ripple's rounding is 5-smooth, not power-of-two), so its
        spacing is the only thing that can be right. What was *asked for*
        lives on ``waveform_metadata.frequency_resolution``, and the two can
        differ. Raises on a one-bin catalog, which is a supported shape --
        there is no bin width to report.
        """
        return uniform_grid_spacing(self.frequencies)

    # ----------------------------------------------------------------- #
    # Transformations
    # ----------------------------------------------------------------- #
    def restrict_redshift(
        self, minimum_redshift: float, maximum_redshift: float
    ) -> Self:
        """Restrict to sources inside a redshift window, narrowing the population.

        Both halves move together, which is the whole reason this is one
        method: dropping samples without narrowing the generating model would
        leave the recorded density normalized over a window the samples no
        longer span, and every importance weight would be off by that
        normalization. Draws truncated to a sub-window follow the same law as
        draws made directly from it, so only the support changes.

        Returns a new catalog; the original is untouched.
        """
        model_kwargs = self.population.model_kwargs
        missing = [name for name in REDSHIFT_WINDOW_KWARGS if name not in model_kwargs]
        if missing:
            raise ValueError(
                f"population {self.population.model_name!r} takes no "
                f"{missing} construction setting(s), so its redshift window cannot "
                "be narrowed"
            )
        generated_min = float(model_kwargs["minimum_redshift"])
        generated_max = float(model_kwargs["maximum_redshift"])
        if not generated_min <= minimum_redshift < maximum_redshift <= generated_max:
            raise ValueError(
                f"analysis redshift support [{minimum_redshift:.4g}, "
                f"{maximum_redshift:.4g}] must lie within the catalog generation "
                f"support [{generated_min:.4g}, {generated_max:.4g}]"
            )

        redshift = self.source_parameters[REDSHIFT_SITE]
        keep = np.flatnonzero(
            (redshift >= minimum_redshift) & (redshift <= maximum_redshift)
        )
        if keep.size == 0:
            raise ValueError(
                f"catalog has no samples in the "
                f"redshift window [{minimum_redshift:.4g}, {maximum_redshift:.4g}]"
            )
        return replace(
            self,
            source_parameters={
                name: values[keep] for name, values in self.source_parameters.items()
            },
            polarization_power=self.polarization_power[:, keep],
            _metadata=CatalogMetadata(
                waveform=self.waveform_metadata,
                population=self.population.with_model_kwargs(
                    minimum_redshift=float(minimum_redshift),
                    maximum_redshift=float(maximum_redshift),
                ),
            ),
        )

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a catalog file, reconstructing and validating its population record.

        Loading calls :meth:`~astrogwb.metadata.PopulationMetadata.check_registered`
        to verify the recorded population name is still registered; it does not
        re-execute the population or compare derived columns against the stored
        arrays. A catalog whose columns have drifted from its declared
        population is not caught here.

        Files written in older catalog formats are rejected; there is no
        compatibility reader.
        """
        from astrogwb.catalog import _io

        return _io.load_polarization_power_catalog(cls, path)

    def save(self, path: str | Path, *, compression: str | None = None) -> None:
        """Write source arrays, power, waveform metadata, and the population record.

        The population travels as ``(population name, construction kwargs,
        generating parameters, density sites)``. Neither this nor :meth:`load`
        serializes a Python callable.
        """
        from astrogwb.catalog import _io

        _io.save_polarization_power_catalog(self, path, compression=compression)
