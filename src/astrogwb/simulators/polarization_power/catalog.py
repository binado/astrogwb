"""A catalog of polarization power that describes the density that drew it.

Before this, three partial records described one run and none was sufficient:
the merged run TOML, a catalog attribute naming only the *shape* of the
redshift proposal, and a config object derived from the two. They were
reconciled by exact float equality over five hard-coded parameter names, which
left ``xi_0``, ``xi_n`` and ``local_merger_rate`` checked by nothing at all.

A catalog now records the complete :class:`~astrogwb.simulators.polarization_power.CatalogMetadata`
it was generated from: the waveform settings, the population declaration -- the
registered population name and its construction kwargs, carried as one
:class:`~astrogwb.populations.PopulationMetadata` -- the hyperparameters it was
drawn at, the draw size and the ``astrogwb`` version. That is enough to
reconstruct the exact map from hyperparameters to source density, so the run
config no longer restates any of it and nothing has to be cross-checked; and
its :meth:`~astrogwb.simulators.polarization_power.CatalogMetadata.key` names
the cache file with the seed input
(:func:`~astrogwb.simulators.polarization_power.polarization_power`). The seed
picks the realization and is not part of the record.

Immutability is a contract, not a language guarantee. The dataclass is frozen
and transformations such as
:meth:`PolarizationPowerCatalog.restrict_redshift` return new catalogs, but the
underlying arrays are ordinary NumPy arrays and nothing stops a caller writing
through them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from astrogwb.frequency import bin_widths, validate_frequency_grid
from astrogwb.populations import Population
from astrogwb.populations.metadata import PopulationMetadata
from astrogwb.simulators.polarization_power.metadata import CatalogMetadata
from astrogwb.waveform.metadata import WaveformMetadata

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

    The record is private and read through :attr:`metadata` and the
    properties below, so a caller cannot rebind it away from the arrays it
    describes; what a caller needs from the population is
    :meth:`get_population`, not the strings it was rebuilt from. It is
    persisted as HDF5 attributes; the callables themselves never are.
    """

    source_parameters: Mapping[str, NDArray[Any]]
    polarization_power: NDArray[Any]
    frequencies: NDArray[np.floating[Any]]
    _metadata: CatalogMetadata

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
        if num_samples != self._metadata.num_samples:
            raise ValueError(
                f"polarization_power has {num_samples} samples; the metadata "
                f"records {self._metadata.num_samples}"
            )
        # One-dimensional, finite and strictly increasing is the whole grid
        # invariant, checked against itself rather than against a second
        # record. The spacing is free to vary: widths are derived from the axis
        # by `bin_widths`. Runs once per catalog, including every `load`.
        frequencies = validate_frequency_grid(self.frequencies)
        if num_frequencies != frequencies.size:
            raise ValueError(
                "polarization_power frequency axis does not match catalog frequencies"
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

        object.__setattr__(self, "polarization_power", power)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "source_parameters", parameters)

    @classmethod
    def from_arrays(cls, outputs: Mapping[str, Any], metadata: CatalogMetadata) -> Self:
        """Wrap the arrays :func:`~astrogwb.simulators.polarization_power.polarization_power`
        returns, and the record they were drawn from.

        The metadata is supplied rather than inferred: the caller ran the node
        from it, so it is the only place that knows what drew these arrays.
        Construction still checks that the two agree on the sample count.
        """
        return cls(
            source_parameters=outputs["source_parameters"],
            polarization_power=outputs["polarization_power"],
            frequencies=outputs["frequencies"],
            _metadata=metadata,
        )

    # ----------------------------------------------------------------- #
    # The recorded population
    # ----------------------------------------------------------------- #
    @property
    def metadata(self) -> CatalogMetadata:
        """Everything that determined this catalog; its key names the file."""
        return self._metadata

    @property
    def population(self) -> PopulationMetadata:
        """The population declaration this catalog was drawn from."""
        return self._metadata.population

    @property
    def waveform_metadata(self) -> WaveformMetadata:
        """The waveform settings that produced this catalog."""
        return self._metadata.waveform

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
        return dict(self._metadata.fiducials)

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
        return self._metadata.version

    @property
    def num_samples(self) -> int:
        """The number of source samples in this catalog."""
        return int(self.polarization_power.shape[1])

    @property
    def bin_widths(self) -> NDArray[np.float64]:
        """Each frequency bin's width, derived from the catalog's own grid.

        Derived, not recorded: the generating backend chooses the actual grid
        (Ripple's rounding is 5-smooth, not power-of-two), so the axis it
        produced is the only thing that can be right. What was *asked for*
        lives on ``waveform_metadata.frequency_resolution``, and the two can
        differ. The widths belong to the full grid; mask arrays, never this
        axis. Raises on a one-bin catalog, which is a supported shape -- there
        is no bin width to report.
        """
        return np.asarray(bin_widths(self.frequencies))

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

        Returns a new catalog; the original is untouched. Its
        :attr:`metadata` records the narrowed window and the kept sample count,
        so it no longer names a cached draw -- it is derived from one.
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
            _metadata=self._metadata.model_copy(
                update={
                    "population": self.population.with_model_kwargs(
                        minimum_redshift=float(minimum_redshift),
                        maximum_redshift=float(maximum_redshift),
                    ),
                    "num_samples": int(keep.size),
                }
            ),
        )
