"""Importance-integration reference catalogs: per-source intensity.

:class:`CatalogMetadata` is the record a catalog is drawn from.
:func:`intensity_data` turns a generator and source columns into
:class:`IntensityData`, one intensity column per source.
:func:`~astrogwb.importance.spectrum.reference_catalog` draws at a reference
redshift. :func:`read_catalog` reads back what
:func:`~astrogwb.utils.io.write` stored.

Reference generation uses fixed-size intrinsic draws.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Self, TypedDict, cast

import jax
import numpy as np
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.populations.metadata import AnyPopulation
from astrogwb.simulators.sources import Sampling
from astrogwb.utils.io import load
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = [
    "CatalogMetadata",
    "IntensityData",
    "ReferenceCatalog",
    "intensity_data",
    "read_catalog",
    "validate_matching_frequency_grids",
]

logger = logging.getLogger(__name__)


class CatalogMetadata(BaseModel):
    """Everything that determines an intensity catalog's contents.

    Both the request and the provenance: :func:`~astrogwb.importance.spectrum.reference_catalog`
    draws from it, and :func:`~astrogwb.utils.io.write` stores it beside the
    catalog. The seed is not part of it.

    ``sampling`` is how a reference catalog
    (:func:`~astrogwb.importance.spectrum.reference_catalog`) draws its intrinsic
    parameters: ``"random"`` i.i.d., or ``"sobol"`` on a scrambled Sobol net
    (:func:`~astrogwb.utils.numpyro.sample_sources_qmc`), which needs a
    power-of-two ``num_samples``.

    ``reference_redshift`` is the redshift a reference catalog's waveforms are
    generated at; ``None`` means the population's ``minimum_redshift``, the
    window's lower edge. Set it below the edge to let one catalog serve every
    window that starts at or above it
    (:func:`~astrogwb.importance.spectrum.reference_redshift`).
    ``reference_inclination`` records the reference-only face-on convention.
    Intensity always means the literal sum of squared polarization amplitudes.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: AnyPopulation
    fiducials: dict[str, float]
    num_samples: Annotated[int, Field(gt=0)]
    sampling: Sampling = "random"
    reference_redshift: Annotated[float, Field(ge=0.0)] | None = None
    reference_inclination: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    version: str = __version__

    @model_validator(mode="after")
    def _sobol_needs_a_power_of_two(self) -> Self:
        if self.sampling == "sobol" and self.num_samples & (self.num_samples - 1):
            raise ValueError(
                "sampling='sobol' needs a power-of-two num_samples, "
                f"got {self.num_samples}"
            )
        return self


class IntensityData(TypedDict):
    """One catalog: ``frequencies`` ``(F,)``, ``intensity`` ``(F, N)``.

    ``source_parameters`` columns are each ``(N,)``.
    """

    frequencies: NDArray[np.float64]
    intensity: NDArray[np.float64]
    source_parameters: dict[str, NDArray[Any]]


@dataclass(frozen=True)
class ReferenceCatalog:
    """Waveform intensities at one distance and redshift.

    Reference waveforms are face-on, quasi-circular, and aligned-spin with
    only the dominant quadrupole mode. Intrinsic columns exclude redshift,
    luminosity distance, and inclination.

    Attributes
    ----------
    frequencies
        Positive, increasing reference frequency grid in Hz, shape
        ``(F_ref,)``. Cover the redshifted query frequencies through the
        waveform's nonzero support.
    intensity
        Sum of the squared polarization amplitudes, shape ``(F_ref, N)``.
    intrinsic
        Source-frame intrinsic columns, in intensity column order.
    proposal_log_prob
        Finite joint log density of each draw under the proposal, shape
        ``(N,)``. Use the same variables and measure as the target log density.
    reference_redshift
        Redshift at which all reference waveforms were generated.
    reference_distance
        Positive gravitational-wave luminosity distance in Mpc used for the
        waveforms.
    """

    frequencies: JaxArrayLike
    intensity: JaxArrayLike
    intrinsic: Mapping[str, JaxArrayLike]
    proposal_log_prob: JaxArrayLike
    reference_redshift: float
    reference_distance: float


def intensity_data(
    generator: WaveformGenerator,
    source_parameters: Mapping[str, ArrayLike],
    *,
    chunk_size: int | None = None,
) -> IntensityData:
    """The intensity of ``source_parameters`` as a catalog, generated eagerly.

    Values are checked here, eagerly: traced generation would silently drop a
    degree of freedom the approximant cannot represent. ``chunk_size`` is
    :meth:`~astrogwb.waveform.WaveformGenerator.generate_batch`'s.

    Parameters
    ----------
    generator
        Waveform generator.
    source_parameters
        Source parameter values by name.
    chunk_size
        Number of sources per chunk.
    """
    jax.config.update("jax_enable_x64", True)
    columns = {name: np.asarray(values) for name, values in source_parameters.items()}
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_max=%.1f Hz)",
        generator.metadata.approximant,
        len(next(iter(columns.values()))),
        generator.metadata.minimum_frequency,
        generator.metadata.maximum_frequency,
    )
    generator.check_sources(columns)
    # Generators are jit-free to compose inside NumPyro models; eagerly, an
    # unfused graph would hold every waveform intermediate at once.
    generate = jax.jit(partial(generator.generate_batch, chunk_size=chunk_size))
    return {
        "frequencies": np.asarray(generator.frequencies),
        "intensity": np.asarray(generate(columns)),
        "source_parameters": columns,
    }


def read_catalog(
    path: str | Path,
) -> tuple[IntensityData, CatalogMetadata, int]:
    """The catalog at ``path``, its record and the seed it was drawn at.

    ``astrogwb model-catalog`` writes it; nothing here draws.

    Parameters
    ----------
    path
        File path.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"no catalog at {path}")
    data, metadata, attrs = load(path, CatalogMetadata)
    return cast(IntensityData, data), metadata, int(attrs["seed"])


def validate_matching_frequency_grids(
    first: ArrayLike, second: ArrayLike, *, label: str = "injection and proposal"
) -> None:
    """Require two catalogs to share the exact frequency grid.

    Parameters
    ----------
    first, second
        The catalogs' frequency grids.
    label
        Label used in error messages.

    Raises
    ------
    ValueError
        If the grids differ.
    """
    if not np.array_equal(np.asarray(first), np.asarray(second)):
        raise ValueError(f"{label} catalogs must have identical frequency grids")
