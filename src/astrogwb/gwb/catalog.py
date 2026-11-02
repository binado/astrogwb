"""Importance-integration reference catalogs: per-source polarization power.

:class:`CatalogMetadata` is the record a catalog is drawn from and cached
under. :func:`polarization_power_data` turns a generator and source columns
into :class:`PolarizationPowerData`, one power column per source; where the
sources come from is the caller's. :func:`draw_catalog` is the plain catalog --
the population a record names, drawn at its fiducials from one key -- and
:func:`~astrogwb.gwb.importance.reference_catalog` draws at the redshift
window's lower edge.

These are fixed-size draws for importance weighting, not Poisson realizations;
those are :mod:`astrogwb.simulators`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial
from typing import Annotated, Any, Literal, Self, TypedDict

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.populations.evaluation import sample_and_logprob
from astrogwb.populations.joint import fix_extrinsic_parameters, joint_model
from astrogwb.populations.metadata import PopulationMetadata
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.factory import waveform_generator
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = [
    "REDSHIFT_SITE",
    "REDSHIFT_WINDOW_KWARGS",
    "CatalogMetadata",
    "PolarizationPowerData",
    "draw_catalog",
    "polarization_power_data",
]

logger = logging.getLogger(__name__)


class CatalogMetadata(BaseModel):
    """Everything that determines a polarization-power catalog's contents.

    Both the request and the provenance: :func:`draw_catalog` turns it and a
    PRNG key into a catalog, stored beside it by
    :func:`~astrogwb.utils.io.write`. The seed is not part of it.

    ``sampling`` is how a reference catalog
    (:func:`~astrogwb.gwb.importance.reference_catalog`) draws its intrinsic
    parameters: ``"random"`` i.i.d., or ``"sobol"`` on a scrambled Sobol net
    (:func:`~astrogwb.populations.evaluation.sample_sources_qmc`), which needs a
    power-of-two ``num_samples``. A plain catalog is always random.

    ``reference_redshift`` is the redshift a reference catalog's waveforms are
    generated at; ``None`` means the population's ``minimum_redshift``, the
    window's lower edge. Set it below the edge to let one catalog serve every
    window that starts at or above it
    (:func:`~astrogwb.gwb.importance.reference_redshift`).
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata
    fiducials: dict[str, float]
    num_samples: Annotated[int, Field(gt=0)]
    sampling: Literal["random", "sobol"] = "random"
    reference_redshift: Annotated[float, Field(ge=0.0)] | None = None
    version: str = __version__

    @model_validator(mode="after")
    def _sobol_needs_a_power_of_two(self) -> Self:
        if self.sampling == "sobol" and self.num_samples & (self.num_samples - 1):
            raise ValueError(
                "sampling='sobol' needs a power-of-two num_samples, "
                f"got {self.num_samples}"
            )
        return self


class PolarizationPowerData(TypedDict):
    """One catalog: ``frequencies`` ``(F,)``, ``polarization_power`` ``(F, N)``.

    ``source_parameters`` columns are each ``(N,)``.
    """

    frequencies: NDArray[np.float64]
    polarization_power: NDArray[np.float64]
    source_parameters: dict[str, NDArray[Any]]


def polarization_power_data(
    generator: WaveformGenerator,
    source_parameters: Mapping[str, ArrayLike],
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The power of ``source_parameters`` as a catalog, generated eagerly.

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
    power = jax.jit(partial(generator.generate_batch, chunk_size=chunk_size))
    return {
        "frequencies": np.asarray(generator.frequencies),
        "polarization_power": np.asarray(power(columns)),
        "source_parameters": columns,
    }


def draw_catalog(
    metadata: CatalogMetadata, key: jax.Array, *, chunk_size: int | None = None
) -> PolarizationPowerData:
    """The plain catalog ``key`` draws: ``num_samples`` sources at the fiducials.

    Parameters
    ----------
    metadata
        Metadata describing the object.
    key
        JAX random key.
    chunk_size
        Number of sources per chunk.
    """
    # Before the draw: ripplegw enables x64 on import, but only once the
    # generator runs, after the sources are sampled.
    jax.config.update("jax_enable_x64", True)
    if metadata.version != __version__:
        raise ValueError(
            f"metadata is for astrogwb {metadata.version}, but {__version__} "
            "is installed; regenerate the catalog"
        )
    logger.info(
        "Catalog: population=%s num_samples=%d kwargs=%s",
        metadata.population.model_name,
        metadata.num_samples,
        metadata.population.model_kwargs,
    )
    model = fix_extrinsic_parameters(
        joint_model(*metadata.population.build()(metadata.fiducials))
    )
    samples, _ = sample_and_logprob(
        model, key, num_samples=metadata.num_samples, density_sites=()
    )
    return polarization_power_data(
        waveform_generator(metadata.waveform), samples, chunk_size=chunk_size
    )


#: The redshift site every population must declare; its density is never
#: excluded.
REDSHIFT_SITE = "redshift"

#: Construction kwargs defining a population's redshift window.
REDSHIFT_WINDOW_KWARGS = ("minimum_redshift", "maximum_redshift")
