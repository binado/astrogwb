"""Importance-integration reference catalogs: per-source polarization power.

:class:`CatalogMetadata` is the record a catalog is drawn from and cached
under. :func:`polarization_power_data` turns a generator and source columns
into :class:`PolarizationPowerData`, one power column per source; where the
sources come from is the caller's. :func:`draw_catalog` is the plain catalog --
the population a record names, drawn at its fiducials from one key -- and
:func:`~astrogwb.gwb.importance.reference_catalog` draws at the redshift
window's lower edge. :func:`restrict_redshift` narrows a plain catalog and its
record together: dropping samples without narrowing the recorded density would
leave it normalized over a window the samples no longer span.

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
from astrogwb.populations import joint_model
from astrogwb.populations.evaluation import sample_sources
from astrogwb.populations.metadata import PopulationMetadata, widen_model_kwargs
from astrogwb.utils.io import content_key
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = [
    "REDSHIFT_SITE",
    "REDSHIFT_WINDOW_KWARGS",
    "CatalogMetadata",
    "PolarizationPowerData",
    "catalog_stem",
    "draw_catalog",
    "polarization_power_data",
    "restrict_redshift",
]

logger = logging.getLogger(__name__)


class CatalogMetadata(BaseModel):
    """Everything that determines a polarization-power catalog's contents.

    It is both the request and the provenance: the
    :func:`~astrogwb.gwb.catalog.draw_catalog`
    turns it and a key into a catalog, the draw is stored beside it
    (:func:`~astrogwb.utils.io.write`),
    and :meth:`key` is the middle part of the file name by convention
    (:func:`catalog_stem`). The seed is not part of it: it picks one realization of the
    density this record describes.

    The waveform and population record alone are not enough: two catalogs
    with the same ones still differ if they were drawn at other
    hyperparameters or at another size, and both differ if the code that drew
    them changed. ``fiducials`` and ``num_samples`` close the first gap and
    ``version`` the second.

    Every caller -- the workflow, the generator script, a notebook -- derives
    the key from this one record, so there is one canonical form and no second
    hash to drift.

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

    def key(self) -> str:
        """The content hash this catalog is cached under.

        Hashes the canonical JSON of the whole record. Numeric construction
        kwargs are widened to ``float``, so ``2`` and ``2.0`` name the same
        catalog; boolean choices retain their type. Everything else is
        already type-stable under strict validation.

        The payload nests ``waveform`` and ``population`` under ``metadata``,
        the shape this record had before it was flattened, so every catalog
        already on disk keeps its key. For the same reason the default
        ``sampling="random"`` and ``reference_redshift=None`` are left out of
        the payload.
        """
        payload = self.model_dump(mode="json")
        if payload["sampling"] == "random":
            del payload["sampling"]
        if payload["reference_redshift"] is None:
            del payload["reference_redshift"]
        widen_model_kwargs(payload["population"])
        payload["metadata"] = {
            "waveform": payload.pop("waveform"),
            "population": payload.pop("population"),
        }
        return content_key(payload)


def catalog_stem(metadata: CatalogMetadata, seed: int | np.integer) -> str:
    """The file stem a catalog is kept under by convention: its record and seed.

    ``polarization_power-<key>-<seed>``. A naming convention for the workflow and
    notebooks, not an address: the file itself records the metadata and seed it
    was drawn at, which a reader checks.
    """
    return f"polarization_power-{metadata.key()}-{int(seed)}"


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

    Values are checked once, on the concrete sources: generation is trace-safe
    and trusts its inputs, so a source carrying a degree of freedom the
    approximant cannot represent would otherwise be silently dropped. The
    columns are passed through as arrays; ``chunk_size`` is
    :meth:`~astrogwb.waveform.WaveformGenerator.generate_batch`'s.
    """
    # Power is float64 throughout; see draw_catalog for why sources too.
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
    # Generators are deliberately jit-free so they compose inside NumPyro
    # models. This is the opposite case -- eager and whole-catalog -- and at
    # production sizes an unfused graph would hold every waveform intermediate
    # at once, so the jit belongs here.
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

    The key is an input, so one record serves any number of independent
    realizations.
    """
    # x64 must be on before the draw. The population is sampled before the
    # Ripple-backed generator runs, and importing ripplegw -- which turns this
    # on globally -- happens only inside that generator, so relying on it would
    # draw every source column in float32.
    jax.config.update("jax_enable_x64", True)
    if metadata.version != __version__:
        raise ValueError(
            f"metadata is for astrogwb {metadata.version}, but {__version__} "
            "is installed; a catalog is generated by the code its key names"
        )
    logger.info(
        "Catalog %s: population=%s num_samples=%d kwargs=%s",
        metadata.key(),
        metadata.population.model_name,
        metadata.num_samples,
        metadata.population.model_kwargs,
    )
    model = joint_model(*metadata.population.build()(metadata.fiducials))
    samples = sample_sources(model, key, num_samples=metadata.num_samples)
    return polarization_power_data(
        metadata.waveform.build(), samples, chunk_size=chunk_size
    )


#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

#: Construction kwargs a population model must take for a draw from it to
#: support :func:`restrict_redshift`. Narrowing the window changes the
#: *normalization* of the generating density, so the arrays and the model kwargs
#: have to move together or the recorded density stops describing the samples.
REDSHIFT_WINDOW_KWARGS = ("minimum_redshift", "maximum_redshift")


def restrict_redshift(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    minimum_redshift: float,
    maximum_redshift: float,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """Restrict a draw to sources inside a redshift window, narrowing the population.

    Both halves move together, which is the whole reason this is one function:
    dropping samples without narrowing the generating model would leave the
    recorded density normalized over a window the samples no longer span, and
    every importance weight would be off by that normalization. Draws truncated
    to a sub-window follow the same law as draws made directly from it, so only
    the support changes.

    Returns a new ``(data, metadata)`` pair; the inputs are untouched. The
    returned metadata records the narrowed window and the kept sample count, so
    it no longer names a cached draw -- it is derived from one.

    Raises
    ------
    ValueError
        If the population takes no redshift window, the window is not inside
        the generation support, or no sample falls in it.
    """
    population = metadata.population
    model_kwargs = population.model_kwargs
    missing = [name for name in REDSHIFT_WINDOW_KWARGS if name not in model_kwargs]
    if missing:
        raise ValueError(
            f"population {population.model_name!r} takes no "
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

    redshift = np.asarray(data["source_parameters"][REDSHIFT_SITE])
    keep = np.flatnonzero(
        (redshift >= minimum_redshift) & (redshift <= maximum_redshift)
    )
    if keep.size == 0:
        raise ValueError(
            f"catalog has no samples in the "
            f"redshift window [{minimum_redshift:.4g}, {maximum_redshift:.4g}]"
        )
    restricted = PolarizationPowerData(
        frequencies=data["frequencies"],
        polarization_power=np.asarray(data["polarization_power"])[:, keep],
        source_parameters={
            name: np.asarray(values)[keep]
            for name, values in data["source_parameters"].items()
        },
    )
    narrowed = metadata.model_copy(
        update={
            "population": population.with_model_kwargs(
                minimum_redshift=float(minimum_redshift),
                maximum_redshift=float(maximum_redshift),
            ),
            "num_samples": int(keep.size),
        }
    )
    return restricted, narrowed
