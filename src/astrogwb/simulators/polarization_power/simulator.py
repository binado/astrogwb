"""Per-source waveform power, and the catalog draw built on it.

:class:`PolarizationPowerSimulator` knows only a waveform: it turns source
parameters into :class:`PolarizationPowerData`, one power column per source.
Where the sources come from is the caller's. :func:`draw_catalog` is the plain
catalog -- the population a
:class:`~astrogwb.simulators.polarization_power.CatalogMetadata` names, drawn
at its fiducials from one key -- and :mod:`astrogwb.gwb.importance` composes
the same simulator over redshift nodes.
:func:`~astrogwb.simulators.polarization_power.restrict_redshift` narrows a
plain catalog and its record together.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypedDict

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb import __version__
from astrogwb.populations.evaluation import sample_sources
from astrogwb.simulators.polarization_power.metadata import CatalogMetadata
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["PolarizationPowerData", "PolarizationPowerSimulator", "draw_catalog"]

logger = logging.getLogger(__name__)


class PolarizationPowerData(TypedDict):
    """One catalog: ``frequencies`` ``(F,)``, ``polarization_power`` ``(F, N)``.

    ``source_parameters`` columns are each ``(N,)``.
    """

    frequencies: NDArray[np.float64]
    polarization_power: NDArray[np.float64]
    source_parameters: dict[str, NDArray[Any]]


class PolarizationPowerSimulator:
    """Waveform power of given sources for one waveform; build once, call often.

    Deterministic: ``simulator(source_parameters)`` returns the power of every
    source, ``(F, N)``, with the columns passed through as arrays. Values are
    checked once, eagerly, on the concrete sources: generation is trace-safe
    and trusts its inputs, so a source carrying a degree of freedom the
    approximant cannot represent would otherwise be silently dropped.

    ``chunk_size`` is forwarded to :func:`jax.lax.map` as ``batch_size``,
    bounding peak memory to ``chunk_size`` sources' worth of waveform
    intermediates; it changes cost, and the output only at rounding level.
    ``None`` generates every source in one vectorized call.
    """

    def __init__(
        self, metadata: WaveformMetadata, *, chunk_size: int | None = None
    ) -> None:
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        # Power is float64 throughout; see draw_catalog for why sources too.
        jax.config.update("jax_enable_x64", True)
        self._metadata = metadata
        self._chunk_size = chunk_size
        self._generator = metadata.build()
        # Generators are deliberately jit-free so they compose inside NumPyro
        # models. This simulator is the opposite case -- eager and
        # whole-catalog -- and at production sizes an unfused graph would hold
        # every waveform intermediate at once, so the jit belongs here.
        self._power = jax.jit(self._generate)

    @property
    def metadata(self) -> WaveformMetadata:
        """The waveform this simulator generates."""
        return self._metadata

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> PolarizationPowerData:
        """The power of ``source_parameters``, one ``(N,)`` column per name."""
        generator = self._generator
        columns = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        count = len(next(iter(columns.values())))
        logger.info(
            "Generating %s waveforms for %d events (f_min=%.1f Hz, f_max=%.1f Hz)",
            generator.metadata.approximant,
            count,
            generator.metadata.minimum_frequency,
            generator.metadata.maximum_frequency,
        )
        generator.check_sources(columns)
        return {
            "frequencies": np.asarray(generator.frequencies),
            "polarization_power": np.asarray(self._power(columns)),
            "source_parameters": columns,
        }

    def _generate(self, columns: Mapping[str, jax.Array]) -> jax.Array:
        generate_batch = self._generator.generate_batch
        if self._chunk_size is None:
            return generate_batch(columns)

        def one(source: Mapping[str, jax.Array]) -> jax.Array:
            return generate_batch(
                {name: value[None] for name, value in source.items()}
            )[:, 0]

        # ``lax.map`` vmaps ``one`` over ``chunk_size`` sources at a time and
        # handles the remainder itself, inside the one compiled call.
        return jax.lax.map(one, dict(columns), batch_size=self._chunk_size).T


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
    _, model = metadata.population.build()(metadata.fiducials)
    samples = sample_sources(model, key, num_samples=metadata.num_samples)
    simulator = PolarizationPowerSimulator(metadata.waveform, chunk_size=chunk_size)
    return simulator(samples)
