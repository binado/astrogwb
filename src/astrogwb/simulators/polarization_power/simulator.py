"""Per-source waveform power as a catalog.

:func:`polarization_power_data` turns a generator and source columns into
:class:`PolarizationPowerData`, one power column per source; where the sources
come from is the caller's. A plain catalog draws them at the fiducials with
:func:`~astrogwb.populations.evaluation.sample_sources`, and
:func:`~astrogwb.gwb.importance.reference_sources` draws them at the redshift
window's lower edge.
:func:`~astrogwb.simulators.polarization_power.restrict_redshift` narrows a
plain catalog and its record together.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial
from typing import Any, TypedDict

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.waveform.generator.base import PolarizationPowerGenerator

__all__ = ["PolarizationPowerData", "polarization_power_data"]

logger = logging.getLogger(__name__)


class PolarizationPowerData(TypedDict):
    """One catalog: ``frequencies`` ``(F,)``, ``polarization_power`` ``(F, N)``.

    ``source_parameters`` columns are each ``(N,)``.
    """

    frequencies: NDArray[np.float64]
    polarization_power: NDArray[np.float64]
    source_parameters: dict[str, NDArray[Any]]


def polarization_power_data(
    generator: PolarizationPowerGenerator,
    source_parameters: Mapping[str, ArrayLike],
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The power of ``source_parameters`` as a catalog, generated eagerly.

    Values are checked once, on the concrete sources: generation is trace-safe
    and trusts its inputs, so a source carrying a degree of freedom the
    approximant cannot represent would otherwise be silently dropped. The
    columns are passed through as arrays; ``chunk_size`` is
    :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`'s.
    """
    # Power is float64 throughout. Callers draw sources first and must turn x64
    # on themselves: importing ripplegw, which does it globally, happens only
    # inside the generator, after the draw.
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
