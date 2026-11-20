"""Importance-integration reference catalogs: per-source intensity.

:func:`intensity_data` turns a generator and source columns into
:class:`IntensityData`, one intensity column per source.
:func:`~astrogwb.background.importance.spectrum.reference_catalog` places caller-drawn
intrinsic sources at a reference redshift and returns a
:class:`ReferenceCatalog`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel

from astrogwb.waveform.base import WaveformGenerator

__all__ = [
    "IntensityData",
    "ReferenceCatalog",
    "intensity_data",
]

_logger = logging.getLogger(__name__)


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

    def take(self, n: int) -> ReferenceCatalog:
        """The first ``n`` draws, as a catalog.

        Parameters
        ----------
        n
            Number of draws to keep, at most the catalog's.

        Returns
        -------
        ReferenceCatalog
            A catalog whose per-draw arrays (``intensity``, ``intrinsic`` and
            ``proposal_log_prob``) are cut to ``n`` draws.

        Raises
        ------
        ValueError
            If ``n`` is negative or exceeds the number of draws.
        """
        size = int(np.shape(self.proposal_log_prob)[0])
        if not 0 <= n <= size:
            raise ValueError(f"cannot take {n} draws from a catalog of {size}")
        return replace(
            self,
            intensity=jnp.asarray(self.intensity)[:, :n],
            intrinsic={
                name: jnp.asarray(column)[:n] for name, column in self.intrinsic.items()
            },
            proposal_log_prob=jnp.asarray(self.proposal_log_prob)[:n],
        )

    def write(self, path: str | Path, metadata: BaseModel, **io_kwargs: Any) -> Path:
        """Write the catalog and the metadata that produced it, atomically.

        A thin wrapper of :func:`astrogwb.utils.io.write`, which needs the
        ``io`` extra.

        Parameters
        ----------
        path
            File path.
        metadata
            Metadata describing how the catalog was made.
        **io_kwargs
            Passed to :func:`astrogwb.utils.io.write` (``seed``, ``batch_size``).

        Returns
        -------
        pathlib.Path
            The file written.
        """
        from astrogwb.utils.io import write

        tree = {
            "frequencies": np.asarray(self.frequencies),
            "intensity": np.asarray(self.intensity),
            "intrinsic": {
                name: np.asarray(column) for name, column in self.intrinsic.items()
            },
            "proposal_log_prob": np.asarray(self.proposal_log_prob),
            "reference_redshift": np.float64(self.reference_redshift),
            "reference_distance": np.float64(self.reference_distance),
        }
        return write(path, tree, metadata, **io_kwargs)

    @classmethod
    def read[M: BaseModel](
        cls, path: str | Path, metadata_type: type[M]
    ) -> tuple[ReferenceCatalog, M]:
        """Read a catalog written by :meth:`write`.

        Parameters
        ----------
        path
            File path.
        metadata_type
            Pydantic model that validates the recorded metadata.

        Returns
        -------
        ReferenceCatalog
            The catalog, with NumPy array fields.
        M
            The metadata it was written with.
        """
        from astrogwb.utils.io import load

        tree, metadata, _ = load(path, metadata_type)
        data: dict[str, Any] = dict(tree)
        catalog = cls(
            frequencies=data["frequencies"],
            intensity=data["intensity"],
            intrinsic=dict(data["intrinsic"]),
            proposal_log_prob=data["proposal_log_prob"],
            reference_redshift=float(data["reference_redshift"]),
            reference_distance=float(data["reference_distance"]),
        )
        return catalog, metadata


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
    _logger.info(
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
