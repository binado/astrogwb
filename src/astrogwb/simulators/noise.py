r"""Detector noise of the background stage's networks, on its frequency grid.

A realization's background holds signal only; the noise a search sees is a
property of the networks and the observation, not of the draw.
:func:`detector_noise` evaluates it once per ensemble, for the networks,
frequencies and observation time a background was simulated with, so the two
cannot drift apart. Per network it holds

- ``pair_overlap`` -- the isotropic ORF :math:`\gamma_{ab}` of each pair,
  ``(P_n, F)``;
- ``pair_scale`` -- the per-pair Gaussian scale of
  :func:`~astrogwb.detector.pair_bin_scale`, ``(P_n, F)``;
- ``effective_psd`` -- the network's :func:`~astrogwb.detector.effective_psd`,
  ``(F,)``;
- ``scale`` -- its :func:`~astrogwb.detector.gaussian_bin_scale`, ``(F,)``;

the pairs in the order of
:meth:`~astrogwb.simulators.DetectorRegistry.pairs`, the background's.
Frequency bands and padding are analysis choices and are left to the reader.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from astrogwb.detector.config import DetectorRegistry
from astrogwb.simulators.background import scientific_payload, select_networks
from astrogwb.utils.io import content_key, load, write

__all__ = [
    "DetectorNoise",
    "DetectorNoiseMetadata",
    "NetworkNoise",
    "detector_noise",
]

#: The layout version of the metadata; a file of another version is regenerated.
FORMAT_VERSION = 2


class DetectorNoiseMetadata(BaseModel):
    """What a :class:`DetectorNoise` is evaluated from, besides its frequencies.

    ``networks`` are the recorded network names, and
    ``observation_time`` is in years.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    format_version: int = Field(strict=True)
    networks: tuple[str, ...]
    scientific_identity: str
    observation_time: Annotated[float, Field(gt=0.0)]

    def key(self) -> str:
        """The content hash of the record."""
        return content_key(self.model_dump(mode="json"))


class NetworkNoise(NamedTuple):
    """One network's noise; see the module docstring for each field."""

    pair_overlap: NDArray[np.float64]
    pair_scale: NDArray[np.float64]
    effective_psd: NDArray[np.float64]
    scale: NDArray[np.float64]


@dataclass(frozen=True)
class DetectorNoise:
    """Every network's :class:`NetworkNoise` on ``frequencies``.

    ``noise`` is keyed as ``metadata.networks``.
    """

    metadata: DetectorNoiseMetadata
    frequencies: NDArray[np.float64]
    noise: dict[str, NetworkNoise]

    def network(self, name: str) -> NetworkNoise:
        """The noise of the network ``name``."""
        return self.noise[name]

    def write(self, path: str | Path) -> Path:
        """Write the noise to the HDF5 file ``path``, atomically.

        Datasets ``frequencies`` and ``networks/<name>/<field>``, the metadata
        as a JSON attribute.
        """
        tree = {
            "frequencies": self.frequencies,
            "networks": {name: noise._asdict() for name, noise in self.noise.items()},
        }
        return write(path, tree, self.metadata)

    @classmethod
    def open(cls, path: str | Path, *, registry: DetectorRegistry) -> Self:
        """Read the noise :meth:`write` wrote to ``path``.

        ``registry`` is trusted to be the one the noise was evaluated with,
        and only gives the pair order and shapes; the recorded identity is not
        compared with it.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file uses another format version, or its arrays do not
            match the recorded networks.
        """
        data, metadata, _ = load(path, DetectorNoiseMetadata)
        if metadata.format_version != FORMAT_VERSION:
            raise ValueError(f"{path} uses old detector noise metadata; regenerate it")
        networks = data["networks"]
        frequencies = np.asarray(data["frequencies"], dtype=np.float64)
        noise = {
            name: NetworkNoise(
                **{
                    field: np.asarray(networks[name][field])
                    for field in NetworkNoise._fields
                }
            )
            for name in metadata.networks
        }
        for name, values in noise.items():
            shape = (len(registry.pairs(name)), frequencies.size)
            if (
                values.pair_overlap.shape != shape
                or values.pair_scale.shape != shape
                or values.effective_psd.shape != (frequencies.size,)
                or values.scale.shape != (frequencies.size,)
            ):
                raise ValueError(
                    f"{path}: noise arrays for {name!r} have invalid shapes"
                )
        return cls(
            metadata=metadata,
            frequencies=frequencies,
            noise=noise,
        )


def detector_noise(
    frequencies: ArrayLike,
    registry: DetectorRegistry,
    observation_time: float,
    *,
    network_names: Sequence[str] | None = None,
) -> DetectorNoise:
    """The noise of selected registry networks on ``frequencies``.

    Parameters
    ----------
    frequencies
        The background's full frequency grid in Hz; bin widths come from it.
    registry
        The frozen detector registry used for the realization.
    observation_time
        Observation time in years.
    network_names
        Select recorded networks; defaults to all registry networks.

    Returns
    -------
    DetectorNoise
        Each network's noise, pairs in the order of its ``pairs()``.

    Raises
    ------
    ValueError
        If a selected network has fewer than two detectors.
    """
    import jax.numpy as jnp

    from astrogwb.detector import (
        effective_psd,
        gaussian_bin_scale,
        pair_bin_scale,
        pair_terms,
    )

    grid = np.asarray(frequencies, dtype=np.float64)
    names = select_networks(registry, network_names)
    noise = {}
    for name in names:
        detectors, sensitivities = registry.build_network(name)
        psd = effective_psd(grid, detectors, sensitivities)
        noise[name] = NetworkNoise(
            pair_overlap=np.stack(
                [term.overlap for term in pair_terms(grid, detectors, sensitivities)]
            ),
            pair_scale=np.asarray(
                pair_bin_scale(grid, detectors, sensitivities, observation_time)
            ),
            effective_psd=np.asarray(psd),
            scale=np.asarray(
                gaussian_bin_scale(jnp.asarray(psd), observation_time, grid)
            ),
        )
    return DetectorNoise(
        metadata=DetectorNoiseMetadata(
            format_version=FORMAT_VERSION,
            networks=names,
            scientific_identity=content_key(scientific_payload(registry, names)),
            observation_time=float(observation_time),
        ),
        frequencies=grid,
        noise=noise,
    )
