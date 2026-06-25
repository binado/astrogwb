"""High-level analysis bundle tying a gwmock network to sensitivities.

``analysis_setup`` is the one-call entry point: give it a network name (or
a ready ``Network``) and it resolves the detector geometry from gwmock and
the matching :class:`Sensitivity` curves from astrogwb's
``sensitivity.toml``, ready for ORF and effective-PSD queries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gwmock_signal.network import Network
from gwmock_signal.stochastic.overlap import detector_names
from numpy.typing import ArrayLike, NDArray

from ._types import DetectorSpec
from .overlap import effective_psd, overlap_reduction_function
from .sensitivity import Sensitivity, load_sensitivity_map


@dataclass(frozen=True)
class AnalysisContext:
    """A network plus the sensitivity curve for each of its detectors."""

    network: Network
    sensitivities: Mapping[str, Sensitivity]

    @property
    def detector_names(self) -> tuple[DetectorSpec, ...]:
        """The network's detector specs (str codes and/or CustomDetectors)."""
        return tuple(self.network.detector_names)

    def overlap(
        self, frequencies: ArrayLike, det_a: DetectorSpec, det_b: DetectorSpec
    ) -> NDArray[np.float64]:
        """Frequency-dependent ORF between two of the network's detectors."""
        return overlap_reduction_function(frequencies, det_a, det_b)

    def effective_psd(self, frequencies: ArrayLike) -> NDArray[np.float64]:
        """Network effective PSD over ``frequencies``."""
        return effective_psd(
            frequencies, self.network.detector_names, self.sensitivities
        )


def analysis_setup(
    network: Network | str, *, sensitivity_path: str | Path | None = None
) -> AnalysisContext:
    """Build an :class:`AnalysisContext` from a network name or instance.

    String arguments are resolved via :meth:`Network.from_name` (see
    ``Network.list_names()`` for the available presets). Sensitivities are
    loaded from ``sensitivity.toml`` (or ``sensitivity_path``) keyed by the
    network's public detector names.
    """
    if isinstance(network, str):
        network = Network.from_name(network)

    names = detector_names(network.detector_names)
    sensitivities = load_sensitivity_map(names, path=sensitivity_path)
    return AnalysisContext(network=network, sensitivities=sensitivities)
