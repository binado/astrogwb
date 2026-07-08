from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pluscross import WaveformCatalog


def polarization_power(catalog: WaveformCatalog) -> NDArray[np.float64]:
    """Reduce a loaded waveform catalog to polarization power ``|h+|^2 + |hx|^2``.

    Returns a ``(nfreq, nsamples)`` float64 array for the inference stack.
    """
    power = np.abs(catalog.plus) ** 2 + np.abs(catalog.cross) ** 2
    return np.asarray(power.T, dtype=np.float64)
