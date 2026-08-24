from __future__ import annotations

import numpy as np
import xarray as xr


def polarization_power(catalog: xr.Dataset) -> xr.DataArray:
    """Reduce a loaded waveform catalog to polarization power ``|h+|^2 + |hx|^2``.

    Returns a ``(frequency, sample)`` float64 DataArray for the inference stack.
    """
    power = (np.abs(catalog.polarizations) ** 2).sum("polarization")
    return power.transpose("frequency", "sample").astype(np.float64)
