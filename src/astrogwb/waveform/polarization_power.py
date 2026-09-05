from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def polarization_power(
    plus: NDArray[np.complex128], cross: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """Reduce ``(n, F)`` complex polarization arrays to ``(F, n)`` float64 power.

    Returns ``|h+|^2 + |hx|^2`` per sample and frequency, transposed to the
    on-disk ``(frequency, sample)`` layout.
    """
    power = np.abs(plus) ** 2 + np.abs(cross) ** 2
    return power.T.astype(np.float64)
