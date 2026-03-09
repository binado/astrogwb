from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .grid import FrequencyGrid


@dataclass
class WaveformPolarizations:
    """Plus and cross polarization arrays on a frequency grid."""

    grid: FrequencyGrid
    plus: npt.NDArray[np.complex128]
    cross: npt.NDArray[np.complex128]

    def dump(self, path: Path, parameters: dict[str, float] | None = None) -> None:
        from .io import dump_waveform_npz

        dump_waveform_npz(path=path, polarizations=self, parameters=parameters)

    @classmethod
    def load(
        cls, path: Path, grid: FrequencyGrid | None = None
    ) -> tuple[WaveformPolarizations, dict[str, float]]:
        from .io import load_waveform_npz

        return load_waveform_npz(path=path, grid=grid)

    def squared_sum(self) -> npt.NDArray[np.float64]:
        out = np.square(self.plus.real, dtype=np.float64)
        tmp_buf = np.empty_like(out, dtype=out.dtype)
        for buf in (
            self.plus.imag,
            self.cross.real,
            self.cross.imag,
        ):
            np.square(buf, out=tmp_buf)
            out += tmp_buf
        return out
