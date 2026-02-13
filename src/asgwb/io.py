from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd

if TYPE_CHECKING:
    from pandas.io.parsers import TextFileReader


@overload
def load_injection_file(
    path: Path, iterator: bool = False, **kwargs
) -> pd.DataFrame: ...


@overload
def load_injection_file(
    path: Path, iterator: bool = True, **kwargs
) -> TextFileReader: ...


def load_injection_file(
    path: Path, iterator: bool, chunksize: int | None = None, **kwargs
) -> pd.DataFrame | TextFileReader:
    csv_kwargs = kwargs.copy()
    csv_kwargs.update(
        {
            "sep": " ",
            "header": 0,
            "engine": "c",
            "iterator": iterator,
            "chunksize": chunksize,
        }
    )
    return pd.read_csv(path, **csv_kwargs)
