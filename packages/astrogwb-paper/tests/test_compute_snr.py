from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from astrogwb.resolved import REQUIRED_SOURCE_PARAMETERS
from astrogwb_paper.cli.compute_snr import load_source_parameters


def _write_catalog(
    path: Path,
    *,
    keys: Sequence[str] | None = None,
    attrs: Mapping[str, Any] | None = None,
    n_events: int = 3,
    with_source_parameters: bool = True,
) -> Path:
    with h5py.File(path, "w") as catalog:
        if with_source_parameters:
            group = catalog.create_group("source_parameters")
            for key in keys if keys is not None else REQUIRED_SOURCE_PARAMETERS:
                group.create_dataset(key, data=np.arange(n_events, dtype=np.float64))
        if attrs is not None:
            for name, value in attrs.items():
                catalog.attrs[name] = value
    return path


def test_load_source_parameters_rejects_missing_group(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "empty.h5", with_source_parameters=False)

    with pytest.raises(SystemExit, match="source_parameters"):
        load_source_parameters(path)


def test_load_source_parameters_rejects_missing_keys(tmp_path: Path) -> None:
    path = _write_catalog(
        tmp_path / "partial.h5",
        keys=("detector_frame_mass_1", "detector_frame_mass_2", "luminosity_distance"),
    )

    with pytest.raises(SystemExit, match="coa_time"):
        load_source_parameters(path)


def test_load_source_parameters_copies_root_attributes(tmp_path: Path) -> None:
    path = _write_catalog(
        tmp_path / "catalog.h5",
        attrs={"approximant": "IMRPhenomXAS_NRTidalv3", "sampling_frequency": 1024.0},
    )

    source_parameters, root_attributes = load_source_parameters(path)

    assert set(REQUIRED_SOURCE_PARAMETERS) <= set(source_parameters)
    assert source_parameters["coa_time"].shape == (3,)
    np.testing.assert_array_equal(source_parameters["coa_time"], np.arange(3.0))
    assert root_attributes["approximant"] == "IMRPhenomXAS_NRTidalv3"
    assert root_attributes["sampling_frequency"] == 1024.0
