from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from astrogwb.waveform import load_catalog, make_catalog, save_catalog
from astrogwb_paper.cli import compute_snr


def _write_catalog(path: Path) -> Path:
    catalog = make_catalog(
        frequencies=np.array([10.0, 20.0]),
        polarization_power=np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
            ]
        ),
        source_parameters={
            "redshift": np.array([0.1, 0.2, 0.3]),
            "detector_frame_mass_1": np.array([1.1, 1.2, 1.3]),
        },
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=20.0,
        reference_frequency=20.0,
        sampling_frequency=64.0,
    )
    save_catalog(path, catalog)
    return path


def _args(catalog: Path) -> argparse.Namespace:
    return argparse.Namespace(
        catalog=catalog,
        detectors="H1,L1",
        batch_size=2,
        progress_log_every=1000,
        backend="lal",
        no_earth_rotation=False,
    )


def _patch_detector_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compute_snr,
        "resolve_detector",
        lambda name: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(
        compute_snr,
        "load_sensitivity_map",
        lambda names: {name: object() for name in names},
    )


def test_main_enriches_and_reorders_catalog_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_catalog(tmp_path / "catalog.h5")
    snrs = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    captured: dict[str, Any] = {}
    monkeypatch.setattr(compute_snr, "parse_args", lambda: _args(path))
    _patch_detector_loading(monkeypatch)

    def fake_optimal_snr(*args: object, **kwargs: object) -> np.ndarray:
        captured.update(kwargs)
        return snrs

    monkeypatch.setattr(compute_snr, "optimal_snr", fake_optimal_snr)

    compute_snr.main()

    loaded = load_catalog(path)
    order = np.array([0, 2, 1])
    np.testing.assert_array_equal(loaded.detector.values, ["H1", "L1"])
    np.testing.assert_allclose(loaded.snr.values, snrs[order])
    np.testing.assert_allclose(
        loaded.polarization_power.values,
        np.array([[10.0, 30.0, 20.0], [11.0, 31.0, 21.0]]),
    )
    np.testing.assert_allclose(
        loaded.source_parameters.sel(parameter="redshift").values,
        [0.1, 0.3, 0.2],
    )
    assert "sample_index" not in loaded
    assert loaded.attrs["snr_batch_size"] == 2
    assert captured["batch_size"] == 2
    assert captured["earth_rotation"] is True


def test_main_failure_preserves_catalog_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_catalog(tmp_path / "catalog.h5")
    original = path.read_bytes()
    monkeypatch.setattr(compute_snr, "parse_args", lambda: _args(path))
    _patch_detector_loading(monkeypatch)
    monkeypatch.setattr(
        compute_snr,
        "optimal_snr",
        lambda *args, **kwargs: np.ones((3, 2)),
    )
    monkeypatch.setattr(
        compute_snr,
        "save_catalog",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )

    with pytest.raises(RuntimeError, match="write failed"):
        compute_snr.main()

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".catalog.h5.*.tmp")) == []
