"""``scripts/simulate_spectra.py`` reads its ``SpectraMetadata`` from config layers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from repo import REPO_ROOT

from astrogwb.metadata import PriorSpec, SpectraMetadata, artifact_path
from astrogwb.paper.config import (
    _table,
    fiducials,
    population_metadata,
    waveform_metadata,
)
from astrogwb.paper.config.runs import base_config_paths

SPECTRUM_LAYER = REPO_ROOT / "config" / "simulations" / "spectrum" / "default.toml"


def _import_script(name: str):
    """Import one ``scripts/*.py``, which are not installed modules."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _argv(*layers: Path) -> list[str]:
    argv: list[str] = []
    for path in (*base_config_paths(REPO_ROOT), *layers):
        argv += ["--config", str(path)]
    return argv


def test_the_committed_layer_is_the_record_built_in_python() -> None:
    """Merged with the shared layers, it keys exactly what the accessors build."""
    script = _import_script("simulate_spectra")
    args = script.parse_args(_argv(SPECTRUM_LAYER))
    metadata = script.spectra_metadata(script.load_merged_config(args))

    fixed = {
        name: value
        for name, value in fiducials(REPO_ROOT).items()
        if name not in {"xi_0", "xi_n", "local_merger_rate"}
    }
    expected = SpectraMetadata(
        waveform=waveform_metadata(REPO_ROOT),
        population=population_metadata(REPO_ROOT),
        hyperparameters={
            **fixed,
            "local_merger_rate": PriorSpec.model_validate(
                _table(REPO_ROOT, "priors", "local_merger_rate")
            ),
        },
        num_draws=64,
        observation_time=1.0,
    )

    assert metadata == expected
    assert metadata.key() == expected.key()


def test_a_priors_reference_is_sampled_and_a_fiducials_one_is_fixed() -> None:
    script = _import_script("simulate_spectra")
    args = script.parse_args(_argv(SPECTRUM_LAYER))
    metadata = script.spectra_metadata(script.load_merged_config(args))

    assert set(metadata.sampled) == {"local_merger_rate"}
    assert metadata.fixed["H0"] == fiducials(REPO_ROOT)["H0"]


def test_the_output_is_named_by_the_derived_key() -> None:
    script = _import_script("simulate_spectra")
    args = script.parse_args(_argv(SPECTRUM_LAYER) + ["--output-dir", "somewhere"])
    metadata = script.spectra_metadata(script.load_merged_config(args))

    assert artifact_path(metadata, args.output_dir) == Path(
        "somewhere", f"{metadata.key()}.h5"
    )


def test_layers_without_a_spectra_table_are_rejected_naming_it() -> None:
    script = _import_script("simulate_spectra")
    args = script.parse_args(_argv())

    with pytest.raises(ValueError, match=r"no \[spectra\] table"):
        script.spectra_metadata(script.load_merged_config(args))


def test_a_stray_key_in_the_table_is_rejected(tmp_path: Path) -> None:
    layer = tmp_path / "stray.toml"
    layer.write_text(SPECTRUM_LAYER.read_text() + "\n[spectra.extra]\nx = 1.0\n")
    script = _import_script("simulate_spectra")
    args = script.parse_args(_argv(layer))

    with pytest.raises(ValueError, match="invalid \\[spectra\\] table"):
        script.spectra_metadata(script.load_merged_config(args))
