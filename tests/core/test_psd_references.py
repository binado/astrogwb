"""Reference resolution shared by runtime PSD loading and configuration."""

from pathlib import Path

import pytest

from astrogwb.psd import NOISE_CURVES_BASE_DIR, resolve_psd_path


def test_resolution_preserves_presets_packaged_files_local_files_and_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # Presets and packaged curves take precedence over local names.
    (tmp_path / "ET_D_psd").write_text("local placeholder")
    (tmp_path / "ET_COBA_10km_psd.txt").write_text("local placeholder")
    preset = resolve_psd_path("ET_D_psd")
    assert isinstance(preset, Path) and preset.is_file()
    assert preset.resolve() != tmp_path / "ET_D_psd"
    assert (
        resolve_psd_path("ET_COBA_10km_psd.txt")
        == NOISE_CURVES_BASE_DIR / "ET_COBA_10km_psd.txt"
    )
    (tmp_path / "data").mkdir()
    path = Path("data/custom.txt")
    path.write_text("10 1e-46\n20 2e-46\n")
    assert resolve_psd_path(path) == path
    assert resolve_psd_path(path.resolve()) == path.resolve()
    assert (
        resolve_psd_path("https://example.org/noise.txt")
        == "https://example.org/noise.txt"
    )
    with pytest.raises(FileNotFoundError):
        resolve_psd_path("missing.txt")
