"""Plot helpers in ``scripts/fiducial_spectrum.py``.

The script is not an installed module, so tests load it from the checkout the
same way ``test_catalog_generation`` loads the catalog generator.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.detector import gaussian_bin_scale
from astrogwb.paper.plotting import Network

matplotlib = pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def _disable_usetex() -> None:
    """Paper style enables usetex; CI paper tests do not install LaTeX."""
    matplotlib.pyplot.rcParams["text.usetex"] = False


@pytest.fixture(scope="module")
def fiducial_spectrum() -> ModuleType:
    """Import ``scripts/fiducial_spectrum.py``."""
    path = REPO_ROOT / "scripts" / "fiducial_spectrum.py"
    spec = importlib.util.spec_from_file_location("fiducial_spectrum_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_band(module: ModuleType) -> tuple[np.ndarray, ...]:
    frequencies = np.linspace(2.0, 256.0, 64)
    spectral_density = 1.0e-45 * (frequencies / 10.0) ** (-7.0 / 3.0)
    effective_psd = 1.0e-43 * (1.0 + (frequencies / 30.0) ** 2)
    mask = frequencies <= 200.0
    freq, omega, sh, seff = module.band_limited_spectrum(
        frequencies,
        spectral_density,
        mask,
        h0=67.66,
        effective_psd_arr=effective_psd,
    )
    assert seff is not None
    return freq, omega, sh, seff, frequencies[1] - frequencies[0]


def test_network_matching_detectors_selects_the_run_network(
    fiducial_spectrum: ModuleType,
) -> None:
    networks = (
        Network("ET-2L-aligned", "ET-2L-par", ("S1", "R1")),
        Network("ET-2L-aligned-CE-Hanford", r"ET-2L-par $+$ CE", ("S1", "R1", "C1")),
    )
    chosen = fiducial_spectrum.network_matching_detectors(networks, ("S1", "R1", "C1"))
    assert chosen.name == "ET-2L-aligned-CE-Hanford"


def test_network_matching_detectors_rejects_missing_and_duplicate(
    fiducial_spectrum: ModuleType,
) -> None:
    networks = (
        Network("a", "A", ("S1", "R1")),
        Network("b", "B", ("S1", "R1")),
    )
    with pytest.raises(ValueError, match="no compared network"):
        fiducial_spectrum.network_matching_detectors(networks, ("C1",))
    with pytest.raises(ValueError, match="multiple compared networks"):
        fiducial_spectrum.network_matching_detectors(networks, ("S1", "R1"))


def test_snr_cumulative_endpoints_match_total_snr(
    fiducial_spectrum: ModuleType,
) -> None:
    _freq, _omega, sh, seff, df = _synthetic_band(fiducial_spectrum)
    snr_squared, snr_lt, snr_gt = fiducial_spectrum.snr_integrand_and_cumulative(
        sh, seff, observation_time_sec=1.0e7, df=df
    )
    total = float(np.sqrt(np.sum(snr_squared)))
    np.testing.assert_allclose(snr_lt[-1], total)
    np.testing.assert_allclose(snr_gt[0], total)
    np.testing.assert_allclose(snr_lt[0], np.sqrt(snr_squared[0]))
    np.testing.assert_allclose(snr_gt[-1], np.sqrt(snr_squared[-1]))
    assert np.all(np.diff(snr_lt) >= -1.0e-12)
    assert np.all(np.diff(snr_gt) <= 1.0e-12)


def test_overlay_and_snr_figures_have_expected_axes(
    fiducial_spectrum: ModuleType,
) -> None:
    freq, omega, sh, seff, df = _synthetic_band(fiducial_spectrum)
    snr_squared, snr_lt, snr_gt = fiducial_spectrum.snr_integrand_and_cumulative(
        sh, seff, observation_time_sec=1.0e7, df=df
    )

    overlay = fiducial_spectrum.plot_omega_sh_and_sigma(
        freq,
        omega,
        sh,
        np.asarray(gaussian_bin_scale(jnp.asarray(seff), 1.0, float(df))),
        omega_gw_min=1.0e-15,
        sigma_label=r"$\sigma$",
    )
    assert isinstance(overlay, matplotlib.figure.Figure)
    assert len(overlay.axes) == 2
    assert sum(len(axis.lines) for axis in overlay.axes) == 3
    matplotlib.pyplot.close(overlay)

    snr_fig = fiducial_spectrum.plot_snr_cumulative(freq, snr_squared, snr_lt, snr_gt)
    assert isinstance(snr_fig, matplotlib.figure.Figure)
    assert len(snr_fig.axes) == 3
    assert all(len(axis.lines) == 1 for axis in snr_fig.axes)
    matplotlib.pyplot.close(snr_fig)

    combined = fiducial_spectrum.plot_spectrum_and_cumulative_snr(
        freq, omega, sh, snr_lt, snr_gt, omega_gw_min=1.0e-15
    )
    assert isinstance(combined, matplotlib.figure.Figure)
    assert len(combined.axes) == 3
    assert sum(len(axis.lines) for axis in combined.axes) == 4
    matplotlib.pyplot.close(combined)
