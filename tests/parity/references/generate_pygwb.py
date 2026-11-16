# /// script
# requires-python = "==3.11.*"
# # pygwb 1.5.1 extends a gwpy attribute that became a tuple in gwpy 4.
# dependencies = ["pygwb==1.5.1", "gwpy<4"]
# ///
"""Freeze the pygwb reference values the parity suite compares against.

pygwb pins ``scipy<1.14`` and cannot share an environment with this package,
so it runs here, alone, from the inline metadata above::

    uv run --no-project tests/parity/references/generate_pygwb.py

Writes ``pygwb.npz`` next to this file. Every array is either an input both
sides read (frequencies, site coordinates, noise curves, cross spectra) or a
pygwb output. Nothing of this package is imported: the only shared input it
produces is ``coba_bns_events.json`` (``draw_coba_events.py``).

Sections, by key prefix:

``site_*`` and ``orf_*``
    Overlap reduction functions. ``exact`` sites sit on the equator or a pole,
    where the geodetic vertical is the geocentric one, the case pygwb's
    spherical-Earth formula is exact for. ``real`` sites are bilby's H1, L1,
    V1 and K1.
``psd``, ``csd``, ``point_estimate``, ``sigma``, ``combined_*``
    The optimal estimator of each baseline of the exact network and their
    inverse-variance combination.
``likelihood_*``
    The power-law log likelihood of the same network.
``cbc_*``
    The energy spectrum of the CoBA BNS events with IMRPhenomD.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import bilby
import numpy as np
import pygwb
import scipy
from bilby.gw.detector.geometry import InterferometerGeometry
from pygwb.background import compute_Omega_from_CBC_dictionary
from pygwb.constants import H0
from pygwb.orfs import calc_orf
from pygwb.pe import PowerLawModel
from pygwb.postprocessing import (
    calculate_point_estimate_sigma_spectra,
    combine_spectra_with_sigma_weights,
)
from pygwb.util import omega_to_power

HERE = Path(__file__).parent
OUTPUT = HERE / "pygwb.npz"
EVENTS = HERE / "coba_bns_events.json"

#: Uniform grid shared by the ORF, estimator and likelihood sections, in Hz.
FREQUENCIES = np.arange(4, 1604) * 0.25

#: Exact sites, in bilby's convention: degrees, metres, arm azimuths
#: counter-clockwise from East. Columns: latitude, longitude, elevation,
#: x-arm azimuth, y-arm azimuth. Sorted by name, the pair order of the tests.
EXACT_SITES: dict[str, tuple[float, float, float, float, float]] = {
    "A": (0.0, 0.0, 0.0, 30.0, 120.0),
    "B": (0.0, 40.0, 500.0, 75.0, 165.0),
    "C": (90.0, 0.0, 0.0, 10.0, 100.0),
}
REAL_SITES = ("H1", "K1", "L1", "V1")

#: The estimator's analysis segment. The boxcar window has unit window
#: factors, so pygwb's variance carries no window correction.
SEGMENT_DURATION = 4096.0
SAMPLE_RATE = 4.0
WINDOW = {"window_fftgram": "boxcar"}
REFERENCE_FREQUENCY = 25.0

#: Power-law points the likelihood is evaluated at: (omega_ref, alpha).
LIKELIHOOD_POINTS = np.array(
    [[0.0, 0.0], [2e-9, 0.0], [5e-9, 2.0 / 3.0], [8e-9, 2.0 / 3.0], [5e-9, 3.0]]
)
INJECTED_POINT = (5e-9, 2.0 / 3.0)

CBC_SAMPLING_FREQUENCY = 4096
CBC_WAVEFORM_DURATION = 8
CBC_MINIMUM_FREQUENCY = 20.0
CBC_MAXIMUM_FREQUENCY = 1024.0
CBC_OBSERVATION_SECONDS = 1000.0


def exact_site_vectors() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Vertex and arm unit vectors of the exact sites, from bilby."""
    vectors = {}
    for name, (latitude, longitude, elevation, xarm, yarm) in EXACT_SITES.items():
        geometry = InterferometerGeometry(
            length=4.0,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            xarm_azimuth=xarm,
            yarm_azimuth=yarm,
        )
        vectors[name] = (geometry.vertex, geometry.x, geometry.y)
    return vectors


def real_site_vectors() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Vertex and arm unit vectors of bilby's bundled interferometers."""
    vectors = {}
    for name in REAL_SITES:
        detector = bilby.gw.detector.get_empty_interferometer(name)
        vectors[name] = (detector.vertex, detector.x, detector.y)
    return vectors


def pair_orfs(
    vectors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Tensor ORF of every sorted pair on ``FREQUENCIES``, ``(P, F)``."""
    rows = []
    for first, second in combinations(sorted(vectors), 2):
        vertex_1, x_1, y_1 = vectors[first]
        vertex_2, x_2, y_2 = vectors[second]
        rows.append(
            calc_orf(FREQUENCIES, vertex_1, vertex_2, x_1, x_2, y_1, y_2, "tensor")
        )
    return np.stack(rows)


def noise_curves() -> np.ndarray:
    """One smooth one-sided strain PSD per exact site, ``(3, F)``, in 1/Hz."""
    knees = np.array([[30.0, 400.0], [45.0, 250.0], [20.0, 600.0]])
    floors = np.array([1.0e-46, 2.5e-46, 6.0e-47])
    low = (knees[:, :1] / FREQUENCIES) ** 4
    high = (FREQUENCIES / knees[:, 1:]) ** 2
    return floors[:, None] * (1.0 + low + high)


def estimator(
    orfs: np.ndarray, psds: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Per-baseline estimator spectra and their combination."""
    omega_ref, alpha = INJECTED_POINT
    omega = omega_ref * (FREQUENCIES / REFERENCE_FREQUENCY) ** alpha
    signal_power = np.asarray(omega_to_power(omega, FREQUENCIES))
    bin_width = FREQUENCIES[1] - FREQUENCIES[0]
    pairs = list(combinations(range(psds.shape[0]), 2))
    csds, estimates, sigmas = [], [], []
    for orf, (first, second) in zip(orfs, pairs, strict=True):
        noise_scale = np.sqrt(
            psds[first] * psds[second] / (2.0 * SEGMENT_DURATION * bin_width)
        )
        csd = orf * signal_power + noise_scale * rng.standard_normal(orf.shape)
        estimate, variance = calculate_point_estimate_sigma_spectra(
            FREQUENCIES,
            csd,
            psds[first],
            psds[second],
            orf,
            SAMPLE_RATE,
            SEGMENT_DURATION,
            window_fftgram_dict=WINDOW,
            overlap_factor=0.0,
            fref=REFERENCE_FREQUENCY,
            alpha=0.0,
        )
        csds.append(csd)
        estimates.append(np.asarray(estimate))
        sigmas.append(np.sqrt(np.asarray(variance)))
    combined, combined_sigma = combine_spectra_with_sigma_weights(estimates, sigmas)
    return {
        "csd": np.stack(csds),
        "point_estimate": np.stack(estimates),
        "sigma": np.stack(sigmas),
        "combined_point_estimate": np.asarray(combined),
        "combined_sigma": np.asarray(combined_sigma),
    }


def likelihood(
    orfs: np.ndarray, estimates: np.ndarray, sigmas: np.ndarray
) -> np.ndarray:
    """pygwb's power-law log likelihood at ``LIKELIHOOD_POINTS``."""
    baselines = [
        SimpleNamespace(
            tensor_overlap_reduction_function=orf,
            point_estimate_spectrum=estimate,
            sigma_spectrum=sigma,
            frequencies=FREQUENCIES,
            frequency_mask=None,
            calibration_epsilon=0,
        )
        for orf, estimate, sigma in zip(orfs, estimates, sigmas, strict=True)
    ]
    model = PowerLawModel(
        baselines=baselines, model_name="power_law", fref=REFERENCE_FREQUENCY
    )
    values = []
    for omega_ref, alpha in LIKELIHOOD_POINTS:
        model.parameters = {"omega_ref": omega_ref, "alpha": alpha}
        values.append(model.log_likelihood())
    return np.array(values)


def cbc_spectrum() -> dict[str, np.ndarray]:
    """Energy spectrum of the CoBA events, restricted to the compared band."""
    events = json.loads(EVENTS.read_text())["events"]
    redshift = np.array(events["redshift"])
    spin_1 = np.array(events["spin_1z"])
    spin_2 = np.array(events["spin_2z"])
    count = redshift.shape[0]
    injections = {
        "mass_1": np.array(events["source_frame_mass_1"]) * (1.0 + redshift),
        "mass_2": np.array(events["source_frame_mass_2"]) * (1.0 + redshift),
        "luminosity_distance": np.array(events["luminosity_distance"]),
        "theta_jn": np.array(events["inclination"]),
        "phase": np.array(events["coa_phase"]),
        "a_1": np.abs(spin_1),
        "a_2": np.abs(spin_2),
        "tilt_1": np.where(spin_1 < 0.0, np.pi, 0.0),
        "tilt_2": np.where(spin_2 < 0.0, np.pi, 0.0),
        "phi_12": np.zeros(count),
        "phi_jl": np.zeros(count),
    }
    frequencies, omega = compute_Omega_from_CBC_dictionary(
        injections,
        CBC_SAMPLING_FREQUENCY,
        CBC_OBSERVATION_SECONDS,
        waveform_duration=CBC_WAVEFORM_DURATION,
        waveform_approximant="IMRPhenomD",
        waveform_reference_frequency=CBC_MINIMUM_FREQUENCY,
        waveform_minimum_frequency=CBC_MINIMUM_FREQUENCY,
    )
    band = (frequencies >= CBC_MINIMUM_FREQUENCY) & (
        frequencies <= CBC_MAXIMUM_FREQUENCY
    )
    return {
        "cbc_frequencies": frequencies[band],
        "cbc_omega_gw": omega[band],
        "cbc_observation_seconds": np.array(CBC_OBSERVATION_SECONDS),
        "cbc_sampling_frequency": np.array(float(CBC_SAMPLING_FREQUENCY)),
        "cbc_reference_frequency": np.array(CBC_MINIMUM_FREQUENCY),
    }


def main() -> None:
    """Compute every section and write the archive."""
    rng = np.random.default_rng(20261116)
    exact = exact_site_vectors()
    exact_orfs = pair_orfs(exact)
    psds = noise_curves()
    spectra = estimator(exact_orfs, psds, rng)
    unit_omega = np.ones_like(FREQUENCIES)
    np.savez_compressed(
        OUTPUT,
        provenance=np.array(
            f"pygwb {pygwb.__version__}, bilby {bilby.__version__}, "
            f"numpy {np.__version__}, scipy {scipy.__version__}"
        ),
        hubble_constant=np.array(H0.value),
        frequencies=FREQUENCIES,
        site_exact_names=np.array(sorted(EXACT_SITES)),
        site_exact_coordinates=np.array([EXACT_SITES[n] for n in sorted(EXACT_SITES)]),
        site_real_names=np.array(sorted(REAL_SITES)),
        orf_exact=exact_orfs,
        orf_real=pair_orfs(real_site_vectors()),
        power_per_unit_omega=np.asarray(omega_to_power(unit_omega, FREQUENCIES)),
        psd=psds,
        segment_duration=np.array(SEGMENT_DURATION),
        reference_frequency=np.array(REFERENCE_FREQUENCY),
        likelihood_points=LIKELIHOOD_POINTS,
        likelihood_log_values=likelihood(
            exact_orfs, spectra["point_estimate"], spectra["sigma"]
        ),
        **spectra,
        **cbc_spectrum(),
    )


if __name__ == "__main__":
    main()
