"""Check the Hadamard catalog covariance against direct network simulation.

The likelihood's catalog covariance under a per-source network response is the
Hadamard product of the sky-averaged second moment of the power and the
response correlation :math:`\\langle r_f r_{f'} \\rangle`
(:class:`~astrogwb.inference.likelihood.Network`). This script tests that
identity by simulation rather than by the factorization argument.

With Poisson counts, Campbell's theorem makes the covariance of a spectrum
exactly the rate over time times the second moment of one source's
contribution. Both simulators draw the same sources for a key, so

.. math::

    \\mathrm{Cov}[S^{\\mathrm{net}}]
        = \\mathrm{Cov}[S^{\\mathrm{sky}}] \\circ
          \\langle r r^T \\rangle\\, \\frac{\\langle g \\rangle^2}{\\langle g^2 \\rangle},

where :math:`\\langle g \\rangle^2/\\langle g^2 \\rangle = 252/355` removes the
inclination scatter the sky-averaged power carries and :math:`r` carries
instead. The script draws ``--draws`` realizations of both
:class:`~astrogwb.simulators.spectra.NetworkSpectralDensitySimulator` and
:class:`~astrogwb.simulators.spectra.BackgroundSpectralDensitySimulator` on the
closed-form inspiral. It compares the leading eigenvalues of the covariance
whitened by each network's detector noise, :math:`C_{ff'}/(\\sigma_f \\sigma_{f'})`
as the likelihood sees it, from the network draws and from the Hadamard
prediction, with the Monte Carlo spread from the two halves of the draws. Both
the catalog and the noise variance scale as :math:`1/T`, so the whitened
eigenvalues do not depend on ``--observation-time``; they are the
shot-noise-to-noise ratios of a full observation of this population.

    uv run python scripts/network_spectra_hadamard.py --draws 4000
"""

from __future__ import annotations

import argparse
import tomllib

import jax
import numpy as np
from numpy.typing import NDArray

from astrogwb.detector import (
    effective_psd,
    gaussian_bin_scale,
    sample_network_response,
)
from astrogwb.detector.sensitivity import SENSITIVITY_FILE
from astrogwb.gwb.importance import INCLINATION_SECOND_MOMENT
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import batch_keys
from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityMetadata,
    BackgroundSpectralDensitySimulator,
    NetworkSpectralDensityMetadata,
    NetworkSpectralDensitySimulator,
    stack_spectra,
)
from astrogwb.waveform import WaveformMetadata

NETWORKS = {
    "ET-2L-aligned": ["S1", "R1"],
    "ET-2L-misaligned-CE-Hanford": ["S2", "R2", "C1"],
    "ET-triangular-CE-Hanford": ["E1", "E2", "E3", "C1"],
}
HYPERPARAMETERS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}
NUM_EIGENVALUES = 3


def _background_metadata(observation_time: float) -> BackgroundSpectralDensityMetadata:
    """Poisson draws on a log grid, the redshift window starting at 0.35."""
    return BackgroundSpectralDensityMetadata(
        waveform=WaveformMetadata(
            approximant="AnalyticInspiral",
            frequency_spacing="log",
            minimum_frequency=5.0,
            maximum_frequency=500.0,
            frequency_resolution=0.5,
            reference_frequency=5.0,
            sampling_frequency=2048.0,
        ),
        population=PopulationMetadata(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": 0.35,
                "maximum_redshift": 5.0,
                "n_grid": 256,
            },
        ),
        hyperparameters=HYPERPARAMETERS,
        observation_time=observation_time,
        count="poisson",
    )


def _whitened_eigenvalues(covariance: NDArray, scale: NDArray) -> NDArray:
    """Leading eigenvalues of ``covariance`` whitened by the noise ``scale``."""
    return np.linalg.eigvalsh(covariance / np.outer(scale, scale))[::-1][
        :NUM_EIGENVALUES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--observation-time", type=float, default=5e-5)
    parser.add_argument("--response-draws", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(SENSITIVITY_FILE, "rb") as handle:
        references = {
            name: row["psd_reference"]
            for name, row in tomllib.load(handle)["detectors"].items()
        }
    background_metadata = _background_metadata(args.observation_time)
    background = BackgroundSpectralDensitySimulator(background_metadata)
    keys = batch_keys(args.seed, args.draws)
    sky = stack_spectra([background(key) for key in keys])
    spectra_sky = sky["spectral_density"]
    frequencies = sky["frequencies"]
    print(
        f"{args.draws} draws, {sky['n_events'].mean():.1f} sources per draw, "
        f"{frequencies.size} bins; whitened eigenvalues, detector-noise units"
    )

    for name, detectors in NETWORKS.items():
        metadata = NetworkSpectralDensityMetadata(
            **background_metadata.model_dump(),
            detectors=detectors,
            psd_references={detector: references[detector] for detector in detectors},
        )
        simulator = NetworkSpectralDensitySimulator(metadata)
        spectra_net = stack_spectra([simulator(key) for key in keys])[
            "spectral_density"
        ]
        response = np.asarray(
            sample_network_response(
                jax.random.key(args.seed + 1),
                args.response_draws,
                sky["frequencies"],
                detectors,
                metadata.sensitivities(),
            )
        )
        correlation = response.T @ response / response.shape[0]
        sky_covariance = np.cov(spectra_sky, rowvar=False)
        predicted = sky_covariance * correlation / INCLINATION_SECOND_MOMENT
        psd = effective_psd(frequencies, detectors, metadata.sensitivities())
        scale = np.asarray(gaussian_bin_scale(psd, args.observation_time, frequencies))
        keep = np.isfinite(scale) & (scale > 0.0)
        scale = scale[keep]
        net = spectra_net[:, keep]
        halves = [
            _whitened_eigenvalues(np.cov(part, rowvar=False), scale)
            for part in np.array_split(net, 2)
        ]
        print(f"\n{name}")
        print(
            f"  sky average {_whitened_eigenvalues(sky_covariance[np.ix_(keep, keep)], scale)}"
        )
        print(
            f"  simulated   {_whitened_eigenvalues(np.cov(net, rowvar=False), scale)}"
        )
        print(f"  halves      {halves[0]}  {halves[1]}")
        print(
            f"  Hadamard    {_whitened_eigenvalues(predicted[np.ix_(keep, keep)], scale)}"
        )


if __name__ == "__main__":
    main()
