"""Does the common-shape shot-noise model survive the per-source network response?

The common-shape likelihood keeps one shot-noise mode, the amplitude along the
template. With the sky-averaged power the whitened catalog covariance is
rank one to a very good approximation: its leading eigenvalue is
``zeta^2 = rho^2 sigma_B^2`` and the rest are orders of magnitude below the
detector noise. The per-source response ``r(n, psi, iota; f)`` varies across
the band where the ORFs do (separated pairs, CE pairs, the misaligned ET 2L
pair), and a frequency-dependent ``r`` feeds the modes perpendicular to the
template.

For every network of ``config/detectors.toml``, at the fiducial parameters,
``T = 1`` yr and ``z_min = 0.35``, this script whitens the catalog covariance
by the detector noise, ``C = N^{-1/2} Cov N^{-1/2}``, with and without the
response (``ImportanceGaussianLikelihood.catalog_covariance`` and the
Hadamard form of ``Network.response``), and prints the leading eigenvalue,
``lambda_2 / lambda_1``, the sum of the others, ``kappa^2 = e^T C e`` and the
largest perpendicular eigenvalue ``lambda_max(P C P)`` with
``P = I - e e^T``. Eigenvalues are in units of the detector-noise variance.

The response's second moment is a Monte Carlo average over ``--draws`` draws,
whose noise also lands in the perpendicular modes; ``--draws`` is run at two
values so the converged part can be told apart from that noise (it falls as
``1 / sqrt(L)``)::

    uv run python scripts/common_shape_response.py --draws 2048 8192

The reference catalog is the cached one of ``notebooks/shot_noise_coverage.py``
(drawn on a miss).
"""

from __future__ import annotations

import argparse

from astrogwb.paper.runtime import configure_runtime

jax, _ = configure_runtime(num_chains=1)

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from astrogwb.detector import (
    effective_psd,
    gaussian_bin_scale,
    sample_network_response,
)
from astrogwb.frequency import frequency_mask
from astrogwb.gwb.catalog import CatalogMetadata
from astrogwb.gwb.importance import (
    reference_catalog,
    reference_catalog_stem,
)
from astrogwb.inference import ImportanceGaussianLikelihood, Network
from astrogwb.paper.config import (
    detector_registry,
    fiducials,
    population_metadata,
    waveform_metadata,
)
from astrogwb.paper.config.runs import CATALOGS_ROOT
from astrogwb.paper.plotting import DETECTOR_NETWORKS
from astrogwb.populations import build_population
from astrogwb.simulators import batch_keys
from astrogwb.utils.io import load, write

# The configuration of notebooks/shot_noise_coverage.py.
APPROXIMANT = "IMRPhenomXAS_NRTidalv3"
MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT = 0.35, 20.0
CATALOG_SEED = 42
NUM_SAMPLES = 2**14
REDSHIFT_NODES = 32
OBSERVATION_TIME = 1.0  # years
MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY = 2.0, 2048.0


def _likelihood() -> tuple[ImportanceGaussianLikelihood, NDArray[np.float64], dict]:
    """The notebook's amplitude-mode likelihood, its grid and the fiducials."""
    params = fiducials()
    waveform = waveform_metadata(
        approximant=APPROXIMANT,
        frequency_resolution=0.125,
        turnover_frequency=100.0,
        dlogf=0.01266,
    )
    frequencies = np.asarray(waveform.build().frequencies)
    population = population_metadata(
        minimum_redshift=MINIMUM_REDSHIFT, maximum_redshift=MAXIMUM_REDSHIFT
    )
    metadata = CatalogMetadata(
        waveform=waveform_metadata(
            approximant=APPROXIMANT,
            frequency_spacing="log",
            minimum_frequency=MINIMUM_FREQUENCY,
            maximum_frequency=4000.0,
            frequency_resolution=0.02,
            turnover_frequency=None,
        ),
        population=population,
        fiducials=params,
        num_samples=NUM_SAMPLES,
        sampling="sobol",
    )
    path = CATALOGS_ROOT / f"{reference_catalog_stem(metadata, CATALOG_SEED)}.h5"
    if path.is_file():
        data, _, _ = load(path, CatalogMetadata)
    else:
        data = reference_catalog(metadata, batch_keys(CATALOG_SEED, 1)[0])
        write(path, data, metadata, seed=CATALOG_SEED)
    likelihood = ImportanceGaussianLikelihood.from_catalog(
        data,  # ty: ignore[invalid-argument-type]
        metadata,
        population=build_population(population.model_name, **population.model_kwargs),
        frequencies=frequencies,
        num_redshift_nodes=REDSHIFT_NODES,
        density_sites=(),
        shot_noise="amplitude",
        observation_time=OBSERVATION_TIME,
    )
    return likelihood, frequencies, params


def _spectrum(covariance: NDArray, template: NDArray, scale: NDArray) -> dict:
    """Eigen-summary of the whitened covariance over the counted bins."""
    whitened = covariance / np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(whitened)[::-1]
    signal = template / scale
    e = signal / np.linalg.norm(signal)
    projector = np.eye(e.size) - np.outer(e, e)
    return {
        "lambda_1": eigenvalues[0],
        "lambda_2/lambda_1": eigenvalues[1] / eigenvalues[0],
        "sum_k>=2": np.sum(eigenvalues[1:]),
        "kappa^2": e @ whitened @ e,
        "lambda_max(PCP)": np.linalg.eigvalsh(projector @ whitened @ projector)[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    likelihood, frequencies, params = _likelihood()
    template = np.asarray(likelihood.spectrum(params))
    sky_averaged = np.asarray(likelihood.catalog_covariance(params))
    registry = detector_registry()
    grid = jnp.asarray(frequencies)

    columns = [
        "lambda_1",
        "lambda_2/lambda_1",
        "sum_k>=2",
        "kappa^2",
        "lambda_max(PCP)",
    ]
    print("| network | model | " + " | ".join(columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for name, _ in DETECTOR_NETWORKS:
        detectors = registry.build_network(name)
        psd = jnp.asarray(effective_psd(grid, detectors))
        mask = np.asarray(
            frequency_mask(grid, fmin=MINIMUM_FREQUENCY, fmax=MAXIMUM_FREQUENCY)
            & jnp.isfinite(psd)
            & (psd > 0.0)
        )
        scale = np.asarray(
            jnp.where(mask, gaussian_bin_scale(psd, OBSERVATION_TIME, grid), 1.0)
        )
        keep = mask & (template > 0.0)
        models = {"sky average": sky_averaged}
        for num_draws in args.draws:
            response = sample_network_response(
                jax.random.key(args.seed),
                num_draws,
                frequencies,
                detectors,
            )
            network = Network(jnp.asarray(scale), jnp.asarray(mask), response=response)
            models[f"response, L={num_draws}"] = np.asarray(
                likelihood.catalog_covariance(params, network)
            )
        for model, covariance in models.items():
            summary = _spectrum(
                covariance[np.ix_(keep, keep)], template[keep], scale[keep]
            )
            values = " | ".join(f"{summary[column]:.3e}" for column in columns)
            print(f"| {name} | {model} | {values} |")


if __name__ == "__main__":
    main()
