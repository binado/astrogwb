"""Grid posteriors with the k-mode shot noise against the rank-one amplitude mode.

The common-shape likelihood keeps one shot-noise mode; ``"modes"`` keeps the
leading eigenmodes of the whitened catalog covariance with the per-source
network response (``ImportanceGaussianLikelihood.with_shot_noise_modes``). For
every network of ``config/detectors.toml`` this script evaluates, on the data
and grids of ``notebooks/cosmology_grid_posteriors.py`` and
``notebooks/madau_dickinson_grid.py``,

- ``amplitude``: the rank-one mode with the sky-averaged power, s^2 at every
  point (the paper's likelihood);
- ``fixed+response``: the rank-one mode with the response's s^2, fixed at the
  fiducials;
- ``modes``: the k-mode covariance with the response.

It prints, per network: the 1D H0 posterior; the marginals of the
(H0, Omega_m) grid; and the Madau-Dickinson ridge-grid quantiles. Before the
Madau-Dickinson grids it checks that the fiducial basis still spans the catalog
covariance at the grid's corners (the largest whitened eigenvalue left outside
the basis, in units of the detector noise); above ``--drift-tolerance`` the
basis is rebuilt from the covariance summed over the corners and the
fiducials. The (H0, Omega_m) corners are checked the same way::

    uv run python scripts/shot_noise_modes_grids.py \
        --ensemble outputs/ensembles/fiducial \
        --reference-catalog outputs/catalogs/reference-fiducial.h5

It reads, never draws: draw 0 of seed 41 of the ensemble, its
``detector_noise.h5`` and the reference catalog, all written by
``workflow/coverage_draws.smk``.

The notebooks' helpers are imported from their setup blocks, so the grids are
the notebooks' own.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from astrogwb.paper.runtime import configure_runtime

configure_runtime(num_chains=1)

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from astrogwb.detector import sample_network_response
from astrogwb.inference import ImportanceGaussianLikelihood, Network, shot_noise_modes
from astrogwb.paper.catalogs import read_reference_catalog
from astrogwb.paper.config import detector_registry, fiducials, priors
from astrogwb.paper.config.runs import CATALOGS_ROOT, ENSEMBLES_ROOT
from astrogwb.paper.ensemble import network_spectra_ensemble
from astrogwb.paper.plotting import DETECTOR_NETWORKS
from astrogwb.paper.posteriors import network_data
from astrogwb.populations import build_population
from astrogwb.simulators import DetectorNoise

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import cosmology_grid_posteriors as cosmology
import madau_dickinson_grid as madau

# The configuration of the two grid notebooks.
DATA_SEED = 41
REDSHIFT_NODES = 32
MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY = 2.0, 2048.0
H0_AXIS = {"H0": (58.5, 77.0, 926)}
H0_OMEGA_AXES = {"H0": (59.0, 76.5, 351), "Omega_m": (0.2856, 0.3336, 41)}
GRID_CHUNK_SIZE = 8


def _setup(
    ensemble: Path, catalog_path: Path
) -> tuple[ImportanceGaussianLikelihood, dict[str, NDArray], DetectorNoise, dict]:
    """The notebooks' amplitude-mode likelihood, each network's data, and its noise."""
    params = fiducials()
    names = [name for name, _ in DETECTOR_NETWORKS]
    # Draw 0 of DATA_SEED: one catalog, as each network records it.
    drawn = network_spectra_ensemble(ensemble, DATA_SEED, 1, names)
    observed = {
        name: np.asarray(data["spectral_density"][0])
        for name, data in zip(names, drawn, strict=True)
    }
    noise = DetectorNoise.open(ensemble / "detector_noise.h5")
    catalog, catalog_metadata, _ = read_reference_catalog(catalog_path)
    population = catalog_metadata.population
    likelihood = ImportanceGaussianLikelihood.from_catalog(
        catalog,
        catalog_metadata,
        population=build_population(population.model_name, **population.model_kwargs),
        frequencies=noise.frequencies,
        num_redshift_nodes=REDSHIFT_NODES,
        density_sites=(),
        shot_noise="amplitude",
        observation_time=noise.metadata.observation_time,
    )
    return likelihood, observed, noise, params


def _whitened(
    likelihood: ImportanceGaussianLikelihood, params: dict, network: Network
) -> NDArray:
    """The whitened catalog covariance over the counted bins."""
    keep = np.asarray(network.mask)
    scale = np.asarray(network.scale)[keep]
    covariance = np.asarray(likelihood.catalog_covariance(params, network))
    return covariance[np.ix_(keep, keep)] / np.outer(scale, scale)


def _drift(
    likelihood: ImportanceGaussianLikelihood, corners: list[dict], network: Network
) -> tuple[float, float]:
    """Largest eigenvalue outside the basis and the worst captured trace fraction."""
    keep = np.asarray(network.mask)
    basis = np.asarray(network.modes)[keep]
    outside = np.eye(basis.shape[0]) - basis @ basis.T
    residual, captured = 0.0, 1.0
    for params in corners:
        whitened = _whitened(likelihood, params, network)
        residual = max(
            residual, float(np.linalg.eigvalsh(outside @ whitened @ outside)[-1])
        )
        captured = min(
            captured, float(np.trace(basis.T @ whitened @ basis) / np.trace(whitened))
        )
    return residual, captured


def _rebuilt(
    likelihood: ImportanceGaussianLikelihood,
    points: list[dict],
    network: Network,
    num_modes: int,
) -> Network:
    """The basis of the covariance summed over ``points``, with its node blocks."""
    covariance = sum(
        np.asarray(likelihood.catalog_covariance(params, network)) for params in points
    )
    modes = shot_noise_modes(
        covariance, network.scale, network.mask, max_modes=num_modes
    )
    network = network._replace(
        modes=jnp.pad(modes, ((0, 0), (0, num_modes - modes.shape[1])))
    )
    return network._replace(mode_blocks=likelihood.mode_blocks(points[0], network))


def _moments(grid: NDArray, density: NDArray) -> tuple[float, float]:
    weights = density / density.sum()
    mean = float(weights @ grid)
    return mean, float(np.sqrt(weights @ (grid - mean) ** 2))


def _marginals(grids: dict, log_density: NDArray) -> dict[str, tuple[float, float]]:
    density = np.exp(log_density - log_density.max())
    out = {}
    for axis, (name, grid) in enumerate(grids.items()):
        other = tuple(i for i in range(density.ndim) if i != axis)
        out[name] = _moments(grid, density.sum(axis=other) if other else density)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--response-draws", type=int, default=8192)
    parser.add_argument("--max-modes", type=int, default=32)
    parser.add_argument("--num-gamma", type=int, default=100)
    parser.add_argument("--num-kappa", type=int, default=60)
    parser.add_argument("--drift-tolerance", type=float, default=1e-3)
    parser.add_argument("--skip-madau", action="store_true")
    parser.add_argument("--skip-cosmology", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", default=["amplitude", "fixed+response", "modes"]
    )
    parser.add_argument("--problems", nargs="+", default=["H0", "H0_Omega_m"])
    parser.add_argument("--ensemble", type=Path, default=ENSEMBLES_ROOT / "fiducial")
    parser.add_argument(
        "--reference-catalog",
        type=Path,
        default=CATALOGS_ROOT / "reference-fiducial.h5",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    likelihood, observed, noise, params = _setup(args.ensemble, args.reference_catalog)
    frequencies = noise.frequencies
    registry = detector_registry()
    names = [name for name, _ in DETECTOR_NETWORKS]
    per_network, networks, sky_s2, response_s2 = {}, {}, [], []
    for name in names:
        data = network_data(
            noise,
            name,
            minimum_frequency=MINIMUM_FREQUENCY,
            maximum_frequency=MAXIMUM_FREQUENCY,
        )
        detectors, sensitivities = registry.build_network(name)
        plain = Network(jnp.asarray(data["scale"]), jnp.asarray(data["frequency_mask"]))
        responsive = plain._replace(
            response=sample_network_response(
                jax.random.key(0),
                args.response_draws,
                frequencies,
                detectors,
                sensitivities,
            )
        )
        network = likelihood.with_shot_noise_modes(
            params,
            responsive,
            tolerance=args.tolerance,
            max_modes=args.max_modes,
            num_modes=args.max_modes,
        )
        k = int(np.sum(np.any(np.asarray(network.modes) != 0.0, axis=0)))
        whitened = _whitened(likelihood, params, network)
        basis = np.asarray(network.modes)[np.asarray(network.mask)]
        dropped = np.trace(whitened) - np.trace(basis.T @ whitened @ basis)
        sky_s2.append(float(likelihood.relative_variance(params, plain)))
        response_s2.append(float(likelihood.relative_variance(params, responsive)))
        print(
            f"{name}: k = {k}, dropped = {dropped:.2e}, "
            f"lambda_1 = {np.linalg.eigvalsh(whitened)[-1]:.3e}, "
            f"s_sky = {np.sqrt(sky_s2[-1]):.3e}, "
            f"s_response = {np.sqrt(response_s2[-1]):.3e}",
            flush=True,
        )
        networks[name] = network
        per_network[name] = {
            **data,
            "response": network.response,
            "modes": network.modes,
            "mode_blocks": network.mode_blocks,
        }

    corners = [
        {**params, "H0": h0, "Omega_m": om}
        for h0 in (59.0, 76.5)
        for om in (0.2856, 0.3336)
    ]
    print("\n## Drift at the (H0, Omega_m) corners\n")
    print("| network | largest eigenvalue outside U | worst captured trace |")
    print("|---|---|---|")
    for name in names:
        residual, captured = _drift(likelihood, corners, networks[name])
        print(f"| {name} | {residual:.2e} | {captured:.8f} |", flush=True)

    all_variants = {
        "amplitude": (likelihood, None),
        "fixed+response": (
            dataclasses.replace(likelihood, shot_noise="fixed"),
            np.asarray(response_s2),
        ),
        "modes": (dataclasses.replace(likelihood, shot_noise="modes"), None),
    }
    variants = {name: all_variants[name] for name in args.variants}
    cosmology_priors = priors()
    problems = (
        ()
        if args.skip_cosmology
        else tuple(
            (title, axes)
            for key, title, axes in (
                ("H0", "H0", H0_AXIS),
                ("H0_Omega_m", "(H0, Omega_m)", H0_OMEGA_AXES),
            )
            if key in args.problems
        )
    )
    for title, axes in problems:
        fixed = {k: v for k, v in params.items() if k not in axes}
        results = {}
        for variant, (variant_likelihood, s2) in variants.items():
            grids, log_densities = cosmology.evaluate_grid(
                variant_likelihood,
                cosmology_priors,
                GRID_CHUNK_SIZE,
                axes,
                fixed,
                observed,
                per_network,
                s2,
            )
            results[variant] = {n: _marginals(grids, log_densities[n]) for n in names}
        print(f"\n## {title} grid: mean and sd of each marginal\n")
        header = " | ".join(f"{v} {p}" for p in axes for v in variants)
        print(f"| network | {header} |")
        print("|---|" + "---|" * (len(axes) * len(variants)))
        for name in names:
            cells = " | ".join(
                f"{results[v][name][p][0]:.4f} +/- {results[v][name][p][1]:.4f}"
                for p in axes
                for v in variants
            )
            print(f"| {name} | {cells} |", flush=True)

    if args.skip_madau:
        return
    madau_priors = priors(
        gamma={"dist": "Uniform", "kwargs": {"low": 0.0, "high": 10.0}},
        kappa={"dist": "Uniform", "kwargs": {"low": 0.0, "high": 10.0}},
    )
    z_range = (float(madau_priors["z_peak"].low), float(madau_priors["z_peak"].high))
    gamma_range = (float(madau_priors["gamma"].low), float(madau_priors["gamma"].high))
    kappa_range = (float(madau_priors["kappa"].low), float(madau_priors["kappa"].high))
    sampled = ("z_peak", "gamma", "kappa")
    # The ridge grid's notebook default: s^2 fixed at the fiducials, which keeps
    # the per-point projection (and its memory) out of the vmapped ridge solve.
    madau_variants = {
        "fixed": (
            dataclasses.replace(likelihood, shot_noise="fixed"),
            np.asarray(sky_s2),
        ),
        "fixed+response": all_variants["fixed+response"],
        "modes": all_variants["modes"],
    }

    def ridge(variant: str, name: str, network: Network) -> dict:
        variant_likelihood, s2 = madau_variants[variant]
        if s2 is not None:
            network = network._replace(
                relative_variance=jnp.asarray(s2[names.index(name)])
            )
        if variant == "modes":
            # The held blocks carry the response; the draws themselves are idle.
            network = network._replace(response=None)
        else:
            network = Network(network.scale, network.mask, network.relative_variance)
        bound = dataclasses.replace(
            variant_likelihood, observed=jnp.asarray(observed[name]), network=network
        )
        return madau.ridge_grid(
            bound,
            params,
            gamma_range,
            kappa_range,
            z_range,
            args.num_gamma,
            args.num_kappa,
        )

    print("\n## Madau-Dickinson: drift at the ridge-grid corners\n")
    print("| network | outside U, fiducial basis | worst captured | rebuilt |")
    print("|---|---|---|---|")
    baseline_grids = {}
    for name in names:
        baseline_grids[name] = ridge("fixed", name, networks[name])
        grid = baseline_grids[name]
        corner_points = [
            {
                **params,
                "gamma": grid["gamma"][i],
                "kappa": grid["kappa"][j],
                "z_peak": grid["z_star"][i, j],
            }
            for i in (0, -1)
            for j in (0, -1)
        ]
        residual, captured = _drift(likelihood, corner_points, networks[name])
        rebuilt = ""
        if residual > args.drift_tolerance:
            networks[name] = _rebuilt(
                likelihood, [params, *corner_points], networks[name], args.max_modes
            )
            after, _ = _drift(likelihood, corner_points, networks[name])
            rebuilt = f"yes, outside U then {after:.2e}"
        print(f"| {name} | {residual:.2e} | {captured:.6f} | {rebuilt} |", flush=True)

    print("\n## Madau-Dickinson ridge grids: median and half 68% width\n")
    header = " | ".join(f"{v} {p}" for p in sampled for v in madau_variants)
    print(f"| network | {header} |")
    print("|---|" + "---|" * (len(sampled) * len(madau_variants)))
    for name in names:
        grids = {
            "fixed": baseline_grids[name],
            "fixed+response": ridge("fixed+response", name, networks[name]),
            "modes": ridge("modes", name, networks[name]),
        }
        cells = []
        for parameter in sampled:
            for variant in madau_variants:
                q = madau.weighted_quantiles(grids[variant], parameter)
                cells.append(f"{q[2]:.4f} +/- {(q[3] - q[1]) / 2:.4f}")
        print(f"| {name} | {' | '.join(cells)} |", flush=True)


if __name__ == "__main__":
    main()
