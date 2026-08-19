"""Assemble production populations from finite ``gwmock-pop`` source pools."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_distance_and_logprob,
)
from gwmock_pop.cosmology.flat_lambda_cdm import DEFAULT_LOOKUP_GRID_SIZE
from gwmock_pop.loaders.file_loader import (
    read_population_catalogue,
    write_population_catalogue,
)

from astrogwb_paper.catalogs import PROPOSAL_REDSHIFT_LOGPDF
from astrogwb_paper.config.catalogs import ASSEMBLY_OPERATIONS, AssemblyOperation
from astrogwb_paper.config.loading import load_mapping

PROPOSAL_COMPONENT = "proposal_component"

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a production population by copying, subsampling, or "
            "categorically mixing finite gwmock-pop source catalogs."
        )
    )
    parser.add_argument(
        "--operation",
        choices=ASSEMBLY_OPERATIONS,
        required=True,
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="Source population path; repeat in component order.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=float,
        help="Mixture component weight; repeat in source order.",
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-config",
        action="append",
        type=Path,
        help=(
            "gwmock-pop graph config for each source. Two configs plus "
            "--uniform-redshift-fraction attach the guarded proposal density."
        ),
    )
    parser.add_argument("--uniform-redshift-fraction", type=float)
    return parser.parse_args(argv)


def assemble_population(
    sources: Sequence[Mapping[str, np.ndarray]],
    *,
    operation: AssemblyOperation,
    num_samples: int,
    seed: int,
    weights: Sequence[float] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Assemble rows and return ``(population, component_assignments)``."""
    if not sources:
        raise ValueError("at least one source population is required")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    columns = tuple(sources[0])
    if not columns:
        raise ValueError("source population is empty")
    for index, source in enumerate(sources):
        if tuple(source) != columns:
            raise ValueError(
                f"source {index} columns do not match the first source population"
            )
        _population_size(source)

    if operation in {"identity", "subsample"}:
        if len(sources) != 1:
            raise ValueError(f"{operation} requires exactly one source population")
        source_size = _population_size(sources[0])
        if num_samples > source_size:
            raise ValueError(
                f"requested {num_samples} rows from source with {source_size} rows"
            )
        if operation == "identity":
            if num_samples != source_size:
                raise ValueError(
                    "identity requires num_samples to equal the source population size"
                )
            indices = np.arange(source_size)
        else:
            indices = np.random.default_rng(seed).choice(
                source_size, size=num_samples, replace=False
            )
        return (
            {name: np.asarray(values)[indices] for name, values in sources[0].items()},
            np.zeros(num_samples, dtype=np.int64),
        )

    if operation != "mixture":
        raise ValueError(f"unsupported assembly operation {operation!r}")
    if len(sources) < 2:
        raise ValueError("mixture requires at least two source populations")
    probabilities = _normalize_weights(weights, len(sources))
    rng = np.random.default_rng(seed)
    assignments = rng.choice(len(sources), size=num_samples, p=probabilities)
    output = {
        name: np.empty(num_samples, dtype=np.asarray(values).dtype)
        for name, values in sources[0].items()
    }
    for component, source in enumerate(sources):
        positions = np.flatnonzero(assignments == component)
        source_size = _population_size(source)
        if positions.size > source_size:
            raise ValueError(
                f"component {component} needs {positions.size} rows but its source "
                f"population contains only {source_size}"
            )
        indices = rng.choice(source_size, size=positions.size, replace=False)
        for name, values in source.items():
            output[name][positions] = np.asarray(values)[indices]
    return output, assignments


def guarded_proposal_logpdf(
    redshift: np.ndarray,
    *,
    fiducial_config: Mapping[str, Any],
    uniform_config: Mapping[str, Any],
    epsilon: float,
) -> np.ndarray:
    """Evaluate the analytic fiducial-plus-uniform proposal log-density."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("uniform_redshift_fraction must satisfy 0 < epsilon < 1")
    fiducial_parameters = _parameters(fiducial_config)
    uniform_parameters = _parameters(uniform_config)
    _validate_component_graphs(fiducial_parameters, uniform_parameters)

    fiducial_sampler = _sampler(fiducial_parameters, "redshift")
    if fiducial_sampler["function"] != "madau_dickinson_redshift":
        raise ValueError("fiducial redshift sampler must be 'madau_dickinson_redshift'")
    uniform_sampler = _sampler(uniform_parameters, "redshift")
    if uniform_sampler["function"] != "uniform":
        raise ValueError("uniform redshift sampler must be 'uniform'")

    fiducial_args = dict(fiducial_sampler.get("arguments") or {})
    uniform_args = dict(uniform_sampler.get("arguments") or {})
    z_min = float(fiducial_args.pop("z_min", 0.0))
    z_max = float(fiducial_args.pop("z_max"))
    uniform_min = float(uniform_args["minimum"])
    uniform_max = float(uniform_args["maximum"])
    if not (
        math.isclose(z_min, uniform_min, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(z_max, uniform_max, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("fiducial and uniform redshift supports must match exactly")

    n_grid = int(fiducial_args.pop("n_grid", DEFAULT_LOOKUP_GRID_SIZE))
    try:
        params = {
            "H0": float(fiducial_args.pop("hubble_constant")),
            "Omega_m": float(fiducial_args.pop("omega_m")),
            "gamma": float(fiducial_args.pop("gamma")),
            "kappa": float(fiducial_args.pop("kappa")),
            "z_peak": float(fiducial_args.pop("z_peak")),
        }
    except KeyError as exc:
        raise ValueError(
            f"fiducial redshift sampler is missing argument {exc.args[0]!r}"
        ) from None
    unknown = sorted(fiducial_args)
    if unknown:
        raise ValueError(
            "unsupported fiducial redshift sampler argument(s): " + ", ".join(unknown)
        )

    # Evaluated on the sampler's own lookup grid (``n_grid``, defaulting to
    # gwmock-pop's), not the coarser grid the model integrates on: this is the
    # density the catalog was actually drawn from, so it must mirror the
    # generator rather than the target. Only the formula is shared with the
    # model, via ``compute_merger_rate_distance_and_logprob``. The
    # ``local_merger_rate`` placeholder is arbitrary: it only feeds the
    # discarded ``total_merger_rate`` output -- the normalized density is
    # amplitude-independent, so the rate is normalized away here.
    redshift_grid = jnp.linspace(z_min, z_max, n_grid)
    _, _, logpdf = compute_merger_rate_distance_and_logprob(
        {**params, "local_merger_rate": 1.0},
        {"redshift": jnp.asarray(redshift)},
        redshift_grid=redshift_grid,
    )
    fiducial_logpdf = np.asarray(logpdf)
    in_support = (redshift >= z_min) & (redshift <= z_max)
    uniform_logpdf = np.where(in_support, -np.log(z_max - z_min), -np.inf)
    return np.logaddexp(
        np.log1p(-epsilon) + fiducial_logpdf,
        np.log(epsilon) + uniform_logpdf,
    )


def _parameters(config: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = config.get("parameters", config)
    if not isinstance(parameters, Mapping):
        raise TypeError("gwmock-pop graph config must define a parameters mapping")
    return parameters


def _sampler(parameters: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    parameter = parameters.get(name)
    if not isinstance(parameter, Mapping) or not isinstance(
        parameter.get("sampler"), Mapping
    ):
        raise TypeError(f"graph parameter {name!r} must define a sampler")
    return parameter["sampler"]


def _validate_component_graphs(
    fiducial: Mapping[str, Any], uniform: Mapping[str, Any]
) -> None:
    if set(fiducial) != set(uniform):
        raise ValueError("proposal component graphs must define identical parameters")
    mismatched = [
        name
        for name in fiducial
        if name != "redshift" and fiducial[name] != uniform[name]
    ]
    if mismatched:
        raise ValueError(
            "proposal component graphs may differ only in redshift; mismatched: "
            + ", ".join(mismatched)
        )


def _population_size(population: Mapping[str, np.ndarray]) -> int:
    lengths = {np.asarray(values).shape for values in population.values()}
    if len(lengths) != 1:
        raise ValueError("source population columns must have identical shapes")
    shape = next(iter(lengths))
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError(
            "source population columns must be non-empty and one-dimensional"
        )
    return int(shape[0])


def _normalize_weights(weights: Sequence[float] | None, num_sources: int) -> np.ndarray:
    if weights is None or len(weights) != num_sources:
        raise ValueError("mixture requires one weight per source population")
    probabilities = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("mixture weights must be finite and non-negative")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("mixture weights must sum to a positive value")
    return probabilities / total


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    source_paths = [path.resolve() for path in args.source]
    populations = [read_population_catalogue(path) for path in source_paths]
    production, assignments = assemble_population(
        populations,
        operation=args.operation,
        num_samples=args.num_samples,
        seed=args.seed,
        weights=args.weight,
    )
    if args.uniform_redshift_fraction is not None:
        if not args.source_config or len(args.source_config) != 2:
            raise ValueError(
                "guarded proposal density requires exactly two --source-config values"
            )
        configs = [load_mapping(path.resolve()) for path in args.source_config]
        production[PROPOSAL_COMPONENT] = assignments
        production[PROPOSAL_REDSHIFT_LOGPDF] = guarded_proposal_logpdf(
            np.asarray(production["redshift"]),
            fiducial_config=configs[0],
            uniform_config=configs[1],
            epsilon=args.uniform_redshift_fraction,
        )
    elif args.source_config:
        raise ValueError("--source-config requires --uniform-redshift-fraction")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_population_catalogue(output_path, production)
    counts = np.bincount(assignments, minlength=len(populations))
    logger.info(
        "Saved %d production samples to %s (component counts: %s)",
        args.num_samples,
        output_path,
        counts.tolist(),
    )


if __name__ == "__main__":
    main()
