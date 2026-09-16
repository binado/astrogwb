"""Simulate many spectrum-only SGWB draws in one process.

A polarization-power catalog persists ``(F, N)`` power. This writes only the
forward-model spectrum ``(draws, F)``: a static ``max_events`` plate, a
free Poisson count per draw, and no catalog materialization.

``gwb_forward_model`` already samples ``n_events ~ Poisson(R T)`` and masks a
static plate. This script draws that model with
:class:`~numpyro.infer.Predictive` -- not a host-side Poisson draw fed back as
``observed_num_events``. ``--n-max-sigma`` sizes ``max_events`` from the rate
tail; a Poisson draw above that capacity is silently capped, as the model
documents.

Usage::

    uv run --extra io python scripts/simulate_spectra.py \\
        --approximant TaylorF2 \\
        --sampling-frequency 128 --minimum-frequency 20 \\
        --maximum-frequency 48 --reference-frequency 20 \\
        --frequency-resolution 4 \\
        --population bns_md_cosmological \\
        --model-kwargs '{"minimum_redshift": 0.3, "maximum_redshift": 20, "n_grid": 64}' \\
        --params '{"H0": 67.66, "Omega_m": 0.3096, "gamma": 1.42, "kappa": 4.62, "z_peak": 1.84, "local_merger_rate": 770.0, "minimum_mass": 1.0, "mass_width": 1.5}' \\
        --observation-time 1 --draws 8 --seed 0 \\
        --output outputs/spectra.h5
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive

from astrogwb.catalog import SpectralDensityCatalog
from astrogwb.metadata import CatalogMetadata, PopulationMetadata, WaveformMetadata
from astrogwb.populations import build_population
from astrogwb.sampling import gwb_forward_model, validate_source_model
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import RippleGenerator

# x64 must be on before any arrays. Ripple's import turns it on as a side
# effect, but the rate evaluation and Predictive draws must not depend on that.
jax.config.update("jax_enable_x64", True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate many forward-model spectral-density draws and save "
            "(draws, F) without materializing catalog power."
        )
    )
    parser.add_argument("--approximant", type=str, required=True)
    parser.add_argument("--sampling-frequency", type=float, required=True)
    parser.add_argument("--minimum-frequency", type=float, required=True)
    parser.add_argument("--maximum-frequency", type=float, required=True)
    parser.add_argument("--reference-frequency", type=float, required=True)
    parser.add_argument("--frequency-resolution", type=float, required=True)

    parser.add_argument("--population", type=str, required=True)
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        metavar="JSON",
        help='JSON object of model construction kwargs, e.g. \'{"minimum_redshift": 0.3, "maximum_redshift": 20}\'',
    )
    parser.add_argument(
        "--params",
        type=json.loads,
        default={},
        metavar="JSON",
        help=(
            "JSON object of population hyperparameters, e.g. "
            '\'{"H0": 67.66, "local_merger_rate": 770}\''
        ),
    )

    parser.add_argument("--observation-time", type=float, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-max-sigma", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output spectra file.",
    )
    return parser.parse_args(argv)


def padded_event_capacity(mean_count: float, n_max_sigma: float) -> int:
    """Static plate size: the Poisson mean plus ``n_max_sigma`` standard deviations."""
    if mean_count < 0.0:
        raise ValueError("Poisson mean must be non-negative")
    if n_max_sigma < 0.0:
        raise ValueError("--n-max-sigma must be non-negative")
    return max(int(np.ceil(mean_count + n_max_sigma * np.sqrt(mean_count))), 1)


def _require_mapping(value: Any, *, flag: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{flag} must be a JSON object, got {type(value).__name__}")
    return value


def _float_params(params: Mapping[str, Any]) -> dict[str, float]:
    converted: dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, str):
            raise TypeError(f"parameter {key!r} must be numeric, got {value!r}")
        converted[key] = float(value)
    return converted


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.draws <= 0:
        raise ValueError("--draws must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.observation_time <= 0.0:
        raise ValueError("--observation-time must be positive")

    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing spectra: {output}. "
            "Pass --force only for an intentional replacement."
        )

    model_kwargs = _require_mapping(args.model_kwargs, flag="--model-kwargs")
    params = _float_params(_require_mapping(args.params, flag="--params"))
    source_model, merger_rate_fn = build_population(args.population, **model_kwargs)
    if merger_rate_fn is None:
        raise ValueError(
            f"population {args.population!r} declares no merger rate, so there "
            "is no Poisson mean to draw an event count from; it is a proposal "
            "density, not a population to simulate observations from"
        )

    generator = RippleGenerator(
        WaveformMetadata(
            approximant=args.approximant,
            sampling_frequency=args.sampling_frequency,
            minimum_frequency=args.minimum_frequency,
            maximum_frequency=args.maximum_frequency,
            reference_frequency=args.reference_frequency,
            frequency_resolution=args.frequency_resolution,
        )
    )

    validate_source_model(
        params,
        source_model=source_model,
        generator=generator,
        rng_key=jax.random.key(args.seed),
    )

    rate = float(jnp.asarray(merger_rate_fn(params)))
    mean_count = rate * years_to_seconds(args.observation_time)
    max_events = padded_event_capacity(mean_count, args.n_max_sigma)

    simulate = Predictive(
        partial(
            gwb_forward_model,
            source_model=source_model,
            merger_rate_fn=merger_rate_fn,
            generator=generator,
            observation_time=args.observation_time,
            batch_size=args.batch_size,
            max_events=max_events,
        ),
        num_samples=args.draws,
        return_sites=("spectral_density", "n_events", "total_merger_rate"),
    )
    draws = simulate(jax.random.key(args.seed), params)
    frequencies = np.asarray(generator.frequencies)
    spectral_density = np.asarray(draws["spectral_density"])
    n_events = np.asarray(draws["n_events"], dtype=np.int64)
    merger_rates = np.asarray(draws["total_merger_rate"], dtype=np.float64)

    draw_hyperparameters = {
        name: np.full(args.draws, value, dtype=np.float64)
        for name, value in params.items()
    }

    catalog = SpectralDensityCatalog(
        spectral_density=spectral_density,
        frequencies=frequencies,
        n_events=n_events,
        total_merger_rate=merger_rates,
        hyperparameters=draw_hyperparameters,
        _metadata=CatalogMetadata(
            waveform=generator.metadata,
            population=PopulationMetadata(
                model_name=args.population,
                model_kwargs=model_kwargs,
                seed=args.seed,
            ),
        ),
        n_max_sigma=args.n_max_sigma,
        observation_time=args.observation_time,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    catalog.save(output)


if __name__ == "__main__":
    main()
