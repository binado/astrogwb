"""Simulate many spectrum-only SGWB draws in one process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive

from astrogwb.populations import (
    DEFAULT_DENSITY_SITES,
    build_merger_rate_fn,
    build_source_model,
)
from astrogwb.sampling import gwb_forward_model, validate_source_model
from astrogwb.sampling._io import SpectraArtifact, save_spectra
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import RippleGenerator


def _parse_value(raw: str) -> float | int | str:
    value = raw.strip()
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_key_value(raw: str) -> tuple[str, float | int | str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"expected KEY=VALUE, got {raw!r}"
        )
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("key in KEY=VALUE cannot be empty")
    return key, _parse_value(value)


def _as_mapping(pairs: Sequence[tuple[str, float | int | str]]) -> dict[str, float | int | str]:
    mapping: dict[str, float | int | str] = {}
    for key, value in pairs:
        mapping[key] = value
    return mapping


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate many padded forward-model spectral-density draws and save (draws, F)."
    )
    parser.add_argument("--approximant", type=str, required=True)
    parser.add_argument("--sampling-frequency", type=float, required=True)
    parser.add_argument("--minimum-frequency", type=float, required=True)
    parser.add_argument("--maximum-frequency", type=float, required=True)
    parser.add_argument("--reference-frequency", type=float, required=True)
    parser.add_argument("--frequency-resolution", type=float, required=True)

    parser.add_argument("--source-model", type=str, required=True)
    parser.add_argument("--rate-model", type=str, required=True)
    parser.add_argument(
        "--model-kwarg",
        action="append",
        default=[],
        type=_parse_key_value,
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=_parse_key_value,
        metavar="KEY=VALUE",
    )

    parser.add_argument("--observation-time", type=float, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-max-sigma", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _float_params(params: Mapping[str, float | int | str]) -> dict[str, float]:
    converted: dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, str):
            raise ValueError(f"parameter {key!r} must be numeric, got {value!r}")
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

    model_kwargs = _as_mapping(args.model_kwarg)
    params = _float_params(_as_mapping(args.param))

    source_model = build_source_model(args.source_model, settings=model_kwargs)
    merger_rate_fn = build_merger_rate_fn(args.rate_model, settings=model_kwargs)

    generator = RippleGenerator(
        approximant=args.approximant,
        sampling_frequency=args.sampling_frequency,
        minimum_frequency=args.minimum_frequency,
        maximum_frequency=args.maximum_frequency,
        reference_frequency=args.reference_frequency,
        frequency_resolution=args.frequency_resolution,
        chunk_size=args.batch_size,
    )

    validate_source_model(
        params,
        source_model=source_model,
        generator=generator,
        rng_key=jax.random.PRNGKey(args.seed),
    )

    rate = float(jnp.asarray(merger_rate_fn(params)))
    observation_seconds = years_to_seconds(args.observation_time)
    mu = rate * observation_seconds
    rng = np.random.default_rng(args.seed)
    n_events = rng.poisson(mu, size=args.draws)

    padded_ceiling = int(np.ceil(mu + args.n_max_sigma * np.sqrt(max(mu, 0.0))))
    n_max = max(int(n_events.max(initial=0)), padded_ceiling, 1)

    simulate = Predictive(
        gwb_forward_model,
        num_samples=1,
        return_sites=("spectral_density", "total_merger_rate"),
    )

    def one_draw(key: jax.Array, n_active: int) -> tuple[jax.Array, jax.Array]:
        draws = simulate(
            key,
            source_model=source_model,
            merger_rate_fn=merger_rate_fn,
            generator=generator,
            observation_time=args.observation_time,
            batch_size=args.batch_size,
            num_events=n_max,
            params=params,
            n_active=n_active,
        )
        return draws["spectral_density"][0], draws["total_merger_rate"][0]

    model = jax.jit(one_draw)

    first_spectrum, first_rate = model(
        jax.random.fold_in(jax.random.PRNGKey(args.seed), 0), int(n_events[0])
    )
    first_spectrum = np.asarray(first_spectrum)
    frequencies = np.asarray(generator.frequencies)
    spectral_density = np.empty((args.draws, first_spectrum.shape[0]), dtype=np.float64)
    spectral_density[0] = first_spectrum
    merger_rates = np.empty(args.draws, dtype=np.float64)
    merger_rates[0] = float(np.asarray(first_rate))

    for draw in range(1, args.draws):
        spectrum, draw_rate = model(
            jax.random.fold_in(jax.random.PRNGKey(args.seed), draw),
            int(n_events[draw]),
        )
        spectral_density[draw] = np.asarray(spectrum)
        merger_rates[draw] = float(np.asarray(draw_rate))

    draw_hyperparameters = {
        name: np.full(args.draws, value, dtype=np.float64) for name, value in params.items()
    }

    artifact = SpectraArtifact(
        frequencies=frequencies,
        spectral_density=spectral_density,
        n_events=np.asarray(n_events, dtype=np.int64),
        total_merger_rate=merger_rates,
        hyperparameters=draw_hyperparameters,
        source_model_name=args.source_model,
        rate_model_name=args.rate_model,
        model_kwargs=model_kwargs,
        density_sites=DEFAULT_DENSITY_SITES,
        waveform_metadata=generator,
        n_max_sigma=args.n_max_sigma,
        seed=args.seed,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    save_spectra(artifact, output)


if __name__ == "__main__":
    main()
