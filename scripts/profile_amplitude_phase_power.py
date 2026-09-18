"""Compare Ripple polarization power with amplitude-only power.

This benchmarks one ``generate_batch``-sized waveform evaluation, not the
notebook or its output file.  The amplitude-only expression is exact for
non-higher-mode ``AmplitudePhaseWaveform`` models when the inclination
polarization factor is retained::

    |h_+|^2 + |h_x|^2 = A^2 * (((1 + cos(iota)^2) / 2)^2 + cos(iota)^2)

Run from the repository root, for example::

    uv run python scripts/profile_amplitude_phase_power.py --events 128
"""

from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Mapping, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import ripplegw
from ripplegw.interfaces import AmplitudePhaseWaveform

from astrogwb.metadata import WaveformMetadata
from astrogwb.populations import build_population, with_isotropic_inclination
from astrogwb.utils.sampling import sample_sources
from astrogwb.waveform import RippleGenerator
from astrogwb.waveform.generator._ripple import ripple_parameters

jax.config.update("jax_enable_x64", True)

_APPROXIMANTS = (
    "TaylorF2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
)
_PARAMETERS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=int,
        default=128,
        help="sources in the profiled batch (default: 128)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="warm evaluations used for the timing median (default: 5)",
    )
    return parser


def _metadata(approximant: str) -> WaveformMetadata:
    return WaveformMetadata(
        approximant=approximant,
        sampling_frequency=4096.0,
        minimum_frequency=2.0,
        maximum_frequency=2048.0,
        reference_frequency=20.0,
        frequency_resolution=2.0,
    )


def _amplitude_power(
    generator: RippleGenerator,
    waveform: AmplitudePhaseWaveform,
    source_parameters: Mapping[str, jax.Array],
) -> jax.Array:
    """Return amplitude-only polarization power in generator layout ``(F, N)``."""
    events = ripple_parameters(generator.metadata.approximant, source_parameters)
    frequencies = jnp.asarray(generator._frequencies)
    amplitudes = jax.vmap(
        lambda event: waveform.amplitude(frequencies, dict(event)), in_axes=0
    )(events)
    cosine = jnp.cos(events["iota"])
    polarization_factor = ((1.0 + cosine**2) / 2.0) ** 2 + cosine**2
    power = amplitudes**2 * polarization_factor[:, None]
    return jnp.nan_to_num(power[:, generator._band].T)


def _raw_amplitude_power(
    generator: RippleGenerator,
    waveform: AmplitudePhaseWaveform,
    source_parameters: Mapping[str, jax.Array],
) -> jax.Array:
    """Return uncorrected ``amplitude**2`` for the qualification check."""
    events = ripple_parameters(generator.metadata.approximant, source_parameters)
    frequencies = jnp.asarray(generator._frequencies)
    amplitudes = jax.vmap(
        lambda event: waveform.amplitude(frequencies, dict(event)), in_axes=0
    )(events)
    return jnp.nan_to_num(amplitudes[:, generator._band].T) ** 2


def _timed(
    function, source_parameters: Mapping[str, jax.Array], repeats: int
) -> tuple[float, float, np.ndarray]:
    """Compile once, then return first-call time, warm median, and output."""
    start = time.perf_counter()
    output = function(source_parameters)
    output.block_until_ready()
    first = time.perf_counter() - start

    warm_times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = function(source_parameters)
        output.block_until_ready()
        warm_times.append(time.perf_counter() - start)
    return first, float(np.median(warm_times)), np.asarray(output)


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), np.finfo(reference.dtype).tiny)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    return float(np.max(np.abs(reference[finite] - candidate[finite]) / scale[finite]))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.events < 1 or args.repeats < 1:
        raise ValueError("--events and --repeats must be positive")

    source_model, _ = build_population(
        "bns_md_cosmological",
        minimum_redshift=0.3,
        maximum_redshift=20.0,
        n_grid=256,
    )
    source_model = with_isotropic_inclination(source_model)
    source_parameters = sample_sources(
        source_model,
        jax.random.key(20250314),
        _PARAMETERS,
        num_samples=args.events,
    )

    print(f"platform={platform.platform()}")
    print(f"jax={jax.__version__} devices={jax.devices()}")
    print(f"events={args.events} repeats={args.repeats}")
    print(
        "approximant                         max_rel_exact  "
        "max_rel_raw  current_ms  amplitude_ms  speedup"
    )

    for approximant in _APPROXIMANTS:
        generator = RippleGenerator(_metadata(approximant))
        waveform = ripplegw.waveform(approximant, f_ref=20.0)
        if not isinstance(waveform, AmplitudePhaseWaveform):
            raise TypeError(f"{approximant} is not an AmplitudePhaseWaveform")

        current = jax.jit(generator.generate_batch)
        amplitude = jax.jit(partial(_amplitude_power, generator, waveform))
        raw_amplitude = jax.jit(partial(_raw_amplitude_power, generator, waveform))

        current_first, current_warm, current_output = _timed(
            current, source_parameters, args.repeats
        )
        amplitude_first, amplitude_warm, amplitude_output = _timed(
            amplitude, source_parameters, args.repeats
        )
        raw_output = np.asarray(raw_amplitude(source_parameters))
        raw_output = np.nan_to_num(raw_output)

        exact_error = _relative_error(current_output, amplitude_output)
        raw_error = _relative_error(current_output, raw_output)
        speedup = current_warm / amplitude_warm
        print(
            f"{approximant:35s} {exact_error:13.3e} {raw_error:12.3e} "
            f"{current_warm * 1e3:10.2f} {amplitude_warm * 1e3:12.2f} "
            f"{speedup:7.2f}x"
        )
        print(
            f"  first-call compile+run: current={current_first:.3f}s "
            f"amplitude={amplitude_first:.3f}s"
        )


if __name__ == "__main__":
    main()
