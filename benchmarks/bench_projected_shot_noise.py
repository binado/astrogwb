# /// script
# requires-python = ">=3.12"
# dependencies = ["jax>=0.4.35", "numpy"]
# ///
"""Cost of the exact projected shot-noise variance in the amplitude likelihood.

Compares one log-likelihood evaluation (and its gradient) of the importance
spectrum with a rank-one shot-noise term, vmapped over K networks:

- ``snr_weighted``: the previous amplitude mode. The per-bin variance is a
  network-independent weighted mean of the squared power, and s^2 is its
  SNR-weighted mean over bins, an O(F) step per network.
- ``projected``: the new amplitude mode. Each source is projected on the
  network's noise-weighted template before squaring, a (Z, F_ref) @ (F_ref, N)
  contraction per network.

The kernels are copied from astrogwb.inference.likelihood so the benchmark
survives refactors. The two give different s^2 by design (the projected one is
the exact variance, the SNR-weighted one its upper bound), so the correctness
gate is finiteness and the bound, not allclose.

Run with:  uv run --script benchmarks/bench_projected_shot_noise.py
Defaults to CPU and float64.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

INCLINATION_SECOND_MOMENT = 355.0 / 252.0
NUM_REFERENCE = 765  # the default loglinear waveform grid
NUM_OBSERVED = 765
# (N draws, Z redshift nodes, K networks)
REGIMES = [
    (4_096, 8, 1),
    (32_768, 8, 1),
    (32_768, 32, 1),
    (32_768, 32, 5),
]


def make_problem(num_draws: int, num_nodes: int, num_networks: int, seed: int = 0):
    """A catalog with inspiral-like spectra and per-source high-frequency cutoffs."""
    rng = np.random.default_rng(seed)
    log_ref = np.log(np.geomspace(2.0, 2048.0, NUM_REFERENCE))
    cutoff = rng.uniform(np.log(200.0), np.log(1600.0), num_draws)
    normalization = rng.lognormal(0.0, 0.5, num_draws)
    power = (
        normalization
        * np.exp(-7.0 / 3.0 * log_ref[:, None])
        * (log_ref[:, None] < cutoff)
    )
    log_observed = np.log(np.geomspace(2.0, 1000.0, NUM_OBSERVED))
    redshift = np.linspace(0.35, 8.0, num_nodes)
    query = log_observed[:, None] + np.log1p(redshift)[None, :]
    data = {
        "power": jnp.asarray(power),
        "squared_power": jnp.asarray(power**2),
        "log_ref": jnp.asarray(log_ref),
        "query": jnp.asarray(query),
        "amplitude": jnp.asarray((1.0 + redshift) ** (-1.0 / 3.0)),
        "kernel_shape": jnp.asarray(np.exp(-redshift / 2.0)),
        "direction": jnp.asarray(rng.normal(size=num_draws)),
    }
    scale = rng.uniform(0.5, 2.0, (num_networks, NUM_OBSERVED))
    return data, jnp.asarray(scale)


def node_power(power, log_weights, log_ref, query, amplitude):
    mean = power @ jnp.exp(log_weights) / log_weights.shape[0]
    return amplitude * jnp.interp(query, log_ref, mean, right=0.0)


def predict(data, theta):
    """Spectrum and per-bin variance; theta tilts the weights and the kernel."""
    log_weights = theta * 1e-2 * data["direction"]
    kernel = theta * data["kernel_shape"]
    mean = (
        node_power(
            data["power"],
            log_weights,
            data["log_ref"],
            data["query"],
            data["amplitude"],
        )
        @ kernel
    )
    variance = (
        INCLINATION_SECOND_MOMENT
        * node_power(
            data["squared_power"],
            log_weights,
            data["log_ref"],
            data["query"],
            data["amplitude"] ** 2,
        )
        @ kernel
        / 1e4
    )
    return mean, variance, log_weights, kernel


def rank_one_log_likelihood(observed, mean, scale, direction):
    whitened = (observed - mean) / scale
    v = direction / scale
    vv = v @ v
    chi2 = whitened @ whitened - (whitened @ v) ** 2 / (1.0 + vv)
    return -0.5 * (chi2 + jnp.log1p(vv) + 2.0 * jnp.sum(jnp.log(scale)))


def snr_weighted(mean, variance, scale):
    weight = (mean / scale) ** 2
    ratio = jnp.sqrt(variance) / mean
    return (jnp.sum(weight * ratio) / jnp.sum(weight)) ** 2


def projected(data, log_weights, kernel, mean, scale):
    projector = mean / scale**2

    def node_projector(query):
        _, pullback = jax.vjp(
            lambda reference: jnp.interp(query, data["log_ref"], reference, right=0.0),
            jnp.zeros_like(data["log_ref"]),
        )
        return pullback(projector)[0]

    projectors = jax.vmap(node_projector, in_axes=1)(data["query"])
    proj = data["amplitude"][:, None] * (projectors @ data["power"])
    second = (proj**2 @ jnp.exp(log_weights)) / log_weights.shape[0]
    variance = INCLINATION_SECOND_MOMENT * (second @ kernel) / 1e4
    return variance / (projector @ mean) ** 2


def make_candidates(data, observed):
    def old(theta, scales):
        mean, variance, _, _ = predict(data, theta)

        def one(scale):
            s2 = snr_weighted(mean, variance, scale)
            return rank_one_log_likelihood(observed, mean, scale, jnp.sqrt(s2) * mean)

        return jnp.sum(jax.vmap(one)(scales))

    def new(theta, scales):
        mean, _, log_weights, kernel = predict(data, theta)

        def one(scale):
            s2 = projected(data, log_weights, kernel, mean, scale)
            return rank_one_log_likelihood(observed, mean, scale, jnp.sqrt(s2) * mean)

        return jnp.sum(jax.vmap(one)(scales))

    return {"snr_weighted": old, "projected": new}


def timeit(fn, *args, min_batch_s=0.2, repeats=5):
    jax.block_until_ready(fn(*args))
    inner = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(inner):
            out = fn(*args)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        if dt >= min_batch_s or inner >= 1 << 12:
            break
        inner = max(inner * 2, int(inner * min_batch_s / max(dt, 1e-9)))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / inner)
    return min(times), statistics.fmean(times), statistics.pstdev(times)


def compile_stats(fn: Callable, *args) -> tuple[float, str]:
    jax.clear_caches()
    lowered = jax.jit(fn).lower(*args)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    mem = compiled.memory_analysis()
    notes = "" if mem is None else f"temp={mem.temp_size_in_bytes / 1e6:.1f}MB"
    return compile_s, notes


def main() -> None:
    print(f"backend={jax.default_backend()} x64={jax.config.jax_enable_x64}")
    print(f"F_ref={NUM_REFERENCE} F={NUM_OBSERVED}\n")
    theta = jnp.asarray(1.0)
    for num_draws, num_nodes, num_networks in REGIMES:
        data, scales = make_problem(num_draws, num_nodes, num_networks)
        mean, variance, log_weights, kernel = predict(data, theta)
        observed = mean * 1.01
        exact = projected(data, log_weights, kernel, mean, scales[0])
        bound = snr_weighted(mean, variance, scales[0])
        assert bool(jnp.isfinite(exact)) and float(exact) <= float(bound) * (1 + 1e-12)
        candidates = make_candidates(data, observed)

        print(
            f"N={num_draws:,} Z={num_nodes} K={num_networks}  "
            f"(s2 projected/bound = {float(exact) / float(bound):.3f})"
        )
        print(
            f"  {'candidate':<14}{'mean':>11}{'min':>11}{'vs base':>9}"
            f"{'grad mean':>12}{'compile':>10}  notes"
        )
        base = None
        for name, fn in candidates.items():
            compile_s, notes = compile_stats(fn, theta, scales)
            lo, mu, _ = timeit(jax.jit(fn), theta, scales)
            _, gmu, _ = timeit(jax.jit(jax.grad(fn)), theta, scales)
            base = base or mu
            print(
                f"  {name:<14}{mu * 1e3:>9.2f}ms{lo * 1e3:>9.2f}ms{base / mu:>8.2f}x"
                f"{gmu * 1e3:>10.2f}ms{compile_s * 1e3:>8.0f}ms  {notes}"
            )
        print()


if __name__ == "__main__":
    main()
