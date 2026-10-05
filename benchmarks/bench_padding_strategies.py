# /// script
# requires-python = ">=3.12"
# dependencies = ["jax>=0.4.35", "numpy"]
# ///
"""How to reduce per-draw event catalogs to spectra for SBI: padding strategies.

Question: with hyperparameters drawn per simulation, expected counts vary
across draws. Which reduction strategy gives the best throughput?

  global-N   one static N for every draw (capacity at the prior's largest mean)
             -> one compile, but most waveform evaluations are masked padding
  bucketed   each draw padded to a geometric bucket of its own count
             -> evaluations ~ sum(counts) * ratio, one compile per bucket
  packed     events of all draws concatenated into one stream (segment ids),
             reduced with a traced-trip-count loop over fixed chunks
             -> evaluations ~ sum(counts), one compile per (coarse) buffer cap

Standalone on purpose: the waveform is a surrogate with the same cost shape as
the real one (an ``(events, F)`` array of transcendental-heavy arithmetic), so
the comparison is about padding and compilation, not about Ripple. Stage 1
(hyperparameters and counts) is shared and excluded: it is O(B), negligible
next to waveforms. Sources use one key per event (``fold_in``), so a draw's
first n events do not depend on the padded size and every candidate must
return identical spectra, which is asserted before any timing.

Run:  uv run --script benchmarks/bench_padding_strategies.py
Defaults to CPU and float64 (override with JAX_PLATFORMS / JAX_ENABLE_X64).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

F = 256
CHUNK = 256
BATCH = 32
RATIO = 1.25  # geometric bucket ratio
FREQS = jnp.linspace(20.0, 1000.0, F)


# --- surrogate population and waveform ---------------------------------------


def draw_source(key: jax.Array) -> jax.Array:
    k1, k2, k3 = jax.random.split(key, 3)
    a = 1.0 + 1.5 * jax.random.uniform(k1)
    b = 1.0 + 1.5 * jax.random.uniform(k2)
    distance = 100.0 + 3000.0 * jax.random.uniform(k3) ** (1.0 / 3.0)
    return jnp.stack([jnp.maximum(a, b), jnp.minimum(a, b), distance])


def draw_sources(key: jax.Array, size: int) -> jax.Array:
    """``(size, 3)``; event ``i`` depends on ``(key, i)`` only."""
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(size))
    return jax.vmap(draw_source)(keys)


def power(src: jax.Array) -> jax.Array:
    """``(chunk, F)`` surrogate polarization power."""
    m1, m2, d = src[:, 0:1], src[:, 1:2], src[:, 2:3]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    f = FREQS[None, :]
    amp = mc ** (5.0 / 3.0) * f ** (-7.0 / 3.0) / d**2
    phase = jnp.sin(2.0 * jnp.pi * f * 1e-3 * m1) + jnp.cos(f * 1e-3 * m2)
    return amp * (1.0 + 0.25 * phase) ** 2 * (f < 4400.0 / (m1 + m2))


# --- candidates: (keys (B,), counts (B,)) -> (B, F) ---------------------------


@partial(jax.jit, static_argnames=("size",))
def _masked_draw(key: jax.Array, count: jax.Array, *, size: int) -> jax.Array:
    """Draw ``size`` events, sum power over the first ``count``, chunk by chunk."""
    src = draw_sources(key, size).reshape(size // CHUNK, CHUNK, 3)
    idx = jnp.arange(size).reshape(size // CHUNK, CHUNK)

    def body(carry: jax.Array, xs: tuple[jax.Array, jax.Array]):
        chunk, i = xs
        return carry + (power(chunk) * (i < count)[:, None]).sum(0), None

    total, _ = jax.lax.scan(body, jnp.zeros(F), (src, idx))
    return total


def bucket(n: int, ratio: float) -> int:
    """Smallest element >= n/CHUNK of a geometric chunk-count ladder, in events."""
    need, b = max(1, math.ceil(n / CHUNK)), 1
    while b < need:
        b = max(b + 1, math.ceil(b * ratio))
    return b * CHUNK


def make_global(n_cap: int) -> Callable:
    size = bucket(n_cap, 1.0)

    def run(keys, counts):
        return np.stack(
            [np.asarray(_masked_draw(k, c, size=size)) for k, c in zip(keys, counts)]
        )

    run.evals = lambda counts: len(counts) * size  # type: ignore[attr-defined]
    return run


def run_bucketed(keys, counts):
    return np.stack(
        [
            np.asarray(_masked_draw(k, c, size=bucket(int(c), RATIO)))
            for k, c in zip(keys, counts)
        ]
    )


run_bucketed.evals = lambda counts: sum(bucket(int(c), RATIO) for c in counts)  # type: ignore[attr-defined]


@partial(jax.jit, static_argnames=("size",))
def _draw_only(key: jax.Array, *, size: int) -> jax.Array:
    return draw_sources(key, size)


@partial(jax.jit, static_argnames=("num_segments",))
def _reduce_packed(src, seg, n_chunks, *, num_segments: int):
    def body(i, acc):
        c = jax.lax.dynamic_slice_in_dim(src, i * CHUNK, CHUNK)
        s = jax.lax.dynamic_slice_in_dim(seg, i * CHUNK, CHUNK)
        return acc + jax.ops.segment_sum(power(c), s, num_segments=num_segments + 1)

    acc = jax.lax.fori_loop(0, n_chunks, body, jnp.zeros((num_segments + 1, F)))
    return acc[:num_segments]


def run_packed(keys, counts):
    counts = np.asarray(counts, dtype=np.int64)
    rows = [
        np.asarray(_draw_only(k, size=bucket(int(c), RATIO)))[: int(c)]
        for k, c in zip(keys, counts)
    ]
    total = int(counts.sum())
    cap = bucket(total, 1.5)  # coarse: a few static buffer shapes, not one per batch
    src = np.tile(np.array([[1.0, 1.0, 1.0]]), (cap, 1))  # benign padding rows
    seg = np.full(cap, len(counts), dtype=np.int32)  # padding -> dropped segment
    src[:total] = np.concatenate(rows)
    seg[:total] = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    out = _reduce_packed(
        jnp.asarray(src),
        jnp.asarray(seg),
        jnp.asarray(math.ceil(total / CHUNK)),
        num_segments=len(counts),
    )
    return np.asarray(out)


run_packed.evals = lambda counts: CHUNK * math.ceil(int(np.sum(counts)) / CHUNK)  # type: ignore[attr-defined]


# --- scenarios ----------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    name: str
    rate: tuple[float, float]  # local_merger_rate bounds
    h0: tuple[float, float]  # H0 bounds

    def mean_count(self, mu0: float, rate, h0):
        return mu0 * rate / 770.0 * (67.66 / h0) ** 3

    def max_count(self, mu0: float) -> float:
        return float(self.mean_count(mu0, self.rate[1], self.h0[0]))


PRIORS = [
    Prior("narrow", (770 * 0.97, 770 * 1.03), (67.66, 67.66)),
    Prior("wide", (300.0, 1200.0), (60.0, 80.0)),
]


def make_batch(prior: Prior, mu0: float, seed: int):
    rng = np.random.default_rng(seed)
    rate = rng.uniform(*prior.rate, BATCH)
    h0 = (
        rng.uniform(*prior.h0, BATCH)
        if prior.h0[0] != prior.h0[1]
        else np.full(BATCH, prior.h0[0])
    )
    counts = rng.poisson(prior.mean_count(mu0, rate, h0))
    keys = [jax.random.key(int(s)) for s in rng.integers(0, 2**31, BATCH)]
    return keys, counts


def main() -> None:
    print(
        f"backend={jax.default_backend()} x64={jax.config.jax_enable_x64} "
        f"F={F} chunk={CHUNK} batch={BATCH} bucket-ratio={RATIO}\n"
    )
    for prior in PRIORS:
        for mu0 in (1_000.0, 10_000.0):
            n_cap = math.ceil(
                prior.max_count(mu0) + 5.0 * math.sqrt(prior.max_count(mu0))
            )
            candidates: dict[str, Callable] = {
                "global-N": make_global(n_cap),
                "bucketed": run_bucketed,
                "packed": run_packed,
            }
            warm = [make_batch(prior, mu0, s) for s in (0, 1)]
            timed = [make_batch(prior, mu0, s) for s in (2, 3, 4, 5)]
            c_all = np.concatenate([c for _, c in timed])
            print(
                f"prior={prior.name} mu0={mu0:,.0f}  counts "
                f"min/med/max = {c_all.min()}/{int(np.median(c_all))}/{c_all.max()}  "
                f"global-N={bucket(n_cap, 1.0)}"
            )

            # Correctness before speed.
            ref = None
            for name, fn in candidates.items():
                out = fn(*timed[0])
                ref = out if ref is None else ref
                if not np.allclose(out, ref, rtol=1e-9):
                    print(f"  !! {name} disagrees with global-N")

            print(
                f"  {'candidate':<10}{'cold':>9}{'per batch':>12}"
                f"{'ms/sim':>9}{'evals/sum(counts)':>19}{'vs global':>11}"
            )
            base = None
            for name, fn in candidates.items():
                jax.clear_caches()  # cold = compile + first batch
                t0 = time.perf_counter()
                fn(*warm[0])
                cold = time.perf_counter() - t0
                fn(*warm[1])
                times, evals = [], []
                for keys, counts in timed:
                    t0 = time.perf_counter()
                    out = fn(keys, counts)
                    jax.block_until_ready(out)
                    times.append(time.perf_counter() - t0)
                    evals.append(fn.evals(counts) / counts.sum())
                mean = float(np.mean(times))
                base = base or mean
                print(
                    f"  {name:<10}{cold:>8.2f}s{mean:>11.3f}s"
                    f"{mean / BATCH * 1e3:>9.2f}{np.mean(evals):>19.2f}"
                    f"{base / mean:>10.2f}x"
                )
            print()


if __name__ == "__main__":
    main()
