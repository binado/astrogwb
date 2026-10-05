r"""Draw populations -- hyperparameters, a source count, the sources -- per seed.

One seed is one draw: its key splits into a hyperparameter key, a count key and
a source key, always in that order, so a draw depends on its own seed alone and
the same seed replays the same events whatever waveform consumes them.

The three stages differ in cost and in what must be static:

1. *Hyperparameters and counts* are cheap and vectorized over the draws in one
   jitted call. The counts come back to the host as NumPy, because the host uses
   them to decide shapes.
2. *Sources* are drawn per draw at a **bucketed** size, then trimmed to the
   draw's count. Event ``i`` of a draw is drawn from ``fold_in(source_key, i)``,
   so the first ``n`` events do not depend on the size drawn: the bucket ladder
   and ``chunk_size`` change cost, never the realization.
3. The result is *flat*: ``source_parameters`` columns of length
   ``sum(counts)``, with ``counts`` recovering each draw's slice. That is the
   layout :func:`~astrogwb.simulators.spectra.forward.packed_power_sum` reduces
   without padding each draw to the largest.

Source sampling does not use ``numpyro.plate`` at a rate-derived size, so the
population draw is independent of the merger rate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from numpyro import handlers
from numpyro.distributions import Distribution

from astrogwb.populations import MergerRateFn, SourceFn
from astrogwb.populations.evaluation import evaluate_sources
from astrogwb.simulators._keys import seed_key
from astrogwb.utils import years_to_seconds

__all__ = [
    "BUCKET_RATIO",
    "PopulationDraws",
    "PopulationSampler",
    "bucket_size",
    "draw_keys",
    "sample_sources_by_key",
]

#: Geometric growth of the source-draw sizes. Sizes only affect cost -- every
#: distinct one is a compile and padding is drawn and discarded -- so the ratio
#: trades a few more compiles for less discarded sampling. It is not metadata.
BUCKET_RATIO = 1.25


@dataclass(frozen=True, slots=True)
class PopulationDraws:
    """Population draws as NumPy arrays; draws first, sources flat.

    ``hyperparameters`` columns, ``total_merger_rate`` and ``counts`` have shape
    ``(draws,)``. Every ``source_parameters`` column has shape
    ``(counts.sum(),)``; draw ``b`` owns the slice
    ``offsets[b]:offsets[b + 1]`` of ``offsets = concatenate([[0], cumsum(counts)])``.
    """

    hyperparameters: dict[str, NDArray[np.float64]]
    total_merger_rate: NDArray[np.float64]
    counts: NDArray[np.int64]
    source_parameters: dict[str, NDArray[Any]]

    @classmethod
    def from_arrays(cls, outputs: Mapping[str, Any]) -> PopulationDraws:
        """Wrap the arrays :func:`~astrogwb.simulators.population.population` returns."""
        return cls(
            hyperparameters=dict(outputs["hyperparameters"]),
            total_merger_rate=np.asarray(outputs["total_merger_rate"]),
            counts=np.asarray(outputs["counts"], dtype=np.int64),
            source_parameters=dict(outputs["source_parameters"]),
        )

    @property
    def offsets(self) -> NDArray[np.int64]:
        """Start of each draw's slice, plus the total, shape ``(draws + 1,)``."""
        return np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)

    @property
    def segment_ids(self) -> NDArray[np.int32]:
        """The draw index of every source, shape ``(counts.sum(),)``."""
        return np.repeat(np.arange(self.counts.size, dtype=np.int32), self.counts)


def bucket_size(count: int, chunk_size: int, ratio: float = BUCKET_RATIO) -> int:
    """Smallest size of a geometric ladder of ``chunk_size`` multiples holding ``count``.

    At least one chunk, so an empty draw still has a shape to compile.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")
    needed, chunks = max(1, math.ceil(count / chunk_size)), 1
    while chunks < needed:
        chunks = max(chunks + 1, math.ceil(chunks * ratio))
    return chunks * chunk_size


def draw_keys(seeds: np.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-draw ``(hyperparameter, count, source)`` keys for ``uint64`` ``seeds``."""
    keys = jnp.stack([seed_key(seed) for seed in seeds])
    split = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
    return split[:, 0], split[:, 1], split[:, 2]


def sample_sources_by_key(
    source_model: SourceFn,
    params: Mapping[str, Any],
    key: jax.Array,
    size: int,
) -> dict[str, jax.Array]:
    """``size`` sources, event ``i`` drawn from ``fold_in(key, i)``.

    The counterpart of :func:`~astrogwb.populations.evaluation.sample_sources`
    with the same replay -- sampled sites are drawn, then every returned column,
    deterministics included, is recomputed through
    :func:`~astrogwb.populations.evaluation.evaluate_sources` -- but with one key
    per event instead of ``split(key, size)``. Splitting a key into ``n`` and
    into ``m > n`` children does not share a prefix, so there the drawn events
    would depend on ``size``; here the first ``n`` events of ``size = m`` are the
    ``size = n`` draw, which is what lets sizes be bucketed freely.
    """
    event_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(
        jnp.arange(size, dtype=jnp.uint32)
    )

    def one_event(event_key: jax.Array) -> dict[str, jax.Array]:
        with handlers.block():
            trace = handlers.trace(handlers.seed(source_model, event_key)).get_trace(
                params
            )
        return {
            name: site["value"]
            for name, site in trace.items()
            if site["type"] == "sample"
        }

    sampled = jax.vmap(one_event)(event_keys)
    _, outputs = evaluate_sources(source_model, params, sampled, density_sites=())
    return outputs


class PopulationSampler:
    """Draws populations at fixed hyperparameter declarations; build once, call often.

    Holds the jitted stages, so reusing one sampler keeps their compilations
    warm: both callables of a built population hash by identity, and a fresh
    equal rebuild would recompile. Needs ``jax_enable_x64`` before the first
    call (``seed_key`` raises otherwise).

    ``count="poisson"`` draws ``Poisson(R(theta) * observation_time)`` per draw;
    ``count="fixed"`` draws ``num_events`` for every draw. A population that
    declares no merger rate cannot be drawn: even a fixed-count consumer
    normalizes by it.
    """

    def __init__(
        self,
        *,
        source_model: SourceFn,
        merger_rate_fn: MergerRateFn | None,
        fixed: Mapping[str, float],
        priors: Mapping[str, Distribution],
        observation_time: float,
        count: Literal["poisson", "fixed"],
        num_events: int | None = None,
    ) -> None:
        if merger_rate_fn is None:
            raise ValueError(
                "the population declares no physical merger rate; drawing it "
                "requires an observer-frame rate in both count modes"
            )
        if observation_time <= 0.0:
            raise ValueError("observation_time must be positive")
        if count == "fixed" and (num_events is None or num_events <= 0):
            raise ValueError("num_events must be a positive integer for fixed counts")
        if count == "poisson" and num_events is not None:
            raise ValueError("num_events is only valid for fixed counts")
        overlap = set(fixed) & set(priors)
        if overlap:
            raise ValueError(
                f"hyperparameters both fixed and sampled: {sorted(overlap)}"
            )

        self._fixed = dict(fixed)
        self._priors = dict(sorted(priors.items()))
        self._count = count
        self._num_events = num_events
        self._observation_seconds = years_to_seconds(observation_time)

        def hyperparameters_and_counts(
            theta_keys: jax.Array, count_keys: jax.Array
        ) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
            def theta_of(key: jax.Array) -> dict[str, jax.Array]:
                subkeys = jax.random.split(key, max(len(self._priors), 1))
                return {
                    name: jnp.reshape(prior.sample(subkeys[i]), ())
                    for i, (name, prior) in enumerate(self._priors.items())
                }

            draws = theta_keys.shape[0]
            theta = {
                **{name: jnp.full(draws, value) for name, value in self._fixed.items()},
                **jax.vmap(theta_of)(theta_keys),
            }
            rate = jax.vmap(lambda t: jnp.reshape(merger_rate_fn(t), ()))(theta)
            if num_events is not None:
                counts = jnp.full(draws, num_events, dtype=jnp.int64)
            else:
                counts = jax.vmap(
                    lambda key, mean: jax.random.poisson(key, mean).astype(jnp.int64)
                )(count_keys, rate * self._observation_seconds)
            return theta, rate, counts

        self._stage_one = jax.jit(hyperparameters_and_counts)
        self._draw_sources: Callable[..., dict[str, jax.Array]] = jax.jit(
            lambda params, key, size: sample_sources_by_key(
                source_model, params, key, size
            ),
            static_argnames="size",
        )

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds, as Poisson counts use it."""
        return self._observation_seconds

    def __call__(self, seeds: np.ndarray, *, chunk_size: int) -> PopulationDraws:
        """Draw one population per seed; ``chunk_size`` only sets the size ladder."""
        theta_keys, count_keys, source_keys = draw_keys(seeds)
        theta, rate, counts_device = self._stage_one(theta_keys, count_keys)
        hyperparameters = {n: np.asarray(v, dtype=np.float64) for n, v in theta.items()}
        counts = np.asarray(counts_device, dtype=np.int64)

        columns: dict[str, list[np.ndarray]] = {}
        for draw, count in enumerate(counts):
            params = {name: values[draw] for name, values in hyperparameters.items()}
            sources = self._draw_sources(
                params, source_keys[draw], size=bucket_size(int(count), chunk_size)
            )
            for name, values in sources.items():
                columns.setdefault(name, []).append(np.asarray(values)[: int(count)])
        return PopulationDraws(
            hyperparameters=hyperparameters,
            total_merger_rate=np.asarray(rate, dtype=np.float64),
            counts=counts,
            source_parameters={
                name: np.concatenate(parts) for name, parts in columns.items()
            },
        )
