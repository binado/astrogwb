"""Draw populations -- hyperparameters, a source count, the sources -- per key.

One key is one draw: it splits into a hyperparameter key, a count key and
a source key, always in that order, so a draw depends on its own key alone and
the same key replays the same events whatever waveform consumes them.

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

:class:`PopulationSimulator` is the waveform-free half of the spectra forward
model. Its batched layout is *flat and ragged*, so a consumer reduces draws
without padding each to the largest; :class:`PopulationData` names it.
Persisting a draw pays when it is reused, such as two waveform approximants on
the same sources.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from numpyro import handlers
from numpyro.distributions import Distribution

from astrogwb import __version__
from astrogwb.populations import MergerRateFn, SourceFn
from astrogwb.populations.evaluation import evaluate_sources
from astrogwb.simulators.population.metadata import PopulationDrawMetadata
from astrogwb.utils import years_to_seconds

__all__ = [
    "BUCKET_RATIO",
    "PopulationData",
    "PopulationSimulator",
    "bucket_size",
    "draw_keys",
    "sample_sources_by_key",
    "segment_ids",
]

#: Geometric growth of the source-draw sizes. Sizes only affect cost -- every
#: distinct one is a compile and padding is drawn and discarded -- so the ratio
#: trades a few more compiles for less discarded sampling. It is not metadata.
BUCKET_RATIO = 1.25


class PopulationData(TypedDict):
    """A batch of population draws; draws first, sources flat and ragged.

    ``total_merger_rate`` and ``counts`` are ``(D,)``, each ``hyperparameters``
    column ``(D,)`` and each ``source_parameters`` column ``(counts.sum(),)``,
    draw ``b`` owning ``offsets[b]:offsets[b + 1]`` of
    ``offsets = concatenate([[0], cumsum(counts)])``.
    """

    total_merger_rate: NDArray[np.float64]
    counts: NDArray[np.int64]
    hyperparameters: dict[str, NDArray[np.float64]]
    source_parameters: dict[str, NDArray[Any]]


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


def draw_keys(keys: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-draw ``(hyperparameter, count, source)`` keys of a batch of ``keys``."""
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


def segment_ids(counts: NDArray[np.int64]) -> NDArray[np.int32]:
    """The draw index of every source, shape ``(counts.sum(),)``."""
    return np.repeat(np.arange(counts.size, dtype=np.int32), counts)


def _hyperparameters_and_counts(
    theta_keys: jax.Array,
    count_keys: jax.Array,
    *,
    merger_rate_fn: MergerRateFn,
    fixed: Mapping[str, float],
    priors: Mapping[str, Distribution],
    observation_seconds: float,
    num_events: int | None,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    """Stage one: hyperparameters, merger rate and source count of every draw."""

    def theta_of(key: jax.Array) -> dict[str, jax.Array]:
        subkeys = jax.random.split(key, max(len(priors), 1))
        return {
            name: jnp.reshape(prior.sample(subkeys[i]), ())
            for i, (name, prior) in enumerate(priors.items())
        }

    draws = theta_keys.shape[0]
    theta = {
        **{name: jnp.full(draws, value) for name, value in fixed.items()},
        **jax.vmap(theta_of)(theta_keys),
    }
    rate = jax.vmap(lambda t: jnp.reshape(merger_rate_fn(t), ()))(theta)
    if num_events is not None:
        counts = jnp.full(draws, num_events, dtype=jnp.int64)
    else:
        counts = jax.vmap(
            lambda key, mean: jax.random.poisson(key, mean).astype(jnp.int64)
        )(count_keys, rate * observation_seconds)
    return theta, rate, counts


class PopulationSimulator:
    """Draws populations for one :class:`PopulationDrawMetadata`; build once, call often.

    ``simulator(keys)`` is one draw per key in the flat layout of
    :class:`~astrogwb.simulators.population.PopulationData`. A draw depends on
    its own key alone, and event ``j`` comes from a key folded with ``j``. ``chunk_size`` sets the ladder
    of sizes sources are drawn at, so it changes cost and compilation, not the
    draw, which is why it is a setting and not metadata. Keys come from
    :func:`~astrogwb.simulators.core.batch_keys`.

    The simulator holds the jitted stages, so reusing one keeps their
    compilations warm.
    """

    def __init__(
        self, metadata: PopulationDrawMetadata, *, chunk_size: int = 128
    ) -> None:
        # x64 before any array: keys are built from 64-bit seeds and the rate
        # must not depend on a waveform import turning it on as a side effect.
        jax.config.update("jax_enable_x64", True)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if metadata.version != __version__:
            raise ValueError(
                f"metadata is for astrogwb {metadata.version}, but {__version__} "
                "is installed; draws are generated by the code their key names"
            )
        built = metadata.population.build()
        if built.merger_rate_fn is None:
            raise ValueError(
                "the population declares no physical merger rate; drawing it "
                "requires an observer-frame rate in both count modes"
            )
        self._metadata = metadata
        self._chunk_size = chunk_size
        self._observation_seconds = years_to_seconds(metadata.observation_time)
        priors = {
            name: metadata.sampled[name].build() for name in sorted(metadata.sampled)
        }
        # Static configuration is bound by ``partial``, so jit traces only the keys.
        self._hyperparameters_and_counts = jax.jit(
            partial(
                _hyperparameters_and_counts,
                merger_rate_fn=built.merger_rate_fn,
                fixed=dict(metadata.fixed),
                priors=priors,
                observation_seconds=self._observation_seconds,
                num_events=metadata.num_events,
            )
        )
        self._draw_sources = jax.jit(
            partial(sample_sources_by_key, built.source_model),
            static_argnames="size",
        )

    @property
    def metadata(self) -> PopulationDrawMetadata:
        """The record this simulator draws."""
        return self._metadata

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds, as Poisson counts use it."""
        return self._observation_seconds

    def __call__(self, keys: jax.Array) -> PopulationData:
        """One population per key, in the flat layout of :class:`PopulationData`."""
        if keys.ndim != 1 or keys.shape[0] == 0:
            raise ValueError(
                f"keys must be a non-empty 1-d batch of keys, got shape {keys.shape}"
            )
        theta_keys, count_keys, source_keys = draw_keys(keys)
        theta, rate, counts_device = self._hyperparameters_and_counts(
            theta_keys, count_keys
        )
        hyperparameters = {n: np.asarray(v, dtype=np.float64) for n, v in theta.items()}
        counts = np.asarray(counts_device, dtype=np.int64)

        columns: dict[str, list[np.ndarray]] = {}
        for draw, count in enumerate(counts):
            params = {name: values[draw] for name, values in hyperparameters.items()}
            sources = self._draw_sources(
                params,
                source_keys[draw],
                size=bucket_size(int(count), self._chunk_size),
            )
            for name, values in sources.items():
                columns.setdefault(name, []).append(np.asarray(values)[: int(count)])
        return {
            "total_merger_rate": np.asarray(rate, dtype=np.float64),
            "counts": counts,
            "hyperparameters": hyperparameters,
            "source_parameters": {
                name: np.concatenate(parts) for name, parts in columns.items()
            },
        }
