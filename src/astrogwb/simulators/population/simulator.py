"""Draw a population -- hyperparameters, a source count, the sources -- per key.

One key is one draw: it splits into a hyperparameter key, a count key and
a source key, always in that order, so a draw depends on its own key alone and
the same key replays the same events whatever waveform consumes them.

The stages differ in cost and in what must be static:

1. *Hyperparameters and count* are cheap and drawn in one jitted call. The
   count comes back to the host as an integer, because the host uses it to
   decide a shape.
2. *Sources* are drawn in one vmap at a static size. A fixed ``num_events``
   is one size, so it is drawn exactly. A Poisson count differs per key, and
   each distinct size is a compile, so the sources are drawn in fixed
   ``chunk_size`` pieces -- one compile -- and the last piece is trimmed to the
   count. Event ``i`` is drawn from ``fold_in(source_key, i)``, wherever it
   falls, so ``chunk_size`` changes cost, never the realization.

Source sampling does not use ``numpyro.plate`` at a rate-derived size, so the
population draw is independent of the merger rate.

:class:`PopulationSimulator` is the waveform-free half of the spectra forward
model; :class:`PopulationData` names one draw. Persisting a draw pays when it is
reused, such as two waveform approximants on the same sources.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from numpyro import handlers

from astrogwb import __version__
from astrogwb.populations import Population, joint_model
from astrogwb.populations.evaluation import evaluate_sources
from astrogwb.simulators.population.metadata import PopulationDrawMetadata
from astrogwb.utils import years_to_seconds
from astrogwb.utils.numpyro import sample_model

__all__ = [
    "PopulationData",
    "PopulationSimulator",
    "bucket_size",
    "sample_sources_by_key",
]


class PopulationData(TypedDict):
    """One population draw.

    ``total_merger_rate``, ``count`` and each ``hyperparameters`` value are
    0-d; each ``source_parameters`` column is ``(count,)``.
    """

    total_merger_rate: NDArray[np.float64]
    count: NDArray[np.int64]
    hyperparameters: dict[str, NDArray[np.float64]]
    source_parameters: dict[str, NDArray[Any]]


def bucket_size(count: int, chunk_size: int, ratio: float) -> int:
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


def sample_sources_by_key(
    population: Population,
    params: Mapping[str, Any],
    key: jax.Array,
    start: jax.Array,
    size: int,
) -> dict[str, jax.Array]:
    """Events ``start`` to ``start + size``, event ``i`` drawn from ``fold_in(key, i)``.

    The counterpart of :func:`~astrogwb.populations.evaluation.sample_sources`
    with the same replay -- sampled sites are drawn, then every returned column,
    deterministics included, is recomputed through
    :func:`~astrogwb.populations.evaluation.evaluate_sources` -- but with one key
    per event instead of ``split(key, size)``. Splitting a key into ``n`` and
    into ``m > n`` children does not share a prefix, so there the drawn events
    would depend on ``size``; here an event depends on its index alone, so a
    population can be drawn in pieces of any size. ``start`` is a ``uint32``
    scalar, traced, so every piece of one ``size`` shares a compile.
    """
    event_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(
        start + jnp.arange(size, dtype=jnp.uint32)
    )
    # Built once, outside the vmap: the hyperparameters are unbatched, so the
    # redshift grid is computed once, not once per event.
    model = joint_model(*population(params))
    sampled = jax.vmap(lambda k: sample_model(model, k))(event_keys)
    _, outputs = evaluate_sources(model, sampled, density_sites=())
    return outputs


def _hyperparameters_and_count(
    key: jax.Array,
    *,
    population: Population,
    prior_model: Callable[[], Mapping[str, jax.Array]],
    observation_seconds: float,
    num_events: int | None,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    """Stage one: hyperparameters, merger rate and source count of one draw."""
    theta_key, count_key, _ = jax.random.split(key, 3)
    theta = dict(handlers.seed(prior_model, theta_key)())
    redshift_distribution, _ = population(theta)
    rate = jnp.reshape(redshift_distribution.total_merger_rate(), ())
    if num_events is not None:
        count = jnp.asarray(num_events, dtype=jnp.int64)
    else:
        count = jax.random.poisson(count_key, rate * observation_seconds).astype(
            jnp.int64
        )
    return theta, rate, count


class PopulationSimulator:
    """Draws populations for one :class:`PopulationDrawMetadata`; build once, call often.

    ``simulator(key)`` is one draw in the layout of
    :class:`~astrogwb.simulators.population.PopulationData`; loop over
    :func:`~astrogwb.simulators.core.batch_keys` for several. Event ``j`` comes
    from a key folded with ``j``. With Poisson counts, ``chunk_size`` is the
    size of the pieces sources are drawn in, one compile and ``ceil(count /
    chunk_size)`` calls; a fixed ``num_events`` is drawn in one piece. It changes
    cost, not the draw, which is why it is a setting and not metadata.

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
        population = metadata.population.build()
        self._metadata = metadata
        self._chunk_size = chunk_size
        self._observation_seconds = years_to_seconds(metadata.observation_time)
        # Static configuration is bound by ``partial``, so jit traces only the keys.
        self._hyperparameters_and_count = jax.jit(
            partial(
                _hyperparameters_and_count,
                population=population,
                prior_model=metadata.prior_model(),
                observation_seconds=self._observation_seconds,
                num_events=metadata.num_events,
            )
        )
        self._draw_sources = jax.jit(
            partial(sample_sources_by_key, population),
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

    def __call__(self, key: jax.Array) -> PopulationData:
        """One population at ``key``, in the layout of :class:`PopulationData`."""
        if key.shape != ():
            raise ValueError(f"key must be a single key, got shape {key.shape}")
        _, _, source_key = jax.random.split(key, 3)
        theta, rate, count_device = self._hyperparameters_and_count(key)
        hyperparameters = {n: np.asarray(v, dtype=np.float64) for n, v in theta.items()}
        count = int(count_device)
        sources = self._sources(hyperparameters, source_key, count)
        return {
            "total_merger_rate": np.asarray(rate, dtype=np.float64),
            "count": np.asarray(count, dtype=np.int64),
            "hyperparameters": hyperparameters,
            "source_parameters": {
                name: values[:count] for name, values in sources.items()
            },
        }

    def _sources(
        self, hyperparameters: Mapping[str, Any], key: jax.Array, count: int
    ) -> dict[str, NDArray[Any]]:
        """Columns of at least ``count`` events: one piece, or whole chunks."""
        if self._metadata.num_events is not None:
            # One count for every key, so one static size and one compile.
            drawn = self._draw_sources(
                hyperparameters, key, np.uint32(0), size=max(count, 1)
            )
            return {name: np.asarray(values) for name, values in drawn.items()}
        pieces = [
            self._draw_sources(
                hyperparameters, key, np.uint32(start), size=self._chunk_size
            )
            for start in range(0, max(count, 1), self._chunk_size)
        ]
        return {
            name: np.concatenate([np.asarray(piece[name]) for piece in pieces])
            for name in pieces[0]
        }
