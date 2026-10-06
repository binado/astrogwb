"""Draw a population -- hyperparameters, a source count, the sources -- per key.

One key is one draw: it splits into a hyperparameter key, a count key and
a source key, always in that order, so a draw depends on its own key alone and
the same key replays the same events whatever waveform consumes them.

The stages differ in cost and in what must be static:

1. *Hyperparameters and count* are cheap and drawn in one jitted call. The
   count comes back to the host as an integer, because the host uses it to
   decide a shape.
2. *Sources* are drawn at a **bucketed** size, then trimmed to the draw's
   count. Event ``i`` is drawn from ``fold_in(source_key, i)``, so the first
   ``n`` events do not depend on the size drawn: the bucket ladder and
   ``chunk_size`` change cost, never the realization.

Source sampling does not use ``numpyro.plate`` at a rate-derived size, so the
population draw is independent of the merger rate.

:class:`PopulationSimulator` is the waveform-free half of the spectra forward
model; :class:`PopulationData` names one draw. Persisting a draw pays when it is
reused, such as two waveform approximants on the same sources.
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
    "sample_sources_by_key",
]

#: Geometric growth of the source-draw sizes. Sizes only affect cost -- every
#: distinct one is a compile and padding is drawn and discarded -- so the ratio
#: trades a few more compiles for less discarded sampling. It is not metadata.
BUCKET_RATIO = 1.25


class PopulationData(TypedDict):
    """One population draw.

    ``total_merger_rate``, ``count`` and each ``hyperparameters`` value are
    0-d; each ``source_parameters`` column is ``(count,)``.
    """

    total_merger_rate: NDArray[np.float64]
    count: NDArray[np.int64]
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


def _hyperparameters_and_count(
    key: jax.Array,
    *,
    merger_rate_fn: MergerRateFn,
    fixed: Mapping[str, float],
    priors: Mapping[str, Distribution],
    observation_seconds: float,
    num_events: int | None,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    """Stage one: hyperparameters, merger rate and source count of one draw."""
    theta_key, count_key, _ = jax.random.split(key, 3)
    subkeys = jax.random.split(theta_key, max(len(priors), 1))
    theta = {
        **{
            name: jnp.asarray(value, dtype=jnp.float64) for name, value in fixed.items()
        },
        **{
            name: jnp.reshape(prior.sample(subkeys[i]), ())
            for i, (name, prior) in enumerate(priors.items())
        },
    }
    rate = jnp.reshape(merger_rate_fn(theta), ())
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
    from a key folded with ``j``. ``chunk_size`` sets the ladder
    of sizes sources are drawn at, so it changes cost and compilation, not the
    draw, which is why it is a setting and not metadata.

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
        self._hyperparameters_and_count = jax.jit(
            partial(
                _hyperparameters_and_count,
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

    def __call__(self, key: jax.Array) -> PopulationData:
        """One population at ``key``, in the layout of :class:`PopulationData`."""
        if key.shape != ():
            raise ValueError(f"key must be a single key, got shape {key.shape}")
        _, _, source_key = jax.random.split(key, 3)
        theta, rate, count_device = self._hyperparameters_and_count(key)
        hyperparameters = {n: np.asarray(v, dtype=np.float64) for n, v in theta.items()}
        count = int(count_device)
        sources = self._draw_sources(
            hyperparameters,
            source_key,
            size=bucket_size(count, self._chunk_size),
        )
        return {
            "total_merger_rate": np.asarray(rate, dtype=np.float64),
            "count": np.asarray(count, dtype=np.int64),
            "hyperparameters": hyperparameters,
            "source_parameters": {
                name: np.asarray(values)[:count] for name, values in sources.items()
            },
        }
