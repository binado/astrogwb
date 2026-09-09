"""Explicit population contracts shared by generation and importance weighting."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.util import compute_log_probs

#: A registered population: a plain NumPyro model taking the sampled
#: hyperparameters and the model's own construction settings as keywords.
type PopulationFn = Callable[..., None]

#: The raw NumPyro trace escape hatch -- every site, untyped. Only
#: :meth:`Population.trace` returns this; :meth:`Population.evaluate` returns
#: the typed :class:`PopulationTrace` instead.
type RawPopulationTrace = Mapping[str, Mapping[str, Any]]

#: Deterministic sites every registered population declares. Private: nothing
#: outside this module indexes a trace by these names any more.
_LUMINOSITY_DISTANCE_SITE = "luminosity_distance"
_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


class PopulationTrace(NamedTuple):
    """The one model execution's outputs: density, distance, and rate.

    ``total_merger_rate`` is ``None`` when ``params`` carries no physical rate
    -- a proposal density needs none -- rather than propagating a missing
    value into the spectrum.
    """

    log_prob: jax.Array
    """Selected importance-weighting density, one value per source."""

    luminosity_distance: jax.Array
    """Effective distance governing waveform amplitude, in Mpc, shape ``(N,)``."""

    total_merger_rate: jax.Array | None
    """Observer-frame total merger rate, in mergers per second, shape ``()``."""


@dataclass(frozen=True, kw_only=True)
class Population:
    """A NumPyro model with explicit density factors and source outputs.

    ``fn`` is a plain, module-level NumPyro model -- a stable, hashable
    singleton, which is what lets this object serve as static pytree metadata
    without forcing a retrace on every construction. ``settings`` are the
    model's construction keywords, forwarded to ``fn`` alongside ``params`` on
    every call; they are sorted in :meth:`__post_init__` so two ``Population``s
    built from the same settings, in any order, are genuinely equal and hash
    identically.

    ``source_sites`` includes every sampled input needed to replay the model,
    plus selected deterministic outputs. ``density_sites`` names only the
    factors included in importance weighting; omitted factors must cancel
    between the target and proposal. Omitting a factor does not marginalize it.
    """

    fn: PopulationFn
    settings: tuple[tuple[str, float | int], ...]
    density_sites: tuple[str, ...]
    source_sites: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", tuple(sorted(self.settings)))
        object.__setattr__(self, "density_sites", tuple(self.density_sites))
        object.__setattr__(self, "source_sites", tuple(self.source_sites))

    def __call__(self, params: Mapping[str, ArrayLike]) -> None:
        """Declare source sites and population-level quantities with NumPyro."""
        self.fn(params, **dict(self.settings))

    def sample(
        self,
        key: jax.Array,
        params: Mapping[str, ArrayLike],
        *,
        num_samples: int,
    ) -> dict[str, jax.Array]:
        """Draw source outputs; ``num_samples`` must be static under JIT.

        ``Predictive`` draws the sampled inputs, then one batched replay
        recomputes the derived columns identically to evaluation. Predictive's
        per-draw execution can otherwise differ in its final bits, spoiling the
        exact-zero weights of a catalog used as its own proposal. Conditioning
        affects sample sites only, so the deterministic outputs are always the
        model's recomputation.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        with handlers.block():
            draws = Predictive(
                self, num_samples=num_samples, return_sites=self.source_sites
            )(key, params)
            bound = handlers.condition(
                self, data={name: jnp.asarray(value) for name, value in draws.items()}
            )
            trace = handlers.trace(bound).get_trace(params)
        return {name: jnp.asarray(trace[name]["value"]) for name in self.source_sites}

    def log_prob(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> jax.Array:
        """Selected importance-weighting density, one value per source.

        This is not necessarily the full joint or a marginal density. Every
        sampled input must be supplied, including those with omitted factors.
        """
        return self.evaluate(params, sources).log_prob

    def evaluate(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> PopulationTrace:
        """Return selected log density and recomputed deterministics in one pass.

        Effects are isolated from enclosing inference handlers, without
        blocking numerical gradients. Excluded factors execute with supplied
        values but do not reach the log-probability calculation. There is no
        RNG key: evaluation cannot silently draw missing source inputs.
        """
        with handlers.block():
            bound = handlers.condition(
                self, data={name: jnp.asarray(value) for name, value in sources.items()}
            )
            filtered = handlers.block(
                bound,
                hide_fn=lambda site: (
                    site["type"] == "sample" and site["name"] not in self.density_sites
                ),
            )
            log_probs, trace = compute_log_probs(
                filtered, (params,), {}, {}, sum_log_prob=False
            )
        log_prob = jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))
        luminosity_distance = jnp.asarray(trace[_LUMINOSITY_DISTANCE_SITE]["value"])
        total_merger_rate = (
            jnp.asarray(trace[_TOTAL_MERGER_RATE_SITE]["value"])
            if _TOTAL_MERGER_RATE_SITE in trace
            else None
        )
        return PopulationTrace(log_prob, luminosity_distance, total_merger_rate)

    def trace(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> RawPopulationTrace:
        """Raw NumPyro trace escape hatch, isolated from enclosing handlers.

        For consumers that need to inspect an arbitrary site by name -- such as
        the catalog's derived-column consistency check -- rather than the three
        fixed quantities :meth:`evaluate` returns. Runs once at catalog load,
        outside JAX transformations; nothing on the hot inference path uses it.
        """
        with handlers.block():
            bound = handlers.condition(
                self, data={name: jnp.asarray(value) for name, value in sources.items()}
            )
            return handlers.trace(bound).get_trace(params)
