"""Explicit population contracts shared by generation and importance weighting.

A population is two independent declarations composed together: a
:class:`SourceModel` draws and evaluates per-source quantities (masses,
spins, redshift, distance, ...), and a :class:`MergerRateModel` returns one
observer-frame scalar. Composing them into a :class:`Population` is what a
registered population used to do in one callable; splitting them is what lets
a guard-mixture source pair with the same Madau-Dickinson rate the physical
population uses, and what removes the ``total_merger_rate`` deterministic from
the plated draw path instead of hiding it with a handler.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.util import compute_log_probs

#: A registered source model: declares per-source sites as a side effect and
#: returns the mapping that defines the source-output set -- the columns a
#: catalog stores. ``luminosity_distance`` is required in it;
#: ``total_merger_rate`` is a reserved population-level name and is rejected.
type SourceFn = Callable[..., Mapping[str, jax.Array]]

#: A registered merger-rate model: returns one observer-frame scalar,
#: mergers per second, executed once outside any plate.
type MergerRateFn = Callable[..., jax.Array]

#: The raw NumPyro trace escape hatch -- every site, untyped. Only
#: :meth:`SourceModel.trace` returns this.
type RawPopulationTrace = Mapping[str, Mapping[str, Any]]

#: Deterministic site every source model must declare. Private: density
#: evaluation reads it from the model's *return value*, not from a trace.
_LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

#: The population-level rate site name. Reserved: a source model returning
#: this key would collide with the rate model's own declaration.
_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


class SourceEvaluation(NamedTuple):
    """One source-model execution's outputs: selected density and distance."""

    log_prob: jax.Array
    """Selected importance-weighting density, one value per source."""

    luminosity_distance: jax.Array
    """Effective distance governing waveform amplitude, in Mpc, shape ``(N,)``."""


class PopulationEvaluation(NamedTuple):
    """One population evaluation's outputs: density, distance, and rate."""

    log_prob: jax.Array
    """Selected importance-weighting density, one value per source."""

    luminosity_distance: jax.Array
    """Effective distance governing waveform amplitude, in Mpc, shape ``(N,)``."""

    total_merger_rate: jax.Array
    """Observer-frame total merger rate, in mergers per second, shape ``()``."""


class PopulationDraw(NamedTuple):
    """One population draw: the plated source columns and the scalar rate."""

    sources: Mapping[str, jax.Array]
    """Source columns, each shape ``(num_events,)``; empty when ``num_events == 0``."""

    total_merger_rate: jax.Array
    """Observer-frame total merger rate, in mergers per second, shape ``()``."""


@dataclass(frozen=True, kw_only=True)
class SourceModel:
    """A NumPyro model declaring per-source sites, with no notion of a rate.

    ``fn`` is a plain, module-level NumPyro model -- a stable, hashable
    singleton, which is what lets this object serve as static pytree metadata
    without forcing a retrace on every construction. ``model_kwargs`` are the
    model's construction keywords, forwarded to ``fn`` alongside ``params`` on
    every call; they are sorted in :meth:`__post_init__` so two
    ``SourceModel``s built from the same kwargs, in any order, are genuinely
    equal and hash identically.

    ``density_sites`` names only the factors included in importance
    weighting; omitted factors must cancel between the target and proposal.
    Omitting a factor does not marginalize it.
    """

    fn: SourceFn
    model_kwargs: tuple[tuple[str, float | int], ...]
    density_sites: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", tuple(sorted(self.model_kwargs)))
        object.__setattr__(self, "density_sites", tuple(self.density_sites))

    def __call__(self, params: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
        """Declare sites and return the mapping that defines the source outputs.

        The returned mapping is authoritative: it is what a catalog stores,
        what :meth:`evaluate` reads ``luminosity_distance`` from, and what a
        replay must reproduce column-for-column.
        """
        sources = {
            name: jnp.asarray(value)
            for name, value in self.fn(params, **dict(self.model_kwargs)).items()
        }
        if _TOTAL_MERGER_RATE_SITE in sources:
            raise ValueError(
                f"source model must not return {_TOTAL_MERGER_RATE_SITE!r}: that "
                "name is reserved for the merger-rate model it is paired with"
            )
        if _LUMINOSITY_DISTANCE_SITE not in sources:
            raise ValueError(
                f"source model must return {_LUMINOSITY_DISTANCE_SITE!r}: it is "
                "the distance governing waveform amplitude"
            )
        return sources

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
        is restricted to *sample* sites -- read off a plain probe trace, not
        off ``Predictive``'s full return value -- so a deterministic is always
        the model's own recomputation, never a value handed back to it.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        with handlers.block():
            draws = Predictive(self, num_samples=num_samples)(key, params)
            probe = handlers.trace(
                handlers.seed(self, jax.random.PRNGKey(0))
            ).get_trace(params)
            sample_names = {
                name for name, site in probe.items() if site["type"] == "sample"
            }
            bound = handlers.condition(
                self, data={name: jnp.asarray(draws[name]) for name in sample_names}
            )
            return bound(params)

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
    ) -> SourceEvaluation:
        """Return selected log density and the recomputed distance in one pass.

        Effects are isolated from enclosing inference handlers, without
        blocking numerical gradients. Excluded factors execute with supplied
        values but do not reach the log-probability calculation. There is no
        RNG key: evaluation cannot silently draw missing source inputs.

        ``luminosity_distance`` comes from the model's *return value*, read
        out of a single execution alongside the log densities -- not from the
        trace's own copy of the same site -- which is what keeps the return
        mapping the one authoritative source of derived columns.
        """
        captured: list[Mapping[str, jax.Array]] = []

        def _capture(p: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
            result = self(p)
            captured.append(result)
            return result

        with handlers.block():
            bound = handlers.condition(
                _capture,
                data={name: jnp.asarray(value) for name, value in sources.items()},
            )
            filtered = handlers.block(
                bound,
                hide_fn=lambda site: (
                    site["type"] == "sample" and site["name"] not in self.density_sites
                ),
            )
            log_probs, _ = compute_log_probs(
                filtered, (params,), {}, {}, sum_log_prob=False
            )
        log_prob = jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))
        result = captured[0]
        luminosity_distance = jnp.asarray(result[_LUMINOSITY_DISTANCE_SITE])
        return SourceEvaluation(
            log_prob=log_prob, luminosity_distance=luminosity_distance
        )

    def trace(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> RawPopulationTrace:
        """Raw NumPyro trace escape hatch, isolated from enclosing handlers.

        For consumers that need to inspect an arbitrary site by name -- such as
        the test suite's derived-column recomputation. Runs once, outside JAX
        transformations; nothing on the hot inference path uses it.
        """
        with handlers.block():
            bound = handlers.condition(
                self, data={name: jnp.asarray(value) for name, value in sources.items()}
            )
            return handlers.trace(bound).get_trace(params)


@dataclass(frozen=True, kw_only=True)
class MergerRateModel:
    """A NumPyro model returning one observer-frame scalar rate.

    ``fn`` is a plain, module-level NumPyro model, a stable, hashable
    singleton for the same reason :class:`SourceModel.fn` is. ``model_kwargs``
    are sorted in :meth:`__post_init__` for the same equality/hashing reason.
    Has no ``density_sites``: a deterministic scalar carries no density.
    """

    fn: MergerRateFn
    model_kwargs: tuple[tuple[str, float | int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", tuple(sorted(self.model_kwargs)))

    def __call__(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        return jnp.reshape(jnp.asarray(self.fn(params, **dict(self.model_kwargs))), ())


@dataclass(frozen=True, kw_only=True)
class Population:
    """A source model and a merger-rate model, composed.

    The rate is published *outside* the plate the source model draws inside:
    that is the property that used to require executing the whole source
    model under a discarded ``handlers.seed`` just to read one scalar, and
    hiding the plated re-declaration with ``handlers.block``. Composing two
    declarations instead of branching one on ``"local_merger_rate" in params``
    is what makes ``total_merger_rate`` always present rather than optional.
    """

    source: SourceModel
    rate: MergerRateModel

    def __call__(
        self, params: Mapping[str, ArrayLike], *, num_events: int
    ) -> PopulationDraw:
        """Publish the rate, then draw ``num_events`` sources under a plate.

        ``num_events`` is a Python integer, static under JIT: NumPyro plates
        reject size 0, so ``num_events <= 0`` skips the plate and returns an
        empty source mapping.
        """
        total_merger_rate = self.rate(params)
        numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
        if num_events <= 0:
            return PopulationDraw(sources={}, total_merger_rate=total_merger_rate)
        with numpyro.plate("events", num_events):
            sources = dict(self.source(params))
        return PopulationDraw(sources=sources, total_merger_rate=total_merger_rate)

    def evaluate(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> PopulationEvaluation:
        """One rate call plus one source evaluation, both handler-isolated."""
        with handlers.block():
            total_merger_rate = self.rate(params)
            numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
            source_eval = self.source.evaluate(params, sources)
        return PopulationEvaluation(
            log_prob=source_eval.log_prob,
            luminosity_distance=source_eval.luminosity_distance,
            total_merger_rate=total_merger_rate,
        )
