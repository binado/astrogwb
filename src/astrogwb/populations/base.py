"""Explicit population contracts shared by generation and importance weighting.

A population is two independent declarations composed together: a
:class:`SourceModel` draws and evaluates per-source quantities (masses,
spins, redshift, distance, ...), and a :data:`MergerRateFn` returns one
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
from typing import Any

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

#: A bound merger-rate callable: returns one observer-frame scalar,
#: mergers per second, from hyperparameters alone. Construction settings
#: are closed over by the caller that assembled the :class:`Population`,
#: as a :func:`functools.partial` over the registered rate function.
type MergerRateFn = Callable[[Mapping[str, ArrayLike]], jax.Array]

#: The raw NumPyro trace escape hatch -- every site, untyped. Only
#: :meth:`SourceModel.trace` returns this.
type RawPopulationTrace = Mapping[str, Mapping[str, Any]]

#: Deterministic site every source model must declare. Private: density
#: evaluation reads it from the model's *return value*, not from a trace.
_LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

#: The population-level rate site name. Reserved: a source model returning
#: this key would collide with the merger-rate function's own declaration.
_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


@dataclass(frozen=True, kw_only=True)
class SourceModel:
    """A NumPyro model declaring per-source sites, with no notion of a rate.

    ``fn`` is a plain, module-level NumPyro model -- a stable, hashable
    singleton, which is what lets this object serve as static pytree metadata
    without forcing a retrace on every construction. ``model_kwargs`` are the
    model's construction keywords, forwarded to ``fn`` alongside ``params`` on
    every call.

    ``density_sites`` names only the factors included in importance
    weighting; omitted factors must cancel between the target and proposal.
    Omitting a factor does not marginalize it.
    """

    fn: SourceFn
    model_kwargs: tuple[tuple[str, float | int], ...]
    density_sites: tuple[str, ...]

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
                "name is reserved for the merger-rate function it is paired with"
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
        """Draw source outputs; ``num_samples`` is a Python int, static under JIT.

        ``Predictive`` draws the sampled
        inputs, then one batched replay
        recomputes the derived columns identically to evaluation. Predictive's
        per-draw execution can otherwise differ in its final bits, spoiling the
        exact-zero weights of a catalog used as its own proposal. Conditioning
        is restricted to *sample* sites -- read off a plain probe trace, not
        off ``Predictive``'s full return value -- so a deterministic is always
        the model's own recomputation, never a value handed back to it.
        """
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

        Returns an array of shape ``(N,)`` matching the source columns, or
        ``()`` when ``density_sites`` is empty (the sum of no factors).
        """
        log_prob, _ = self.evaluate(params, sources)
        return log_prob

    def evaluate(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, jax.Array]:
        """Return selected log density and the recomputed distance in one pass.

        Effects are isolated from enclosing inference handlers, without
        blocking numerical gradients. Excluded factors execute with supplied
        values but do not reach the log-probability calculation. There is no
        RNG key: evaluation cannot silently draw missing source inputs.

        ``luminosity_distance`` comes from the model's *return value*, read
        out of a single execution alongside the log densities -- not from the
        trace's own copy of the same site -- which is what keeps the return
        mapping the one authoritative source of derived columns.

        Returns ``(log_prob, luminosity_distance)``. ``log_prob`` is the
        selected importance-weighting density, shape ``(N,)`` or ``()`` when
        ``density_sites`` is empty. ``luminosity_distance`` is the effective
        distance governing waveform amplitude, in Mpc, shape ``(N,)``.
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
        return log_prob, luminosity_distance

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
class Population:
    """A source model and a bound merger-rate function, composed.

    The rate is published *outside* the plate the source model draws inside:
    that is the property that used to require executing the whole source
    model under a discarded ``handlers.seed`` just to read one scalar, and
    hiding the plated re-declaration with ``handlers.block``. Composing two
    declarations instead of branching one on ``"local_merger_rate" in params``
    is what makes ``total_merger_rate`` always present rather than optional.

    ``rate`` is already ``(params) -> Array``: construction settings are bound
    by the caller that assembled this object, normally
    :func:`~astrogwb.populations.registry.build_population`, which returns a
    :func:`functools.partial`. It is compared and hashed **by identity**, so a
    ``Population`` used as static metadata in a ``jax.jit`` call must be built
    once and reused -- reconstructing an equal population forces a fresh
    compile.
    """

    source: SourceModel
    rate: MergerRateFn

    def __call__(
        self, params: Mapping[str, ArrayLike], *, num_events: int
    ) -> tuple[Mapping[str, jax.Array], jax.Array]:
        """Publish the rate, then draw ``num_events`` sources under a plate.

        ``num_events`` is a Python integer, static under JIT.

        Returns ``(sources, total_merger_rate)``. ``sources`` maps column
        name to an array of shape ``(num_events,)``. ``total_merger_rate`` is
        the observer-frame total merger rate, in mergers per second, shape
        ``()``.
        """
        total_merger_rate = _as_scalar_rate(self.rate(params))
        numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
        with numpyro.plate("events", num_events):
            sources = dict(self.source(params))
        return sources, total_merger_rate

    def evaluate(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One rate call plus one source evaluation, both handler-isolated.

        Returns ``(log_prob, luminosity_distance, total_merger_rate)``.
        ``log_prob`` and ``luminosity_distance`` are the source-evaluation
        pair: selected density of shape ``(N,)`` (or ``()`` if no density
        sites) and effective distance in Mpc of shape ``(N,)``.
        ``total_merger_rate`` is the observer-frame total merger rate, in
        mergers per second, shape ``()``.
        """
        with handlers.block():
            total_merger_rate = _as_scalar_rate(self.rate(params))
            numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
            log_prob, luminosity_distance = self.source.evaluate(params, sources)
        return log_prob, luminosity_distance, total_merger_rate


def _as_scalar_rate(value: ArrayLike) -> jax.Array:
    """Coerce a rate return value to a shape-``()`` array."""
    return jnp.reshape(jnp.asarray(value), ())
