"""NumPyro effect handlers for single-pass model evaluation.

Replaces :func:`numpyro.infer.util.compute_log_probs` for the one call pattern
source models use: condition every sample site, execute once, and read back
both the model's return mapping and the sum of the selected sites' log
densities. :func:`evaluate_sources` and :func:`sample_sources` apply that
pattern to any per-source NumPyro model -- they know nothing about which
populations exist. Distinct from the sibling :mod:`astrogwb.sampling` package,
which composes spectrum callables into inference models; nothing here depends
on it.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.infer import Predictive
from numpyro.primitives import Messenger

#: The plate :func:`evaluate_sources` executes a source model under.
_SOURCES_PLATE = "sources"


class DensityAccumulator(Messenger):
    """Sum ``fn.log_prob(value)`` over selected sample sites in one execution.

    Only sample sites named in ``sites`` are recognised; every other site is
    ignored, matching the ``handlers.block(hide_fn=...)`` filter this replaces.
    Applied around a conditioned model, the sum reproduces
    :func:`numpyro.infer.util.compute_log_probs` operation-for-operation, so
    values are bit-identical -- including the ``intermediates`` and
    plate-subsample ``scale`` handling.
    """

    def __init__(
        self, fn: Callable[..., Any] | None = None, *, sites: Sequence[str]
    ) -> None:
        self.sites = frozenset(sites)
        self._log_probs: list[jax.Array] = []
        super().__init__(fn)

    def __enter__(self) -> Self:
        super().__enter__()
        self._log_probs = []
        return self

    def postprocess_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] != "sample" or msg["name"] not in self.sites:
            return
        intermediates = msg["intermediates"]
        if intermediates:
            log_prob = msg["fn"].log_prob(msg["value"], intermediates)
        else:
            log_prob = msg["fn"].log_prob(msg["value"])
        scale = msg["scale"]
        if scale is not None and not is_identically_one(scale):
            log_prob = scale * log_prob
        self._log_probs.append(log_prob)

    @property
    def log_prob(self) -> jax.Array:
        """Selected density, shape ``(N,)`` or ``()`` when no sites matched."""
        return jax.tree.reduce(operator.add, self._log_probs, initializer=jnp.zeros(()))


def compute_model_and_log_probs(
    model: Callable[..., Any],
    density_sites: Sequence[str],
    *model_args: Any,
    **model_kwargs: Any,
) -> tuple[Any, jax.Array]:
    """Run ``model`` once in isolation; return its result and selected density.

    ``model`` must already be wrapped for whatever the caller needs -- such as
    ``handlers.condition`` supplying fixed inputs -- because the helper cannot
    know a model's arguments. Execution is isolated from enclosing handlers
    with ``handlers.block``, which is what keeps catalog rows out of an
    enclosing inference trace.

    Returns ``(model return value, log_prob)``. ``log_prob`` has one entry per
    source, or is the scalar zero if no ``density_sites`` matched.
    """
    with handlers.block():
        accumulator = DensityAccumulator(model, sites=density_sites)
        result = accumulator(*model_args, **model_kwargs)
    return result, accumulator.log_prob


class _RequireObserved(Messenger):
    """Raise ``KeyError`` naming any sample site that no condition supplied.

    Evaluation has no RNG key, so an unconditioned site cannot draw -- but
    under an enclosing ``seed`` it silently would, turning a missing column
    into ``N`` fresh latent values. Applied *outside* ``handlers.condition``,
    this messenger sees ``is_observed`` after conditioning has set it.
    """

    def process_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] == "sample" and not msg["is_observed"]:
            raise KeyError(
                f"no source column supplies sample site {msg['name']!r}; every "
                "sampled input must be given, including those whose density "
                "factor is excluded"
            )


def _num_sources(columns: Mapping[str, ArrayLike]) -> int:
    """The shared length ``N`` of one-dimensional source columns."""
    if not columns:
        raise ValueError("at least one source column is required")
    shapes = {name: jnp.shape(values) for name, values in columns.items()}
    lengths = {shape[0] for shape in shapes.values() if len(shape) == 1}
    if len(lengths) != 1 or any(len(shape) != 1 for shape in shapes.values()):
        raise ValueError(
            f"every source column must have the same shape (N,); got {shapes}"
        )
    return lengths.pop()


def evaluate_sources(
    source_model: Callable[[Mapping[str, ArrayLike]], Mapping[str, jax.Array]],
    params: Mapping[str, ArrayLike],
    source_parameters: Mapping[str, ArrayLike],
    *,
    density_sites: Sequence[str],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Selected log density and the model's returned mapping, in one isolated pass.

    ``source_model`` runs once under a ``sources`` plate of length ``N``, with
    every column of ``source_parameters`` conditioned in. Conditioning only
    reaches *sample* sites: a stored deterministic column -- including
    ``luminosity_distance`` -- is ignored, and the returned mapping holds the
    model's own recomputation of it. ``density_sites`` names the factors summed
    into the log density; omitted factors execute but contribute nothing.

    Execution is isolated from enclosing handlers with ``handlers.block``, so
    no source site reaches an outer trace or potential, while gradients with
    respect to ``params`` flow normally.

    Returns ``(log_prob, outputs)``. ``log_prob`` has shape ``(N,)``, zeros when
    ``density_sites`` is empty. ``outputs`` is the model's return mapping.

    Raises ``ValueError`` unless every column has the same shape ``(N,)``, and
    ``KeyError`` if a sample site has no column.
    """
    columns: dict[str, ArrayLike] = {
        name: jnp.asarray(value) for name, value in source_parameters.items()
    }
    num_sources = _num_sources(columns)

    def plated(site_params: Mapping[str, ArrayLike]) -> Mapping[str, jax.Array]:
        with numpyro.plate(_SOURCES_PLATE, num_sources):
            return source_model(site_params)

    model = _RequireObserved(handlers.condition(plated, data=columns))
    outputs, log_prob = compute_model_and_log_probs(model, density_sites, params)
    return (
        jnp.broadcast_to(log_prob, (num_sources,)),
        {name: jnp.asarray(value) for name, value in outputs.items()},
    )


def sample_sources(
    source_model: Callable[[Mapping[str, ArrayLike]], Mapping[str, jax.Array]],
    key: jax.Array,
    params: Mapping[str, ArrayLike],
    *,
    num_samples: int,
) -> dict[str, jax.Array]:
    """Draw ``num_samples`` sources, then replay them through :func:`evaluate_sources`.

    ``Predictive`` draws the sampled inputs from the unplated model; one
    batched replay then recomputes every returned column through the same code
    path evaluation uses. Predictive's per-draw execution can differ from a
    batched one in its final bits, and a stored column that differs from its
    later recomputation spoils the exact-zero weights of a catalog used as its
    own proposal. The replay conditions only the *sample* sites, read off a
    plain probe trace, so a deterministic is always the model's recomputation.

    ``num_samples`` is a Python int, static under JIT. Returns the model's
    output mapping, each column of shape ``(num_samples,)``.
    """
    with handlers.block():
        draws = Predictive(source_model, num_samples=num_samples)(key, params)
        probe = handlers.trace(
            handlers.seed(source_model, jax.random.PRNGKey(0))
        ).get_trace(params)
    sampled = {
        name: draws[name] for name, site in probe.items() if site["type"] == "sample"
    }
    _, outputs = evaluate_sources(source_model, params, sampled, density_sites=())
    return outputs


__all__ = [
    "DensityAccumulator",
    "compute_model_and_log_probs",
    "evaluate_sources",
    "sample_sources",
]
