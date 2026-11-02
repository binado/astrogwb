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
from scipy.stats import qmc

from astrogwb.populations._types import PopulationModel
from astrogwb.utils.numpyro import InverseCDF

#: The plate :func:`evaluate_sources` executes a source model under.
_SOURCES_PLATE = "sources"


class DensityAccumulator(Messenger):
    """Sum ``fn.log_prob(value)`` over selected sample sites in one execution.

    Other sites are ignored. Around a conditioned model, the sum reproduces
    :func:`numpyro.infer.util.compute_log_probs` operation for operation,
    including ``intermediates`` and plate ``scale``, so values are bit-identical.
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
        """Accumulate the density contribution from a selected sample site.

        Parameters
        ----------
        msg
            NumPyro effect message.
        """
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

    ``model`` must already be wrapped as needed, e.g. by ``handlers.condition``.
    ``handlers.block`` keeps catalog rows out of an enclosing inference trace.

    Returns ``(model return value, log_prob)``. ``log_prob`` has one entry per
    source, or is the scalar zero if no ``density_sites`` matched.

    Parameters
    ----------
    model
        NumPyro model.
    density_sites
        Names of sites included in the density.
    *model_args
        Positional arguments passed to the model.
    **model_kwargs
        Keyword arguments passed to the model.
    """
    with handlers.block():
        accumulator = DensityAccumulator(model, sites=density_sites)
        result = accumulator(*model_args, **model_kwargs)
    return result, accumulator.log_prob


class _RequireObserved(Messenger):
    """Raise ``KeyError`` naming any sample site that no condition supplied.

    Under an enclosing ``seed`` a missing column would otherwise silently draw
    ``N`` fresh values. Must wrap ``handlers.condition`` to see ``is_observed``.
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
    model: PopulationModel,
    source_parameters: Mapping[str, ArrayLike],
    *,
    density_sites: Sequence[str],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Selected log density and the model's returned mapping, in one isolated pass.

    ``model`` is a no-argument source model, built from the hyperparameters it
    closes over, so gradients with respect to them flow through its closure.
    It runs once under a ``sources`` plate of length ``N``, with every column
    of ``source_parameters`` conditioned in. Stored deterministic columns, such
    as ``luminosity_distance``, are ignored and recomputed. ``density_sites``
    names the factors summed into the log density. Execution is blocked from
    enclosing handlers.

    Returns ``(log_prob, outputs)``. ``log_prob`` has shape ``(N,)``, zeros when
    ``density_sites`` is empty. ``outputs`` is the model's return mapping.

    Raises ``ValueError`` unless every column has the same shape ``(N,)``, and
    ``KeyError`` if a sample site has no column.

    Parameters
    ----------
    model
        NumPyro model.
    source_parameters
        Source parameter values by name.
    density_sites
        Names of sites included in the density.
    """
    columns: dict[str, ArrayLike] = {
        name: jnp.asarray(value) for name, value in source_parameters.items()
    }
    num_sources = _num_sources(columns)

    def plated() -> Mapping[str, jax.Array]:
        with numpyro.plate(_SOURCES_PLATE, num_sources):
            return model()

    conditioned = _RequireObserved(handlers.condition(plated, data=columns))
    outputs, log_prob = compute_model_and_log_probs(conditioned, density_sites)
    return (
        jnp.broadcast_to(log_prob, (num_sources,)),
        {name: jnp.asarray(value) for name, value in outputs.items()},
    )


def sample_sources(
    model: PopulationModel,
    key: jax.Array,
    *,
    num_samples: int,
) -> dict[str, jax.Array]:
    """Draw ``num_samples`` sources, then replay them through :func:`evaluate_sources`.

    The batched replay recomputes every returned column through the evaluation
    code path: ``Predictive``'s per-draw execution can differ in the last bits,
    which would spoil the exact-zero weights of a catalog used as its own
    proposal.

    ``num_samples`` is a Python int, static under JIT. Returns the model's
    output mapping, each column of shape ``(num_samples,)``.

    Parameters
    ----------
    model
        NumPyro model.
    key
        JAX random key.
    num_samples
        Number of source samples.
    """
    with handlers.block():
        draws = Predictive(model, num_samples=num_samples)(key)
        probe = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace()
    sampled = {
        name: draws[name] for name, site in probe.items() if site["type"] == "sample"
    }
    _, outputs = evaluate_sources(model, sampled, density_sites=())
    return outputs


def sample_sources_qmc(
    model: PopulationModel,
    key: jax.Array,
    *,
    num_samples: int,
) -> dict[str, jax.Array]:
    """Draw ``num_samples`` sources on a scrambled Sobol net, then replay them.

    The quasi-random counterpart of :func:`sample_sources`: ``key`` seeds an
    Owen scrambling of a Sobol sequence with one dimension per free sample
    site, driven through :class:`~astrogwb.utils.numpyro.InverseCDF`, whose
    contracts the model must meet.

    Parameters
    ----------
    model
        NumPyro model.
    key
        JAX random key.
    num_samples
        Number of source samples.

    Raises
    ------
    ValueError
        If ``num_samples`` is not a power of two: only those prefixes of a Sobol
        sequence are balanced nets.
    """
    exponent = int(num_samples).bit_length() - 1
    if num_samples <= 0 or 1 << exponent != num_samples:
        raise ValueError(
            f"num_samples must be a power of two for a Sobol net, got {num_samples}"
        )
    with handlers.block():
        probe = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace()
    samples = [name for name, site in probe.items() if site["type"] == "sample"]
    free = [name for name in samples if not probe[name]["is_observed"]]
    scramble = int(jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
    points = qmc.Sobol(len(free), scramble=True, seed=scramble).random_base2(exponent)
    with handlers.block():
        trace = handlers.trace(InverseCDF(model, points=points)).get_trace()
    sampled = {
        name: jnp.broadcast_to(trace[name]["value"], (num_samples,)) for name in samples
    }
    _, outputs = evaluate_sources(model, sampled, density_sites=())
    return outputs


__all__ = [
    "DensityAccumulator",
    "compute_model_and_log_probs",
    "evaluate_sources",
    "sample_sources",
    "sample_sources_qmc",
]
