"""NumPyro handlers and batched sampling utilities shared across the package."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.primitives import Messenger
from scipy.stats import qmc

__all__ = ["InverseCDF", "sample_and_logprob", "sample_sources_qmc"]


class InverseCDF(Messenger):
    r"""Fill a model's free sample sites from given points through their inverse CDFs.

    At each free sample site, in execution order, the value is
    ``site.fn.icdf(points[..., d])`` with ``d`` the site's position among the
    free sites, so the ``d``-th coordinate of every point drives the ``d``-th
    site. A site is free when no inner handler has fixed its value: sites a
    :func:`numpyro.handlers.condition` *inside* this handler observes consume
    no coordinate. A conditional site is filled after its parents, so its
    inverse CDF is the conditional one (the Rosenblatt transform): uniform
    points on the unit cube map to the model's joint law.

    Unchecked contracts:

    - ``points`` has one trailing coordinate per free site, in ``(0, 1)``.
    - Every free site's distribution has an ``icdf`` that broadcasts over the
      leading shape of ``points``; a model without plates is then evaluated
      once for every point.
    - Any ``condition`` the model needs is applied inside this handler, never
      around it.

    The leading coordinates of a Sobol sequence are the most uniform, so a
    model should sample its most influential parameters first.

    Parameters
    ----------
    fn
        The model.
    points
        Shape ``(..., D)``, ``D`` the number of free sites.
    """

    def __init__(
        self, fn: Callable[..., Any] | None = None, points: ArrayLike | None = None
    ) -> None:
        self.points = jnp.asarray(points)
        self._index = 0
        super().__init__(fn)

    def __enter__(self) -> Any:
        self._index = 0
        return super().__enter__()

    def process_message(self, msg: dict[str, Any]) -> None:
        """Transform the next uniform coordinate at an eligible sample site.

        Parameters
        ----------
        msg
            NumPyro effect message.
        """
        if msg["type"] != "sample" or msg["is_observed"] or msg["value"] is not None:
            return
        msg["value"] = msg["fn"].icdf(self.points[..., self._index])
        self._index += 1


class _RequireObserved(Messenger):
    """Raise ``KeyError`` naming any sample site that no condition supplied.

    Under an enclosing ``seed`` a missing column would otherwise silently draw
    ``N`` fresh values. Must wrap ``handlers.condition`` to see ``is_observed``.
    """

    def process_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] == "sample" and not msg["is_observed"]:
            raise KeyError(
                f"no sample column supplies sample site {msg['name']!r}; every "
                "sampled input must be given, including those whose density "
                "factor is excluded"
            )


def _num_samples(columns: Mapping[str, ArrayLike]) -> int:
    """The shared length ``N`` of one-dimensional sample columns."""
    if not columns:
        raise ValueError("at least one sample column is required")
    shapes = {name: jnp.shape(values) for name, values in columns.items()}
    lengths = {shape[0] for shape in shapes.values() if len(shape) == 1}
    if len(lengths) != 1 or any(len(shape) != 1 for shape in shapes.values()):
        raise ValueError(
            f"every sample column must have the same shape (N,); got {shapes}"
        )
    return lengths.pop()


def _outputs_and_logprob(
    trace: Mapping[str, Any], num_samples: int, density_sites: Sequence[str]
) -> tuple[dict[str, jax.Array], jax.Array]:
    outputs = {
        name: jnp.broadcast_to(jnp.asarray(site["value"]), (num_samples,))
        for name, site in trace.items()
        if site["type"] in {"sample", "deterministic"}
        and not site.get("infer", {}).get("is_auxiliary", False)
    }
    log_probs = {}
    for name in density_sites:
        if name not in trace or trace[name]["type"] != "sample":
            continue
        site = trace[name]
        intermediates = site["intermediates"]
        log_prob = (
            site["fn"].log_prob(site["value"], intermediates)
            if intermediates
            else site["fn"].log_prob(site["value"])
        )
        scale = site["scale"]
        if scale is not None and not is_identically_one(scale):
            log_prob = scale * log_prob
        log_probs[name] = log_prob
    total = jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))
    return outputs, jnp.broadcast_to(total, (num_samples,))


def sample_and_logprob(
    model: Callable[[], None],
    key: jax.Array | None = None,
    *,
    num_samples: int | None = None,
    values: Mapping[str, ArrayLike] | None = None,
    density_sites: Sequence[str],
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Collect sample columns and selected log density from one isolated trace.

    Sample and deterministic sites define the output columns; auxiliary sites
    such as factors are excluded. The model's return value is ignored.
    Deterministic columns supplied in ``values`` are recomputed. All outputs
    and the summed log density have shape ``(num_samples,)``. An empty density
    selection gives zeros. The model must support this batch shape.

    Parameters
    ----------
    model
        No-argument NumPyro model declaring scalar sample sites.
    key
        Random key for free sample sites. Without a key every sample site
        must be conditioned, even under an enclosing seed handler.
    num_samples
        Sample count, static under JIT. Inferred from ``values`` if omitted;
        must agree with their length when both are supplied.
    values
        Columns of shape ``(N,)`` to condition on. Partial conditioning is
        allowed when a key is supplied.
    density_sites
        Sample sites whose log probabilities are summed.

    Raises
    ------
    ValueError
        If the sample count is missing or columns have incompatible shapes.
    KeyError
        If a sample site is unconditioned and no key is supplied.
    """
    columns: dict[str, ArrayLike] = (
        {}
        if values is None
        else {name: jnp.asarray(value) for name, value in values.items()}
    )
    if columns:
        count = _num_samples(columns)
        if num_samples is not None and num_samples != count:
            raise ValueError("num_samples must match the sample column length")
        num_samples = count
    if num_samples is None or num_samples <= 0:
        raise ValueError("num_samples must be positive or inferred from values")

    def plated() -> None:
        with numpyro.plate("batch", num_samples):
            model()

    conditioned = handlers.condition(plated, data=columns)
    execution = (
        _RequireObserved(conditioned)
        if key is None
        else handlers.seed(conditioned, rng_seed=key)
    )
    with handlers.block():
        trace = handlers.trace(execution).get_trace()
    return _outputs_and_logprob(trace, num_samples, density_sites)


def sample_sources_qmc(
    model: Callable[[], None],
    key: jax.Array,
    *,
    num_samples: int,
) -> dict[str, jax.Array]:
    """Draw sample columns on a scrambled Sobol net.

    The quasi-random counterpart of :func:`sample_and_logprob`: ``key`` seeds an
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
        Number of samples.

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
    samples = [
        name
        for name, site in probe.items()
        if site["type"] == "sample"
        and not site.get("infer", {}).get("is_auxiliary", False)
    ]
    free = [name for name in samples if not probe[name]["is_observed"]]
    scramble = int(jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
    points = qmc.Sobol(len(free), scramble=True, seed=scramble).random_base2(exponent)

    # Transform sites before the plate expands their distributions: NumPyro's
    # ExpandedDistribution does not implement icdf.
    with handlers.block(), numpyro.plate("batch", num_samples):
        trace = handlers.trace(InverseCDF(model, points=points)).get_trace()
    outputs, _ = _outputs_and_logprob(trace, num_samples, ())
    return outputs
