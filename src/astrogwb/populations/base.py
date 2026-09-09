"""Explicit population contracts shared by generation and importance weighting."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.util import compute_log_probs

type PopulationTrace = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, kw_only=True)
class Population(ABC):
    """A NumPyro model with explicit density factors and source outputs.

    Construction settings and site names are immutable, hashable metadata.
    Hyperparameters and source arrays arrive as arguments, so a population can
    remain static inside a JAX-transformed estimator. Construct it once and
    reuse it; no backend work happens during construction.

    ``source_sites`` includes every sampled input needed to replay the model,
    plus selected deterministic outputs. ``density_sites`` names only the
    factors included in importance weighting; omitted factors must cancel
    between the target and proposal. Omitting a factor does not marginalize it.
    """

    density_sites: tuple[str, ...]
    source_sites: ClassVar[tuple[str, ...]]

    @abstractmethod
    def __call__(self, params: Mapping[str, ArrayLike]) -> None:
        """Declare source sites and population-level quantities with NumPyro."""

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
        return self.evaluate(params, sources)[0]

    def evaluate(
        self,
        params: Mapping[str, ArrayLike],
        sources: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, PopulationTrace]:
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
        return log_prob, trace
