"""Drawing a catalog's source samples from the population that will describe it.

Generation and density evaluation run the *same* declaration, which is what
makes a catalog self-describing. It also makes one detail load-bearing:
``Predictive`` executes the model under ``vmap``, one draw at a time, while
every later evaluation runs it batched over all N sources at once. Floating
point does not promise those two agree to the last bit, and a stored
``luminosity_distance`` that differs from the recomputed one in its final bit
turns the exact-zero log weight of a catalog reweighted to its own proposal
into a scatter of :math:`10^{-16}`.

So the draw takes only the *stochastic* sites from ``Predictive`` and
recomputes every derived column in one batched pass -- the identical code path
that :class:`~astrogwb.importance.estimator.SpectralDensityImportanceEstimator`
will run. The stored columns are then bit-identical to what any later
evaluation derives from the stored samples, which is what makes both the
exact-zero property and the load-time consistency check exact rather than
approximate.

Population-level deterministics are dropped here rather than stored. The total
merger rate does not vary across draws -- it is a property of the population
and the redshift window, not of the particular Monte Carlo samples -- so
``Predictive`` returns it repeated N times. Recomputing it from the serialized
model is both cheaper and correct after a redshift window is narrowed, where a
stored copy would be stale.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.infer import Predictive

from astrogwb.populations.evaluation import population_log_probs, population_sites
from astrogwb.populations.registry import PopulationModel

__all__ = ["derive_source_columns", "draw_population"]


def derive_source_columns(
    model: PopulationModel,
    params: Mapping[str, ArrayLike],
    stochastic_values: Mapping[str, ArrayLike],
) -> dict[str, jax.Array]:
    """Complete a set of stochastic draws with every per-source derived column.

    One batched execution of the model, which is the same code path every later
    density evaluation takes, so the columns it returns are bit-identical to
    what those evaluations will recompute from the stored samples.

    Population-level deterministics are dropped: they have no sample axis and
    are recomputed from the serialized model when needed, where a stored copy
    would be stale after a redshift window is narrowed.
    """
    sites = population_sites(model, params)
    values = {name: jnp.asarray(value) for name, value in stochastic_values.items()}
    _, trace = population_log_probs(model, params, values)
    derived = {name: jnp.asarray(trace[name]["value"]) for name in sites.deterministic}
    return {
        **values,
        **{name: value for name, value in derived.items() if value.ndim == 1},
    }


def draw_population(
    model: PopulationModel,
    params: Mapping[str, ArrayLike],
    *,
    num_samples: int,
    seed: int,
) -> dict[str, jax.Array]:
    """Draw ``num_samples`` sources, returning stochastic and derived columns.

    Reproducible: the same model, parameters, seed and sample count in the same
    software environment give identical arrays. ``Predictive`` allocates its
    per-draw keys with ``jax.random.split``, which is prefix-stable, so a
    smaller catalog at the same seed is a prefix of a larger one -- the
    property that makes a variable-catalog-size series nested draws rather than
    unrelated ones.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    draws = Predictive(model, num_samples=num_samples)(jax.random.PRNGKey(seed), params)
    sites = population_sites(model, params)
    stochastic = {name: jnp.asarray(draws[name]) for name in sites.stochastic}
    return derive_source_columns(model, params, stochastic)
