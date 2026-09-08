r"""Evaluating a population model at fixed source samples.

Importance sampling needs the *per-sample* source log density, one entry per
catalog source, not the scalar joint that ``numpyro.infer.util.log_density``
returns: that one sums over batch dimensions and destroys exactly the axis the
weights live on. ``numpyro.infer.util.compute_log_probs`` is the same trace
walk without that reduction, and it also handles site scaling and distribution
intermediates. It is an experimental NumPyro interface, which is why its use is
confined to this module and covered by focused tests.

Two boundaries make a population model safe to evaluate *inside* an outer
inference model:

- The whole utility call runs under ``handlers.block()``. The trace the utility
  builds internally still sees the included population sites, but enclosing
  inference handlers do not, so population sites never enter the outer joint
  density and repeated evaluations cannot raise duplicate-site errors. The
  block wraps the *call*. Blocking does not stop gradients flowing through the
  numerical result.
- Constant factors are excluded by selectively blocking the model after
  substituting source values inside that block. Hidden sites receive their
  supplied values but never reach the utility's trace or log-probability
  calculation. Every site still executes, so downstream deterministics --
  detector-frame masses, distances -- use the supplied source values.

Only stochastic source values are substituted. Substituting a stored derived
column would overwrite the value the model recomputes and defeat the
consistency check that catches a catalog whose columns no longer match its
declared population.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer.util import compute_log_probs

from astrogwb.populations.registry import PopulationModel

__all__ = [
    "LUMINOSITY_DISTANCE_SITE",
    "REDSHIFT_SITE",
    "TOTAL_MERGER_RATE_SITE",
    "PopulationSites",
    "PopulationTrace",
    "population_log_probs",
    "population_sites",
    "redshift_log_density",
    "required_deterministic",
    "select_stochastic_values",
]

#: One NumPyro trace: site name -> the site's own mapping. Values are
#: deliberately ``Any`` -- a trace entry holds a distribution, an array, a
#: scale, and more -- so consumers read ``site["value"]`` without a cast.
type PopulationTrace = Mapping[str, Mapping[str, Any]]

#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

#: Deterministic distance governing waveform amplitude, in Mpc, shape ``(N,)``.
#: Includes modified GW propagation where the population declares it.
LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

#: Deterministic observer-frame total merger rate, in mergers per second,
#: shape ``()``. A population declares it only when ``params`` carries the
#: physical rate parameters; a proposal density does not need one.
TOTAL_MERGER_RATE_SITE = "total_merger_rate"


class PopulationSites(tuple[frozenset[str], frozenset[str]]):
    """The stochastic and deterministic site names a population declares."""

    __slots__ = ()

    @property
    def stochastic(self) -> frozenset[str]:
        """Sample sites; every one must be supplied to evaluate a density."""
        return self[0]

    @property
    def deterministic(self) -> frozenset[str]:
        """Deterministic sites; recomputed from the stochastic values."""
        return self[1]


def population_sites(
    model: PopulationModel, params: Mapping[str, ArrayLike]
) -> PopulationSites:
    """Discover a population's site names with one seeded, isolated execution.

    Call outside JAX transformations. The draw itself is discarded -- only the
    names are wanted -- but the model must still be executable at ``params``,
    which is the point: a population whose sites depend on which
    hyperparameters are present is discovered here rather than at weight time.
    """
    with handlers.block(), handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(params)
    stochastic = frozenset(
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    )
    deterministic = frozenset(
        name for name, site in trace.items() if site["type"] == "deterministic"
    )
    return PopulationSites((stochastic, deterministic))


def select_stochastic_values(
    source_parameters: Mapping[str, ArrayLike],
    sites: PopulationSites,
    *,
    label: str,
) -> dict[str, jax.Array]:
    """Pick out exactly the stochastic sites, as JAX arrays.

    Derived columns a catalog also stores are dropped rather than passed
    through: substituting a stored deterministic would overwrite the value the
    model recomputes and hide a catalog that has drifted from its population.
    A missing stochastic value is an error here rather than a silent fall
    through to sampling at weight-evaluation time.
    """
    missing = sorted(sites.stochastic - set(source_parameters))
    if missing:
        raise ValueError(
            f"{label}: no stored column for stochastic population site(s) {missing}; "
            "the catalog must store every site the density is evaluated at"
        )
    return {name: jnp.asarray(source_parameters[name]) for name in sites.stochastic}


def redshift_log_density(
    model: PopulationModel,
    params: Mapping[str, ArrayLike],
    redshift: ArrayLike,
) -> jax.Array:
    """The redshift site's log density, without executing the rest of the model.

    Used as a drift fingerprint: a registered name pins the name, not the
    mathematics, so a catalog records this density at fixed probe redshifts and
    every load recomputes it. Reading the site's distribution rather than
    substituting values keeps the probe independent of which source columns a
    catalog happens to hold.
    """
    with handlers.block(), handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(params)
    site = trace.get(REDSHIFT_SITE)
    if site is None or site["type"] != "sample":
        raise ValueError(
            f"population model declares no {REDSHIFT_SITE!r} sample site; every "
            "population must draw a redshift"
        )
    return jnp.asarray(site["fn"].log_prob(jnp.asarray(redshift)))


def population_log_probs(
    model: PopulationModel,
    params: Mapping[str, ArrayLike],
    source_values: Mapping[str, ArrayLike],
    *,
    hidden_sites: frozenset[str] = frozenset(),
) -> tuple[dict[str, jax.Array], PopulationTrace]:
    """Per-site log densities and the trace, at supplied stochastic values.

    ``source_values`` must supply every stochastic site: a missing one would
    reach the sampler, and this evaluation carries no RNG key by design.
    Returns the densities *unreduced*, so each has the sample axis the weights
    need, together with the trace holding the recomputed deterministics.
    ``hidden_sites`` excludes factors from the densities and trace without
    changing their supplied values. Exclusion names are not validated here.
    """
    with handlers.block():
        bound_model = handlers.substitute(
            model,
            data={name: jnp.asarray(value) for name, value in source_values.items()},
        )
        filtered_model = handlers.block(bound_model, hide=list(hidden_sites))
        return compute_log_probs(filtered_model, (params,), {}, {}, sum_log_prob=False)


def required_deterministic(
    trace: PopulationTrace,
    name: str,
    *,
    ndim: int,
    label: str,
) -> jax.Array:
    """Read one deterministic site off a trace, checking its type and rank."""
    site = trace.get(name)
    if site is None:
        raise ValueError(
            f"{label}: population model declares no {name!r} site; it is required "
            "here. Check that the hyperparameters this model needs to declare it "
            "were supplied."
        )
    if site["type"] != "deterministic":
        raise TypeError(
            f"{label}: {name!r} must be a numpyro.deterministic site, got "
            f"{site['type']!r}"
        )
    value = jnp.asarray(site["value"])
    if value.ndim != ndim:
        raise ValueError(
            f"{label}: {name!r} must have {ndim} dimension(s), got shape {value.shape}"
        )
    return value
