"""Derive a merger-rate function from the redshift law a source model declares.

A source model that draws its redshift from a
:class:`~astrogwb.distributions.redshift.base.RedshiftDistribution` has already
determined its own total merger rate: the rate *is* the normalization of the
table that distribution builds, and
:meth:`~astrogwb.distributions.redshift.base.RedshiftDistribution.total_merger_rate`
reads it straight off. Registering a separate rate model for such a population
restates that density a second time, in a second registry, with nothing
comparing the two -- exactly the drift
:mod:`astrogwb.populations.registry` warns about.

:func:`infer_merger_rate_fn` closes that gap. It probes the model once, reads
the ``redshift`` site's distribution off the trace, and returns a
:data:`~astrogwb.populations.MergerRateFn` that rebuilds an equivalent
distribution at whatever hyperparameters it is later called with. The probe is
one eager execution; every subsequent call costs one
``RedshiftDistribution`` construction and no model execution at all -- the same
cost as the registered rate it replaces.

Not every source model has a rate to read. The guard-mixture proposals declare
``redshift`` with a :class:`~numpyro.distributions.MixtureGeneral`, whose
normalization is a proposal density rather than a physical rate; there is
nothing to infer, and :func:`infer_merger_rate_fn` says so rather than guessing
at a component. Those populations exist to draw importance proposals, which
need no rate, and the catalogs drawn from them pair with the *physical*
population's registered rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
from jax.typing import ArrayLike
from numpyro import handlers

from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.populations.registry import REDSHIFT_SITE, MergerRateFn, SourceFn

__all__ = ["ABSOLUTE_RATE_PARAMETER", "infer_merger_rate_fn", "require_absolute_rate"]

#: The hyperparameter carrying a rate shape's absolute normalization, in
#: :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`. A source *density* normalizes it
#: away, so the shapes this package ships default it to ``1.0``; a *rate* cannot,
#: which is what :func:`require_absolute_rate` enforces.
ABSOLUTE_RATE_PARAMETER = "local_merger_rate"

#: The probe seed. The drawn redshift is discarded -- only the site's
#: distribution is read -- so the value cannot affect the result.
_PROBE_SEED = 0


def require_absolute_rate(params: Mapping[str, ArrayLike], *, label: str) -> None:
    """Raise unless ``params`` carries the absolute rate normalization.

    ``label`` names the caller in the message. Required explicitly rather than
    left to a rate shape's own ``params.get(..., 1.0)`` default: a merger-rate
    model silently normalizing to an implicit 1.0 has no meaningful use, so a
    missing physical rate fails at the model that owns it rather than
    propagating a placeholder into a spectrum.
    """
    if ABSOLUTE_RATE_PARAMETER not in params:
        raise ValueError(
            f"{label} requires params[{ABSOLUTE_RATE_PARAMETER!r}] "
            "(Gpc^-3 yr^-1): a merger-rate model has no meaningful default rate"
        )


def probe_redshift_distribution(
    source_model: SourceFn, params: Mapping[str, ArrayLike]
) -> RedshiftDistribution:
    """The ``redshift`` site's distribution, read off one isolated execution.

    ``handlers.block`` keeps the probe's sites out of any enclosing trace, so
    this is safe to call from inside another model; ``handlers.seed`` supplies
    the key the draw needs, and the drawn value is discarded.

    Raises:
        ValueError: If the model declares no ``redshift`` sample site.
        TypeError: If that site's distribution is not a
            :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution`,
            and so carries no total merger rate.
    """
    with handlers.block(), handlers.seed(rng_seed=_PROBE_SEED):
        trace = handlers.trace(source_model).get_trace(params)

    site = trace.get(REDSHIFT_SITE)
    if site is None or site["type"] != "sample":
        raise ValueError(
            f"source model declares no {REDSHIFT_SITE!r} sample site; every "
            "source model must draw a redshift"
        )

    distribution = site["fn"]
    if not isinstance(distribution, RedshiftDistribution):
        raise TypeError(
            f"the {REDSHIFT_SITE!r} site of this source model is a "
            f"{type(distribution).__name__}, not a RedshiftDistribution, so no "
            "total merger rate can be read off it; pass an explicit "
            "merger_rate_fn, or build one with build_merger_rate_fn. A "
            "guard-mixture redshift law is a proposal density, not a physical "
            "population, and its normalization is not a merger rate"
        )
    return distribution


def infer_merger_rate_fn(
    source_model: SourceFn, params: Mapping[str, ArrayLike]
) -> MergerRateFn:
    r"""Build the merger-rate function a source model's redshift law implies.

    ``params`` is a *probe point only*: it must be complete enough to execute
    ``source_model`` once -- masses, spins and tidal parameters included, plus
    ``xi_0``/``xi_n`` for the modified-propagation variants, the same
    requirement :func:`~astrogwb.sampling.validate_source_model` has -- but the
    returned callable does not close over it. Each call rebuilds the redshift
    distribution at the hyperparameters it is given, so the rate responds to
    ``H0``, the rate-shape parameters and
    :data:`ABSOLUTE_RATE_PARAMETER` exactly as the registered rate does.

    What is captured, once, is the *recipe*: the source-frame rate shape
    :math:`\psi` and the redshift window and grid the normalization runs on,
    read off the probed distribution. The shape is a module-level singleton, so
    every rebuild hashes identically as pytree aux data and a jit-compiled
    consumer is not retraced.

    Like :func:`~astrogwb.populations.build_merger_rate_fn`, the returned
    closure is compared and hashed **by identity**: build it once per run and
    reuse it, or a freshly built, equal rate forces a recompile.

    Raises ``ValueError`` if the model declares no ``redshift`` sample site,
    and ``TypeError`` if that site carries no total merger rate -- see
    :func:`probe_redshift_distribution`.
    """
    distribution = probe_redshift_distribution(source_model, params)
    # ``type(distribution)`` rather than ``RedshiftDistribution``: a subclass
    # rebuilds as itself. The Madau-Dickinson name is a *function* alias, so
    # what comes back here is the base class, as intended.
    rebuild = partial(
        type(distribution),
        source_frame_distribution=distribution.source_frame_distribution_fn,
        minimum_redshift=distribution.minimum_redshift,
        maximum_redshift=distribution.maximum_redshift,
        n_grid=distribution.n_grid,
    )

    def merger_rate_fn(params: Mapping[str, ArrayLike]) -> jax.Array:
        require_absolute_rate(params, label="an inferred merger-rate function")
        return rebuild(params=params).total_merger_rate()

    return merger_rate_fn
