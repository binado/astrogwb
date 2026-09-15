r"""Reweighting one fixed catalog to a target source model, as a spectrum.

:func:`build_importance_spectrum` is the entry point: give it a catalog and
the target's bound callables, and it returns a ``(spectral_density,
log_weights)`` pair -- the two functions inference needs.

Underneath, that builder is two steps, split by when they run. Preparation
runs once, outside JAX transformations: it reads the catalog's stored columns
and power, and evaluates the density the catalog was drawn from at the
parameters it was drawn at. :func:`importance_spectral_density` and
:func:`evaluate_log_weights` run per sampler step, against a target source
model and hyperparameters. The builder binds one dict of prepared arrays into
both with :func:`functools.partial`, which is what makes the invariant below
structural rather than conventional.

One target execution per evaluation supplies everything the weights need:

.. math::

    \log w_i = \log p(x_i \mid \theta) - \log q(x_i)
        - 2\left[\log d_L(z_i \mid \theta) - \log d_i^{\mathrm{ref}}\right],

where :math:`q` is the density the catalog was drawn from, cached once, and
:math:`d^{\mathrm{ref}}` is the effective distance the stored polarization
power was generated at. Power scales as :math:`d^{-2}` in amplitude, hence the
factor of two. The distance the target declares includes modified GW
propagation where the model has it, so nothing outside the model ever applies
a propagation correction.

The reference distance is the *stored* one, never a freshly interpolated
cosmology table: the power on disk corresponds to those exact distances, and
recomputing them on a different grid would bias every weight by the
interpolation difference.

Target and proposal densities must include the same factors, or the weights
are finite and wrong. ``density_sites`` therefore comes from one place -- the
catalog, through preparation -- and :func:`build_importance_spectrum` threads
that value to the target evaluation unchanged, in the one dict it splats into
both returned callables; nothing supplies a default.

Evaluate only where the proposal has support. Subtracting two negative-infinite
log densities produces ``nan``, which propagates silently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.gwb.spectral import AverageMode, spectral_density
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.weights import importance_log_weights
from astrogwb.populations.registry import MergerRateFn, SourceFn
from astrogwb.utils.sampling import evaluate_sources

if TYPE_CHECKING:
    from astrogwb.catalog import PolarizationPowerCatalog
    from astrogwb.sampling.protocol import SpectralDensityFn

__all__ = [
    "LogWeightsFn",
    "build_importance_spectrum",
    "evaluate_log_weights",
    "importance_spectral_density",
]

#: The source output naming the effective distance governing waveform
#: amplitude: a stored catalog column, and a key of every source model's
#: returned mapping.
_LUMINOSITY_DISTANCE = "luminosity_distance"


def evaluate_log_weights(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    source_parameters: Mapping[str, jax.Array],
    proposal_log_prob: jax.Array,
    log_reference_distance: jax.Array,
    density_sites: Sequence[str],
) -> jax.Array:
    """Per-source log importance weights at ``params``, shape ``(N,)``.

    One isolated target execution supplies both the selected density and the
    distance. ``density_sites`` must be the same value ``proposal_log_prob``
    was evaluated with. Requires no merger rate: weights are a density ratio.
    """
    target_log_prob, outputs = evaluate_sources(
        source_model, params, source_parameters, density_sites=density_sites
    )
    if _LUMINOSITY_DISTANCE not in outputs:
        raise KeyError(
            f"target source model must return {_LUMINOSITY_DISTANCE!r}: it is "
            "the distance governing waveform amplitude"
        )
    return importance_log_weights(
        target_log_prob=target_log_prob,
        proposal_log_prob=proposal_log_prob,
        log_luminosity_distance=jnp.log(outputs[_LUMINOSITY_DISTANCE]),
        log_reference_distance=log_reference_distance,
    )


def importance_spectral_density(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    merger_rate_fn: MergerRateFn,
    source_parameters: Mapping[str, jax.Array],
    polarization_power: jax.Array,
    proposal_log_prob: jax.Array,
    log_reference_distance: jax.Array,
    density_sites: Sequence[str],
    average_mode: AverageMode,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The importance-weighted spectrum at ``params``, with its diagnostics.

    Bind every keyword with :func:`functools.partial` to obtain a
    :class:`~astrogwb.sampling.SpectralDensityFn`.

    Returns ``(spectrum, extras)``: ``spectrum`` has shape ``(F,)``, and
    ``extras`` holds ``total_merger_rate`` (observer frame, mergers per second)
    and ``importance_relative_ess``, both shape ``()``. An ``(N,)`` array per
    sampler step is not a diagnostic; :func:`evaluate_log_weights` gives the
    raw weights.
    """
    log_weights = evaluate_log_weights(
        params,
        source_model=source_model,
        source_parameters=source_parameters,
        proposal_log_prob=proposal_log_prob,
        log_reference_distance=log_reference_distance,
        density_sites=density_sites,
    )
    total_merger_rate = jnp.reshape(jnp.asarray(merger_rate_fn(params)), ())
    prediction = spectral_density(
        polarization_power,
        jnp.exp(log_weights),
        total_merger_rate,
        average_mode=average_mode,
    )
    return prediction, {
        "total_merger_rate": total_merger_rate,
        "importance_relative_ess": relative_ess(log_weights),
    }


#: Per-source log importance weights from hyperparameters alone, shape ``(N,)``.
type LogWeightsFn = Callable[[Mapping[str, ArrayLike]], jax.Array]


def build_importance_spectrum(
    catalog: PolarizationPowerCatalog,
    *,
    source_model: SourceFn,
    merger_rate_fn: MergerRateFn,
    average_mode: AverageMode,
    frequency_mask: ArrayLike | None = None,
) -> tuple[SpectralDensityFn, LogWeightsFn]:
    """Prepare one catalog and bind it to a target, as both callables at once.

    Call outside JAX transformations. The proposal density is the catalog's
    *own* recorded source model, evaluated at the parameters it was drawn at
    with the density factors it records, so nothing has to be restated in a
    run config. No merger rate enters the preparation: a proposal is a
    density, not an observation.

    ``source_model`` and ``merger_rate_fn`` are the target's already-built
    callables -- normally :func:`~astrogwb.populations.build_source_model` and
    :func:`~astrogwb.populations.build_merger_rate_fn`, built once per run and
    reused, since the returned partials hash by identity and a fresh,
    equal-but-not-identical rebuild forces a jit recompile. ``catalog`` must
    already be restricted to the analysis redshift window.

    One preparation pass feeds both returned callables from a single keyword
    mapping, so the weights and the spectrum provably use the density factors
    the proposal was evaluated with -- a target evaluated with a different
    factor set would otherwise produce weights that are finite and wrong, with
    no error.

    ``frequency_mask`` reaches the power and nothing else: masking source
    samples would silently truncate the population.

    Returns ``(spectral_density, log_weights)``: the importance-weighted
    spectrum, ready for a sampling model, and per-source log importance
    weights bound to the same arrays and target.
    """
    source_parameters = {
        name: jnp.asarray(value) for name, value in catalog.source_parameters.items()
    }
    density_sites = tuple(catalog.density_sites)
    proposal_log_prob, _ = evaluate_sources(
        catalog.get_source_model(),
        catalog.fiducials,
        source_parameters,
        density_sites=density_sites,
    )

    if _LUMINOSITY_DISTANCE not in source_parameters:
        raise ValueError(
            f"catalog must store a {_LUMINOSITY_DISTANCE!r} column: it is the "
            "effective distance the stored polarization power was generated at"
        )
    reference_distance = source_parameters[_LUMINOSITY_DISTANCE]
    finite_and_positive = jnp.isfinite(reference_distance) & (reference_distance > 0.0)
    if not bool(jnp.all(finite_and_positive)):
        raise ValueError(
            f"catalog {_LUMINOSITY_DISTANCE!r} column must be positive and "
            "finite: it is the effective distance the stored polarization "
            "power was generated at"
        )

    power = jnp.asarray(catalog.polarization_power)
    if frequency_mask is not None:
        power = power[jnp.asarray(frequency_mask), :]
    num_samples = reference_distance.shape[0]
    if power.shape[1] != num_samples:
        raise ValueError(
            f"polarization_power has {power.shape[1]} samples but the catalog "
            f"holds {num_samples} sources"
        )

    weight_kwargs = {
        "source_model": source_model,
        "source_parameters": source_parameters,
        "proposal_log_prob": proposal_log_prob,
        "log_reference_distance": jnp.log(reference_distance),
        "density_sites": density_sites,
    }
    return (
        partial(
            importance_spectral_density,
            merger_rate_fn=merger_rate_fn,
            polarization_power=power,
            average_mode=average_mode,
            **weight_kwargs,
        ),
        partial(evaluate_log_weights, **weight_kwargs),
    )
