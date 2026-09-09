r"""Reweighting one fixed catalog to a target population, as a spectrum.

The estimator holds the prepared inference inputs directly -- the source
samples, the polarization power, the cached proposal density and the reference
distances -- rather than delegating them to a separate catalog object. There
was never a second implementation of that container, and splitting the prepared
arrays from the model they are evaluated against only made it possible to pair
the wrong two.

One model execution per evaluation supplies everything the weights need:

.. math::

    \log w_i = \log p(x_i \mid \theta) - \log q(x_i)
        - 2\left[\log d_L(z_i \mid \theta) - \log d_i^{\mathrm{ref}}\right],

where :math:`q` is the density the catalog was drawn from, cached once, and
:math:`d^{\mathrm{ref}}` is the effective distance the stored polarization
power was generated at. Power scales as :math:`d^{-2}` in amplitude, hence the
factor of two. The distance the model declares includes modified GW
propagation where the population has it, so nothing outside the model ever
applies a propagation correction.

The reference distance is the *stored* one, never a freshly interpolated
cosmology table: the power on disk corresponds to those exact distances, and
recomputing them on a different grid would bias every weight by the
interpolation difference.

Evaluate only where the proposal has support. Subtracting two negative-infinite
log densities produces ``nan``, which propagates silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.gwb.spectral import AverageMode, spectral_density
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.weights import importance_log_weights
from astrogwb.populations import (
    LUMINOSITY_DISTANCE_SITE,
    TOTAL_MERGER_RATE_SITE,
    Population,
    PopulationTrace,
)

if TYPE_CHECKING:
    from astrogwb.catalog import Catalog

__all__ = ["SpectralDensityImportanceEstimator"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralDensityImportanceEstimator:
    """A fixed Monte Carlo realization bound to the population that reweights it.

    Dynamic pytree data: ``source_parameters`` (the stored source columns, each
    shape ``(N,)``), ``polarization_power`` of shape
    ``(F, N)``, and the cached ``proposal_log_prob`` and
    ``log_reference_distance``, both shape ``(N,)``.

    Static metadata: the immutable population ``model`` and ``average_mode``
    inclination convention. Construct the model once and reuse it. Sampled
    hyperparameters arrive through ``__call__``, never through static metadata.

    The constructor performs no conversion, density evaluation, or
    value-dependent validation: JAX rebuilds instances while flattening and
    unflattening pytrees, with tracers and placeholders in the leaves. All of
    that happens once in :meth:`from_catalog`, outside JAX transformations.
    Direct construction from already-prepared arrays stays supported -- the
    proposal need not be a population at all, and no proposal merger rate or
    observation time ever enters the weights.

    ``__call__`` returns ``(spectrum, extras)`` with a fixed diagnostics key
    set, matching the spectral-density sampling protocol without depending on
    it.
    """

    source_parameters: Mapping[str, jax.Array]
    polarization_power: jax.Array
    proposal_log_prob: jax.Array
    log_reference_distance: jax.Array
    model: Population = field(metadata={"static": True})
    average_mode: AverageMode = field(metadata={"static": True})

    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        *,
        model: Population | None = None,
        average_mode: AverageMode,
        frequency_mask: ArrayLike | None = None,
    ) -> Self:
        """Prepare every fixed input, evaluating the proposal density once.

        Call outside JAX transformations. The proposal density is the catalog's
        *own* recorded population, evaluated at the parameters it was drawn at,
        so nothing has to be restated in a run config and nothing has to be
        cross-checked against it. No physical merger rate is required for that:
        a proposal is a density, not an observation.

        ``model`` defaults to the catalog's generating model, which is what
        makes a catalog reweighted to itself give exactly zero log weights. A
        different target model is the normal case -- the same sources under
        modified propagation, say -- and is resolved and bound here once, then
        reused for every later evaluation.

        ``frequency_mask`` selects the analysis band. It reaches the power and
        nothing else: masking source samples would silently truncate the
        population and change every posterior without erroring.
        """
        generating_model = catalog.get_population_model()
        generating_params = catalog.fiducials
        target_model = generating_model if model is None else model
        source_parameters = {
            name: jnp.asarray(value)
            for name, value in catalog.source_parameters.items()
        }
        proposal_log_prob = generating_model.log_prob(
            generating_params, source_parameters
        )
        target_included = set(target_model.density_sites)
        proposal_included = set(generating_model.density_sites)
        if target_included != proposal_included:
            raise ValueError(
                "target and proposal populations must include the same source "
                f"density factors; target includes {sorted(target_included)}, "
                f"proposal includes {sorted(proposal_included)}"
            )

        reference_distance = jnp.asarray(
            catalog.source_parameters[LUMINOSITY_DISTANCE_SITE]
        )
        finite_and_positive = jnp.isfinite(reference_distance) & (
            reference_distance > 0.0
        )
        if not bool(jnp.all(finite_and_positive)):
            raise ValueError(
                f"catalog {LUMINOSITY_DISTANCE_SITE!r} column must be positive and "
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
        if proposal_log_prob.shape != (num_samples,):
            raise ValueError(
                "proposal source density must have one entry per source, got shape "
                f"{proposal_log_prob.shape} for {num_samples} sources"
            )

        return cls(
            source_parameters=source_parameters,
            polarization_power=power,
            proposal_log_prob=proposal_log_prob,
            log_reference_distance=jnp.log(reference_distance),
            model=target_model,
            average_mode=average_mode,
        )

    def log_weights(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        """Per-source log importance weights at ``params``, shape ``(N,)``.

        The estimator's diagnostics deliberately publish only the relative ESS
        -- an ``(N,)`` array per sampler step is not a diagnostic -- but the
        raw weights are what the effective-sample-size figures are made of.
        Requires no merger rate: weights are a density ratio.
        """
        return self._log_weights_and_trace(params)[0]

    def _log_weights_and_trace(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, PopulationTrace]:
        """One model execution: the weights, and the trace holding its rate."""
        target_log_prob, trace = self.model.evaluate(params, self.source_parameters)
        log_distance = jnp.log(trace[LUMINOSITY_DISTANCE_SITE]["value"])
        log_weights = importance_log_weights(
            target_log_prob=target_log_prob,
            proposal_log_prob=self.proposal_log_prob,
            log_luminosity_distance=log_distance,
            log_reference_distance=self.log_reference_distance,
        )
        return log_weights, trace

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
        """Return the spectrum, total merger rate, and relative importance ESS."""
        log_weights, trace = self._log_weights_and_trace(params)
        total_merger_rate = trace[TOTAL_MERGER_RATE_SITE]["value"]
        prediction = spectral_density(
            self.polarization_power,
            jnp.exp(log_weights),
            total_merger_rate,
            average_mode=self.average_mode,
        )
        return prediction, {
            "total_merger_rate": total_merger_rate,
            "importance_relative_ess": relative_ess(log_weights),
        }
