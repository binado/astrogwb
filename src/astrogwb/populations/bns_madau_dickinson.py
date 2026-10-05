r"""BNS population with a Madau-Dickinson-family merger-rate density.

One registered population, ``bns_madau_dickinson``, composed from registered
sub-models: a *redshift model* (:mod:`astrogwb.populations.redshift`) and a
*mass model* (:mod:`astrogwb.populations.mass`). Spins, tidal deformabilities
and the inclination choice are plain construction kwargs. The source density a
catalog records is therefore the population name, the two sub-model records and
those kwargs -- nothing is split across several registered names.

Inclination is sampled isotropically by default; ``sample_inclination=False``
omits it for explicit analytic quadrupole averaging during contraction. This
construction setting is recorded with the population, not a generator option.

Propagation is always the modified one, applied here rather than by a redshift
model: when ``params`` carries ``xi_0`` the declared distance is the *effective*
one governing waveform amplitude,

.. math::

    d_L^{\mathrm{GW}}(z) = d_L(z)\,\Xi(z), \qquad
    \Xi(z) = \Xi_0 + (1 - \Xi_0)(1 + z)^{-n},

so a catalog's stored polarization power and this distance describe the same
signal (``xi_n`` is then required). Without ``xi_0`` the standard distance is
used, which is the :math:`\Xi_0 = 1` special case.

The declaration is a NumPyro model, and that is the whole point of it:

- **Sample sites are exactly the columns a catalog stores.** Density
  evaluation substitutes stored values by name, so a site with no stored column
  would silently fall through to sampling, and a stored column with no site
  would silently drop out of the density.
- **Derived columns are ``numpyro.deterministic``.** Detector-frame masses are
  recomputed from the substituted stochastic values on every evaluation, so a
  catalog whose columns drifted from its declared population fails on load.
- **Distance comes from the same execution as the density.** One source-model
  run supplies the ``(N,)`` source log density and the ``(N,)`` distance
  governing waveform amplitude; the rate is a separate, unplated execution that
  runs only the redshift model.

The merger rate is the redshift law's own normalization, so the population
reads it off the same law that is sampled and cannot pair a redshift law with a
rate that is not its own. A redshift model without a physical rate (a guard
mixture) makes the population a proposal density with no rate at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.populations.mass import MassFn
from astrogwb.populations.redshift import RedshiftFn, RedshiftModel
from astrogwb.populations.registry import Population, register_population

__all__ = [
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_madau_dickinson",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
]


# Absolute scalings as module-level ``def``s (not closures over the fiducial)
# so they are singletons: ``AmplitudeConditional`` carries the amplitude
# function as pytree *aux* data, which JAX hashes into the jit cache key. A
# lambda (or a ``functools.partial`` over a float) is identity-hashed, so a
# fresh one per call would retrace the model on every construction. The
# consumer forms the ratio ``f(varphi)/f(varphi_fid)`` itself.
#
# The predicted spectrum factorizes as ``f = g_R * g_F``. ``local_merger_rate``
# enters only through ``total_merger_rate`` (linear; absent from ``log_weights``),
# so ``g_R = varphi``, ``g_F = 1``, ``f = varphi``. ``H0`` enters the rate via
# ``dV_c/dz ∝ h0^{-3}`` and the mean energy flux via
# ``exp(-2 log d_L) ∝ h0^2``, so ``g_R = varphi^{-3}``, ``g_F = varphi^2``,
# and ``f = varphi^{-1}``.
def merger_rate_H0_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Merger-rate scaling :math:`g_R(H_0) = H_0^{-3}`."""
    return marginalized_parameter**-3


def amplitude_H0_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Total amplitude scaling :math:`f(H_0) = H_0^{-1}` (:math:`g_R g_F`)."""
    return 1.0 / marginalized_parameter


def merger_rate_local_merger_rate_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Merger-rate scaling :math:`g_R(\\mathcal{R}_0) = \\mathcal{R}_0`."""
    return marginalized_parameter


def amplitude_local_merger_rate_fn(marginalized_parameter: jax.Array) -> jax.Array:
    """Total amplitude scaling :math:`f(\\mathcal{R}_0) = \\mathcal{R}_0`."""
    return marginalized_parameter


def _require_sample_inclination(sample_inclination: bool) -> None:
    """Keep the construction choice static and reject numeric substitutes."""
    if not isinstance(sample_inclination, bool):
        raise TypeError("sample_inclination must be a bool")


def _require_local_merger_rate(params: Mapping[str, ArrayLike]) -> None:
    """Fail by name rather than let a rate shape default the rate to 1.0."""
    if "local_merger_rate" not in params:
        raise ValueError(
            "bns_madau_dickinson requires params['local_merger_rate'] "
            "(Gpc^-3 yr^-1): a merger-rate model has no meaningful default rate"
        )


def bns_madau_dickinson(
    params: Mapping[str, ArrayLike],
    *,
    redshift: RedshiftFn,
    mass: MassFn,
    sample_inclination: bool = True,
    maximum_spin: float = 0.05,
    maximum_tidal_deformability: float = 2000.0,
) -> dict[str, jax.Array]:
    r"""Declare the BNS source population and return its stored columns.

    ``redshift`` and ``mass`` are the built sub-models. ``params`` carries
    whatever they read, plus ``xi_0`` and ``xi_n`` for modified propagation
    (omit both for standard propagation). Spins are aligned and uniform on
    ``[-maximum_spin, maximum_spin]``; tidal deformabilities are uniform on
    ``[0, maximum_tidal_deformability]``.

    ``sample_inclination`` defaults to ``True``: each source has an isotropic
    inclination in radians. ``False`` omits the site and column, selecting
    analytic quadrupole inclination averaging during spectral contraction. It
    changes the draw, not the merger rate or default importance factors.
    """
    _require_sample_inclination(sample_inclination)
    law = redshift(params)
    source_redshift = jnp.asarray(numpyro.sample("redshift", law.distribution))
    luminosity_distance = law.luminosity_distance(source_redshift)
    if "xi_0" in params:
        if "xi_n" not in params:
            raise ValueError("params['xi_0'] requires params['xi_n']")
        luminosity_distance = luminosity_distance * jnp.exp(
            log_gw_em_ratio(source_redshift, params["xi_0"], params["xi_n"])
        )

    mass_1, mass_2 = mass(params)
    spin_1z = numpyro.sample("spin_1z", dist.Uniform(-maximum_spin, maximum_spin))
    spin_2z = numpyro.sample("spin_2z", dist.Uniform(-maximum_spin, maximum_spin))
    lambda_1 = numpyro.sample(
        "lambda_1", dist.Uniform(0.0, maximum_tidal_deformability)
    )
    lambda_2 = numpyro.sample(
        "lambda_2", dist.Uniform(0.0, maximum_tidal_deformability)
    )

    one_plus_z = 1.0 + source_redshift
    detector_frame_mass_1 = numpyro.deterministic(
        "detector_frame_mass_1", mass_1 * one_plus_z
    )
    detector_frame_mass_2 = numpyro.deterministic(
        "detector_frame_mass_2", mass_2 * one_plus_z
    )
    declared_luminosity_distance = numpyro.deterministic(
        "luminosity_distance", luminosity_distance
    )

    # Phase- and time-aligned; inclination is a separate stochastic site.
    zeros = jnp.zeros_like(source_redshift)
    coa_phase = numpyro.deterministic("coa_phase", zeros)
    coa_time = numpyro.deterministic("coa_time", zeros)

    sources = {
        "redshift": source_redshift,
        "source_frame_mass_1": mass_1,
        "source_frame_mass_2": mass_2,
        "spin_1z": spin_1z,
        "spin_2z": spin_2z,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "detector_frame_mass_1": detector_frame_mass_1,
        "detector_frame_mass_2": detector_frame_mass_2,
        "luminosity_distance": declared_luminosity_distance,
        "coa_phase": coa_phase,
        "coa_time": coa_time,
    }
    if sample_inclination:
        sources["inclination"] = numpyro.sample(
            "inclination", UniformCosineDistribution(validate_args=True)
        )
    # Without the column, generators use face-on power and the contraction
    # applies the analytic quadrupole average. An explicit zero would skip it.
    return {name: jnp.asarray(values) for name, values in sources.items()}


def _total_merger_rate(
    params: Mapping[str, ArrayLike], *, redshift: RedshiftFn
) -> jax.Array:
    r"""Observer-frame total merger rate, in mergers per second.

    ``params`` must carry ``local_merger_rate`` (in
    :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`), required explicitly rather than
    left to the redshift law's own default of 1.0: a merger-rate model silently
    normalizing to an implicit 1.0 has no meaningful use.
    """
    _require_local_merger_rate(params)
    return redshift(params).distribution.total_merger_rate()  # ty: ignore[unresolved-attribute]


@register_population("bns_madau_dickinson")
def _bns_madau_dickinson_population(
    *,
    redshift: RedshiftModel,
    mass: MassFn,
    sample_inclination: bool = True,
    maximum_spin: float = 0.05,
    maximum_tidal_deformability: float = 2000.0,
) -> Population:
    """:func:`bns_madau_dickinson` with the rate that normalizes its redshift law.

    The modified propagation distance changes the *amplitude* a source at a
    given redshift arrives with, not how many merge, so it does not enter the
    rate. A redshift model without a physical rate leaves ``merger_rate_fn``
    ``None``: a catalog drawn from such a population fails by name if used as an
    observation, and importance weighting takes the *target's* rate instead.
    """
    _require_sample_inclination(sample_inclination)
    return Population(
        source_model=partial(
            bns_madau_dickinson,
            redshift=redshift.law,
            mass=mass,
            sample_inclination=sample_inclination,
            maximum_spin=maximum_spin,
            maximum_tidal_deformability=maximum_tidal_deformability,
        ),
        merger_rate_fn=(
            partial(_total_merger_rate, redshift=redshift.law)
            if redshift.has_merger_rate
            else None
        ),
    )
