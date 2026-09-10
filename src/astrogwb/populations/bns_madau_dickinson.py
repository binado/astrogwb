r"""BNS population with a Madau-Dickinson merger-rate density.

Registered models live here rather than in several files because they are one
population under two mass laws and two propagation laws.
``bns_md_modified_propagation`` reduces *exactly* to ``bns_md_cosmological`` at
:math:`\Xi_0 = 1`; the Gaussian-mass counterparts share that pair. Everything
except the mass sites and the ``luminosity_distance`` deterministic is declared
once, by :func:`_declare_bns_madau_dickinson`, so the variants cannot disagree
about the source density they share.

Each variant is registered as a *source* model: it declares sites and returns
the mapping that defines the source-output set, with no notion of a physical
rate. The rate lives in :func:`madau_dickinson_total_merger_rate`, registered
separately as the *merger-rate* model every variant here pairs with by
default (see ``register_recipe`` at the bottom). Splitting the two is what
lets a guard-mixture *redshift law* pair with the same Madau-Dickinson *rate*
the physical population uses, without executing the whole source model just
to read one scalar.

The declaration is a NumPyro model, and that is the whole point of it:

- **Sample sites are exactly the columns a catalog stores.** Density
  evaluation substitutes stored values by name, so a site with no stored column
  would silently fall through to sampling, and a stored column with no site
  would silently drop out of the density. The gwmock graph this replaces drew
  an ordered ``mass_pair`` and sliced it into the two mass columns; that
  intermediate is not stored, so it cannot be a site, and the ordered pair is
  expressed directly as the two stored names instead.
- **Derived columns are ``numpyro.deterministic``.** Detector-frame masses used
  to be computed by hand in the generation script, which meant nothing checked
  them on the way back in. Here they are recomputed from the substituted
  stochastic values on every evaluation, so a catalog whose columns drifted
  from its declared population fails on load.
- **Distance comes from the same execution as the density.** One source-model
  run supplies the ``(N,)`` source log density and the ``(N,)`` distance
  governing waveform amplitude; the rate is a separate, unplated execution.

The component masses are an ordered pair: the first mass is the larger one.
Both mass sites are included in importance weighting. Two mass laws are
registered:

- **Ordered uniforms** (``bns_md_cosmological`` and its variants). Parameters
  ``minimum_mass`` and ``mass_width``; the fiducial support is ``[1.0, 2.5]``
  solar masses. The conditional factorization has constant joint density
  ``2 / width**2`` on the ordered triangle. That triangle is compact, so a
  NUTS step that moves the edges can send catalog samples outside the
  support and drop their importance weights to zero.
- **Ordered Gaussians** (``bns_md_gaussian_cosmological`` and its variants).
  Both components are i.i.d. :math:`\mathcal{N}(\mu, \sigma^2)`, then ordered;
  the parameters are ``mass_mean`` and ``mass_sigma``. The joint density is
  ``2\,\mathcal{N}(m_1)\,\mathcal{N}(m_2)`` on the half-plane ``m_1 \ge m_2``,
  with no compact mass support. Moving :math:`(\mu, \sigma)` therefore never
  sends an importance weight to zero, which is the property NUTS needs.
  Galactic BNS masses motivate the shape (a Gaussian around
  :math:`1.33\,M_\odot` with width :math:`\sim 0.09\,M_\odot`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.mass import MaxOfTwoNormalsDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)
from astrogwb.populations.registry import (
    register_merger_rate_model,
    register_recipe,
    register_source_model,
)

__all__ = [
    "AMPLITUDE_PARAMETERS",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_md_cosmological",
    "bns_md_gaussian_cosmological",
    "bns_md_gaussian_modified_propagation",
    "bns_md_gaussian_uniform_mixture",
    "bns_md_modified_propagation",
    "bns_md_uniform_mixture",
    "madau_dickinson_total_merger_rate",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
]

AMPLITUDE_PARAMETERS: tuple[str, ...] = ("H0", "local_merger_rate")
"""Parameters this population supports marginalizing analytically."""


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


#: Aligned-spin bounds.
SPIN_MAGNITUDE = 0.05

#: Dimensionless tidal-deformability bounds.
TIDAL_DEFORMABILITY_MAXIMUM = 2000.0


def _declare_ordered_uniform_masses(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, jax.Array]:
    """Ordered pair of i.i.d. uniforms on ``[minimum_mass, minimum_mass + width]``."""
    minimum_mass: jax.Array = jnp.asarray(params["minimum_mass"])
    mass_width: jax.Array = jnp.asarray(params["mass_width"])
    # For two ordered iid uniforms, Beta(2, 1) is the primary mass marginal.
    mass_1 = numpyro.sample(
        "source_frame_mass_1",
        dist.TransformedDistribution(
            dist.Beta(2.0, 1.0, validate_args=True),
            dist.transforms.AffineTransform(minimum_mass, mass_width),
            validate_args=True,
        ),
    )
    mass_2 = numpyro.sample(
        "source_frame_mass_2",
        dist.Uniform(minimum_mass, mass_1, validate_args=True),
    )
    return jnp.asarray(mass_1), jnp.asarray(mass_2)


def _declare_ordered_gaussian_masses(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, jax.Array]:
    """Ordered pair of i.i.d. ``Normal(mass_mean, mass_sigma)`` components."""
    mass_mean: jax.Array = jnp.asarray(params["mass_mean"])
    mass_sigma: jax.Array = jnp.asarray(params["mass_sigma"])
    mass_1 = numpyro.sample(
        "source_frame_mass_1",
        MaxOfTwoNormalsDistribution(mass_mean, mass_sigma, validate_args=True),
    )
    mass_2 = numpyro.sample(
        "source_frame_mass_2",
        dist.TruncatedNormal(mass_mean, mass_sigma, high=mass_1, validate_args=True),
    )
    return jnp.asarray(mass_1), jnp.asarray(mass_2)


def _declare_bns_madau_dickinson(
    params: Mapping[str, ArrayLike],
    *,
    luminosity_distance: jax.Array,
    redshift: jax.Array,
    declare_masses: Callable[
        [Mapping[str, ArrayLike]], tuple[jax.Array, jax.Array]
    ] = _declare_ordered_uniform_masses,
) -> dict[str, jax.Array]:
    """Declare every site the propagation and mass variants share."""
    mass_1, mass_2 = declare_masses(params)
    spin_1z = numpyro.sample("spin_1z", dist.Uniform(-SPIN_MAGNITUDE, SPIN_MAGNITUDE))
    spin_2z = numpyro.sample("spin_2z", dist.Uniform(-SPIN_MAGNITUDE, SPIN_MAGNITUDE))
    lambda_1 = numpyro.sample(
        "lambda_1", dist.Uniform(0.0, TIDAL_DEFORMABILITY_MAXIMUM)
    )
    lambda_2 = numpyro.sample(
        "lambda_2", dist.Uniform(0.0, TIDAL_DEFORMABILITY_MAXIMUM)
    )

    one_plus_z = 1.0 + redshift
    detector_frame_mass_1 = numpyro.deterministic(
        "detector_frame_mass_1", mass_1 * one_plus_z
    )
    detector_frame_mass_2 = numpyro.deterministic(
        "detector_frame_mass_2", mass_2 * one_plus_z
    )
    declared_luminosity_distance = numpyro.deterministic(
        "luminosity_distance", luminosity_distance
    )

    # Face-on, phase- and time-aligned: the catalog pairs with
    # ``average_mode="analytic_inclination"``, which converts face-on power
    # into the inclination average analytically.
    zeros = jnp.zeros_like(redshift)
    inclination = numpyro.deterministic("inclination", zeros)
    coa_phase = numpyro.deterministic("coa_phase", zeros)
    coa_time = numpyro.deterministic("coa_time", zeros)

    sources = {
        "redshift": redshift,
        "source_frame_mass_1": mass_1,
        "source_frame_mass_2": mass_2,
        "spin_1z": spin_1z,
        "spin_2z": spin_2z,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "detector_frame_mass_1": detector_frame_mass_1,
        "detector_frame_mass_2": detector_frame_mass_2,
        "luminosity_distance": declared_luminosity_distance,
        "inclination": inclination,
        "coa_phase": coa_phase,
        "coa_time": coa_time,
    }
    return {name: jnp.asarray(values) for name, values in sources.items()}


def _redshift(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> tuple[jax.Array, RedshiftDistribution]:
    """Draw (or accept) the redshift and return it with its distribution."""
    redshift_distribution = MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=z_min,
        maximum_redshift=z_max,
        n_grid=n_grid,
    )
    redshift = numpyro.sample("redshift", redshift_distribution)
    return jnp.asarray(redshift), redshift_distribution


@register_merger_rate_model("madau_dickinson")
def madau_dickinson_total_merger_rate(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> jax.Array:
    r"""Observer-frame total merger rate under the Madau-Dickinson rate shape.

    ``params`` must carry ``H0``, ``Omega_m``, ``gamma``, ``kappa``, ``z_peak``
    and ``local_merger_rate`` (in :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`).
    ``local_merger_rate`` is required explicitly here rather than left to the
    underlying rate shape's own ``params.get(..., 1.0)`` default: a
    merger-rate model silently normalizing to an implicit 1.0 has no
    meaningful use, so a missing physical rate fails at the model that owns
    it rather than propagating a placeholder into a spectrum.
    """
    if "local_merger_rate" not in params:
        raise ValueError(
            "madau_dickinson_total_merger_rate requires "
            "params['local_merger_rate'] (Gpc^-3 yr^-1): a merger-rate model "
            "has no meaningful default rate"
        )
    redshift_distribution = MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=z_min,
        maximum_redshift=z_max,
        n_grid=n_grid,
    )
    return redshift_distribution.total_merger_rate()


@register_source_model("bns_md_cosmological")
def bns_md_cosmological(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> dict[str, jax.Array]:
    r"""BNS sources on a Madau-Dickinson redshift law, with standard GW propagation.

    ``params`` must carry ``H0``, ``Omega_m``, ``gamma``, ``kappa`` and
    ``z_peak``. ``z_min``, ``z_max`` and ``n_grid`` describe the grid the
    cosmology integrals and the redshift normalization run on; they are
    construction settings, bound once and serialized with the catalog. Pairs
    with :func:`madau_dickinson_total_merger_rate` by default (see the
    ``register_recipe`` call below).
    """
    redshift, redshift_distribution = _redshift(
        params, z_min=z_min, z_max=z_max, n_grid=n_grid
    )
    return _declare_bns_madau_dickinson(
        params,
        redshift=redshift,
        luminosity_distance=redshift_distribution.luminosity_distance(redshift),
    )


@register_source_model("bns_md_uniform_mixture")
def bns_md_uniform_mixture(
    params: Mapping[str, ArrayLike],
    *,
    z_min: float,
    z_max: float,
    n_grid: int,
    uniform_mixing_fraction: float,
) -> dict[str, jax.Array]:
    r"""A Madau-Dickinson redshift law blended with a uniform guard component.

    A *proposal* source model, not a physical one: mixing a fraction
    :math:`\epsilon` of uniform-in-redshift draws into the Madau-Dickinson
    density fattens the tails, so reweighting to a target far from the
    generating parameters keeps a usable effective sample size instead of
    collapsing onto a handful of sources.

    Everything except the redshift law is the physical BNS population, and the
    ``luminosity_distance`` deterministic comes from the Madau-Dickinson
    component's cosmology -- the mixture changes which redshifts are drawn, not
    how far away a source at a given redshift is.
    """
    if not 0.0 <= uniform_mixing_fraction <= 1.0:
        raise ValueError(
            f"uniform_mixing_fraction must lie in [0, 1], got {uniform_mixing_fraction}"
        )
    redshift_distribution = MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=z_min,
        maximum_redshift=z_max,
        n_grid=n_grid,
    )
    mixture = dist.MixtureGeneral(
        dist.Categorical(
            probs=jnp.array([1.0 - uniform_mixing_fraction, uniform_mixing_fraction])
        ),
        [redshift_distribution, dist.Uniform(z_min, z_max)],
        support=redshift_distribution.support,
    )
    redshift = jnp.asarray(numpyro.sample("redshift", mixture))
    return _declare_bns_madau_dickinson(
        params,
        redshift=redshift,
        luminosity_distance=redshift_distribution.luminosity_distance(redshift),
    )


@register_source_model("bns_md_modified_propagation")
def bns_md_modified_propagation(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> dict[str, jax.Array]:
    r"""As :func:`bns_md_cosmological`, with a modified GW propagation distance.

    Additionally requires ``xi_0`` and ``xi_n`` in ``params``. The distance the
    model declares is the *effective* one governing waveform amplitude,

    .. math::

        d_L^{\mathrm{GW}}(z) = d_L(z)\,
            \Xi(z), \qquad
        \Xi(z) = \Xi_0 + (1 - \Xi_0)(1 + z)^{-n},

    so a catalog's stored polarization power and this distance describe the
    same signal. At :math:`\Xi_0 = 1` the ratio is identically one and this
    reduces to :func:`bns_md_cosmological` bit-for-bit.
    """
    redshift, redshift_distribution = _redshift(
        params, z_min=z_min, z_max=z_max, n_grid=n_grid
    )
    return _declare_bns_madau_dickinson(
        params,
        redshift=redshift,
        luminosity_distance=redshift_distribution.luminosity_distance(redshift)
        * jnp.exp(log_gw_em_ratio(redshift, params["xi_0"], params["xi_n"])),
    )


@register_source_model("bns_md_gaussian_cosmological")
def bns_md_gaussian_cosmological(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> dict[str, jax.Array]:
    r"""As :func:`bns_md_cosmological`, with i.i.d. Gaussian component masses.

    ``params`` must carry ``mass_mean`` and ``mass_sigma`` instead of
    ``minimum_mass`` and ``mass_width``. Both components are drawn from
    :math:`\mathcal{N}(\mu, \sigma^2)` and ordered so the first mass is the
    larger one; the joint density ``2\,\mathcal{N}(m_1)\,\mathcal{N}(m_2)``
    has support on the half-plane ``m_1 \ge m_2``. Moving :math:`(\mu, \sigma)`
    therefore never sends an importance weight to zero, unlike the ordered
    uniform triangle whose edges are a hard constraint on the hyperparameters.
    """
    redshift, redshift_distribution = _redshift(
        params, z_min=z_min, z_max=z_max, n_grid=n_grid
    )
    return _declare_bns_madau_dickinson(
        params,
        redshift=redshift,
        luminosity_distance=redshift_distribution.luminosity_distance(redshift),
        declare_masses=_declare_ordered_gaussian_masses,
    )


@register_source_model("bns_md_gaussian_uniform_mixture")
def bns_md_gaussian_uniform_mixture(
    params: Mapping[str, ArrayLike],
    *,
    z_min: float,
    z_max: float,
    n_grid: int,
    uniform_mixing_fraction: float,
) -> dict[str, jax.Array]:
    r"""As :func:`bns_md_uniform_mixture`, with i.i.d. Gaussian component masses.

    The redshift law is the same uniform-guard mixture; only the mass sites
    differ. ``params`` must carry ``mass_mean`` and ``mass_sigma``.
    """
    if not 0.0 <= uniform_mixing_fraction <= 1.0:
        raise ValueError(
            f"uniform_mixing_fraction must lie in [0, 1], got {uniform_mixing_fraction}"
        )
    redshift_distribution = MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=z_min,
        maximum_redshift=z_max,
        n_grid=n_grid,
    )
    mixture = dist.MixtureGeneral(
        dist.Categorical(
            probs=jnp.array([1.0 - uniform_mixing_fraction, uniform_mixing_fraction])
        ),
        [redshift_distribution, dist.Uniform(z_min, z_max)],
        support=redshift_distribution.support,
    )
    redshift = jnp.asarray(numpyro.sample("redshift", mixture))
    return _declare_bns_madau_dickinson(
        params,
        redshift=redshift,
        luminosity_distance=redshift_distribution.luminosity_distance(redshift),
        declare_masses=_declare_ordered_gaussian_masses,
    )


@register_source_model("bns_md_gaussian_modified_propagation")
def bns_md_gaussian_modified_propagation(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> dict[str, jax.Array]:
    r"""As :func:`bns_md_modified_propagation`, with i.i.d. Gaussian component masses.

    ``params`` must carry ``mass_mean`` and ``mass_sigma`` as well as ``xi_0``
    and ``xi_n``. At :math:`\Xi_0 = 1` this reduces to
    :func:`bns_md_gaussian_cosmological` bit-for-bit.
    """
    redshift, redshift_distribution = _redshift(
        params, z_min=z_min, z_max=z_max, n_grid=n_grid
    )
    return _declare_bns_madau_dickinson(
        params,
        redshift=redshift,
        luminosity_distance=redshift_distribution.luminosity_distance(redshift)
        * jnp.exp(log_gw_em_ratio(redshift, params["xi_0"], params["xi_n"])),
        declare_masses=_declare_ordered_gaussian_masses,
    )


# The six previously-registered population names, kept as recipes pairing
# each source model here with the Madau-Dickinson rate above -- the same
# pairing every one of them used before the split. Catalog reconstruction,
# the notebooks, and existing call sites that name one of these six keep
# working unchanged.
register_recipe(
    "bns_md_cosmological",
    source_model="bns_md_cosmological",
    rate_model="madau_dickinson",
)
register_recipe(
    "bns_md_uniform_mixture",
    source_model="bns_md_uniform_mixture",
    rate_model="madau_dickinson",
)
register_recipe(
    "bns_md_modified_propagation",
    source_model="bns_md_modified_propagation",
    rate_model="madau_dickinson",
)
register_recipe(
    "bns_md_gaussian_cosmological",
    source_model="bns_md_gaussian_cosmological",
    rate_model="madau_dickinson",
)
register_recipe(
    "bns_md_gaussian_uniform_mixture",
    source_model="bns_md_gaussian_uniform_mixture",
    rate_model="madau_dickinson",
)
register_recipe(
    "bns_md_gaussian_modified_propagation",
    source_model="bns_md_gaussian_modified_propagation",
    rate_model="madau_dickinson",
)
