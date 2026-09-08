r"""BNS population with a Madau-Dickinson merger-rate density.

Two registered models live here rather than in two files because they are one
population under two propagation laws: ``bns_md_modified_propagation`` reduces
*exactly* to ``bns_md_cosmological`` at :math:`\Xi_0 = 1`. Everything except
the ``luminosity_distance`` deterministic is declared once, by
:func:`_declare_bns_madau_dickinson`, so the two can never disagree about the
source density they share.

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
- **Distance and rate come from the same execution as the density.** One model
  run supplies the ``(N,)`` source log density, the ``(N,)`` distance governing
  waveform amplitude, and the scalar observer-frame rate.

The masses are currently independent uniforms rather than an ordered pair, and
they are omitted from ``density_sites`` for importance weighting, reproducing
the cancellation the analysis relies on today. Including them enables
mass-hyperparameter inference, and at that point the ordered-pair density must
be *correct* rather than merely symmetric: ordering doubles the density on the
retained region and zeroes it elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)
from astrogwb.populations.base import Population
from astrogwb.populations.registry import register_population_model

__all__ = [
    "AMPLITUDE_PARAMETERS",
    "BNSMadauDickinson",
    "BNSMadauDickinsonModifiedPropagation",
    "BNSMadauDickinsonUniformMixture",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
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


#: Source-frame component-mass bounds, in solar masses.
SOURCE_FRAME_MASS_MINIMUM = 1.0
SOURCE_FRAME_MASS_MAXIMUM = 2.5

#: Aligned-spin bounds.
SPIN_MAGNITUDE = 0.05

#: Dimensionless tidal-deformability bounds.
TIDAL_DEFORMABILITY_MAXIMUM = 2000.0


def _declare_bns_madau_dickinson(
    params: Mapping[str, ArrayLike],
    *,
    z_min: float,
    z_max: float,
    n_grid: int,
    luminosity_distance: jax.Array,
    redshift: jax.Array,
    redshift_distribution: MadauDickinsonRedshiftDistribution,
) -> None:
    """Declare every site the two propagation variants share."""
    del z_min, z_max, n_grid

    mass_1 = numpyro.sample(
        "source_frame_mass_1",
        dist.Uniform(SOURCE_FRAME_MASS_MINIMUM, SOURCE_FRAME_MASS_MAXIMUM),
    )
    mass_2 = numpyro.sample(
        "source_frame_mass_2",
        dist.Uniform(SOURCE_FRAME_MASS_MINIMUM, SOURCE_FRAME_MASS_MAXIMUM),
    )
    numpyro.sample("spin_1z", dist.Uniform(-SPIN_MAGNITUDE, SPIN_MAGNITUDE))
    numpyro.sample("spin_2z", dist.Uniform(-SPIN_MAGNITUDE, SPIN_MAGNITUDE))
    numpyro.sample("lambda_1", dist.Uniform(0.0, TIDAL_DEFORMABILITY_MAXIMUM))
    numpyro.sample("lambda_2", dist.Uniform(0.0, TIDAL_DEFORMABILITY_MAXIMUM))

    one_plus_z = 1.0 + redshift
    numpyro.deterministic("detector_frame_mass_1", mass_1 * one_plus_z)
    numpyro.deterministic("detector_frame_mass_2", mass_2 * one_plus_z)
    numpyro.deterministic("luminosity_distance", luminosity_distance)

    # Face-on, phase- and time-aligned: the catalog pairs with
    # ``average_mode="analytic_inclination"``, which converts face-on power
    # into the inclination average analytically.
    zeros = jnp.zeros_like(redshift)
    numpyro.deterministic("inclination", zeros)
    numpyro.deterministic("coa_phase", zeros)
    numpyro.deterministic("coa_time", zeros)

    # The physical rate is optional: a proposal density needs no rate, while a
    # target or injection observation cannot be built without one. Declaring it
    # conditionally is what lets one model serve both.
    if "local_merger_rate" in params:
        numpyro.deterministic(
            "total_merger_rate",
            redshift_distribution.total_merger_rate(params["local_merger_rate"]),
        )


def _redshift(
    params: Mapping[str, ArrayLike], *, z_min: float, z_max: float, n_grid: int
) -> tuple[jax.Array, MadauDickinsonRedshiftDistribution]:
    """Draw (or accept) the redshift and return it with its distribution."""
    redshift_distribution = MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=z_min,
        maximum_redshift=z_max,
        n_grid=n_grid,
    )
    redshift = numpyro.sample("redshift", redshift_distribution)
    return jnp.asarray(redshift), redshift_distribution


@register_population_model("bns_md_cosmological")
@dataclass(frozen=True, kw_only=True)
class BNSMadauDickinson(Population):
    r"""BNS sources on a Madau-Dickinson rate, with standard GW propagation.

    ``params`` must carry ``H0``, ``Omega_m``, ``gamma``, ``kappa`` and
    ``z_peak``; ``local_merger_rate`` (in :math:`\mathrm{Gpc}^{-3}\,
    \mathrm{yr}^{-1}`) is additionally required for the ``total_merger_rate``
    deterministic. ``z_min``, ``z_max`` and ``n_grid`` describe the grid the
    cosmology integrals and the redshift normalization run on; they are
    construction settings, bound once and serialized with the catalog.
    """

    z_min: float
    z_max: float
    n_grid: int
    density_sites: tuple[str, ...] = ("redshift",)
    source_sites: ClassVar[tuple[str, ...]] = (
        "redshift",
        "source_frame_mass_1",
        "source_frame_mass_2",
        "spin_1z",
        "spin_2z",
        "lambda_1",
        "lambda_2",
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "luminosity_distance",
        "inclination",
        "coa_phase",
        "coa_time",
    )

    def __call__(self, params: Mapping[str, ArrayLike]) -> None:
        redshift, redshift_distribution = _redshift(
            params, z_min=self.z_min, z_max=self.z_max, n_grid=self.n_grid
        )
        _declare_bns_madau_dickinson(
            params,
            z_min=self.z_min,
            z_max=self.z_max,
            n_grid=self.n_grid,
            redshift=redshift,
            redshift_distribution=redshift_distribution,
            luminosity_distance=redshift_distribution.luminosity_distance(redshift),
        )


@register_population_model("bns_md_uniform_mixture")
@dataclass(frozen=True, kw_only=True)
class BNSMadauDickinsonUniformMixture(BNSMadauDickinson):
    r"""A Madau-Dickinson redshift law blended with a uniform guard component.

    A *proposal* population, not a physical one: mixing a fraction
    :math:`\epsilon` of uniform-in-redshift draws into the Madau-Dickinson
    density fattens the tails, so reweighting to a target far from the
    generating parameters keeps a usable effective sample size instead of
    collapsing onto a handful of sources.

    Everything except the redshift law is the physical BNS population, and the
    ``luminosity_distance`` deterministic comes from the Madau-Dickinson
    component's cosmology -- the mixture changes which redshifts are drawn, not
    how far away a source at a given redshift is. ``total_merger_rate`` is
    declared only if ``local_merger_rate`` is supplied, and for a guard mixture
    it usually should not be: a sampling density has no physical rate.
    """

    uniform_mixing_fraction: float

    def __call__(self, params: Mapping[str, ArrayLike]) -> None:
        if not 0.0 <= self.uniform_mixing_fraction <= 1.0:
            raise ValueError(
                f"uniform_mixing_fraction must lie in [0, 1], got {self.uniform_mixing_fraction}"
            )
        redshift_distribution = MadauDickinsonRedshiftDistribution(
            params=params,
            minimum_redshift=self.z_min,
            maximum_redshift=self.z_max,
            n_grid=self.n_grid,
        )
        mixture = dist.MixtureGeneral(
            dist.Categorical(
                probs=jnp.array(
                    [1.0 - self.uniform_mixing_fraction, self.uniform_mixing_fraction]
                )
            ),
            [redshift_distribution, dist.Uniform(self.z_min, self.z_max)],
            support=redshift_distribution.support,
        )
        redshift = jnp.asarray(numpyro.sample("redshift", mixture))
        _declare_bns_madau_dickinson(
            params,
            z_min=self.z_min,
            z_max=self.z_max,
            n_grid=self.n_grid,
            redshift=redshift,
            redshift_distribution=redshift_distribution,
            luminosity_distance=redshift_distribution.luminosity_distance(redshift),
        )


@register_population_model("bns_md_modified_propagation")
@dataclass(frozen=True, kw_only=True)
class BNSMadauDickinsonModifiedPropagation(BNSMadauDickinson):
    r"""As :class:`BNSMadauDickinson`, with a modified GW propagation distance.

    Additionally requires ``xi_0`` and ``xi_n`` in ``params``. The distance the
    model declares is the *effective* one governing waveform amplitude,

    .. math::

        d_L^{\mathrm{GW}}(z) = d_L(z)\,
            \Xi(z), \qquad
        \Xi(z) = \Xi_0 + (1 - \Xi_0)(1 + z)^{-n},

    so a catalog's stored polarization power and this distance describe the
    same signal. At :math:`\Xi_0 = 1` the ratio is identically one and this
    reduces to :class:`BNSMadauDickinson` bit-for-bit.
    """

    def __call__(self, params: Mapping[str, ArrayLike]) -> None:
        redshift, redshift_distribution = _redshift(
            params, z_min=self.z_min, z_max=self.z_max, n_grid=self.n_grid
        )
        _declare_bns_madau_dickinson(
            params,
            z_min=self.z_min,
            z_max=self.z_max,
            n_grid=self.n_grid,
            redshift=redshift,
            redshift_distribution=redshift_distribution,
            luminosity_distance=redshift_distribution.luminosity_distance(redshift)
            * jnp.exp(log_gw_em_ratio(redshift, params["xi_0"], params["xi_n"])),
        )
