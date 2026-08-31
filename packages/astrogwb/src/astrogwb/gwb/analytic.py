r"""Analytic Newtonian-inspiral stochastic-background spectrum.

This module evaluates the one-sided strain spectral density implied by
Eqs. 2, 3, and 9 of Cousins et al., *The Stochastic Siren* (2026), for an
ordered joint source-frame component-mass population ``p(m1, m2)`` with
``m1 >= m2``. Integration is performed in total mass ``M = m1 + m2`` and mass
ratio ``q = m2 / m1``. The mass-ratio integral is evaluated first on a fixed
total-mass grid, then cumulatively integrated in ``M``. Linear interpolation of
that cumulative makes the source-frequency cutoff cheap to evaluate and allows
the mass calculation to be precomputed independently of redshift and cosmology.
A uniform ordered component-mass density is a closed-form special case: both
integrals are elementary, so ``uniform_prior_mass_moments`` skips the grid and
the interpolation entirely.

All numerical orders and grid sizes are static and JAX-friendly. Double the
relevant order or refine the interpolation grid until the result is stable to
the accuracy required by the application. Realistic strain spectra require JAX
x64 mode because their values underflow in float32.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.cosmology import (
    MPC_IN_METERS,
    SPEED_OF_LIGHT,
    hubble_constant_si,
    normalized_hubble_parameter,
)
from astrogwb.utils import (
    SECONDS_PER_YEAR,
    cumulative_trapezoid,
    mapped_gauss_legendre_rule,
    require_x64,
)

_GRAVITATIONAL_CONSTANT_SI: float = 6.67430e-11
_SOLAR_MASS_KG: float = 1.988409870698051e30
_GPC_IN_METERS: float = 1.0e3 * MPC_IN_METERS

# Converts a rate density in Gpc^-3 yr^-1 and a total-mass moment in
# solar-mass^(5/3) into the SI coefficient of S_h. Combining these scales
# before entering JAX avoids separately materializing ~1e50 and ~1e-84 terms.
_ASTROPHYSICAL_STRAIN_COEFFICIENT: float = (
    2.0
    * (_GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG) ** (5.0 / 3.0)
    / (
        3.0
        * math.pi ** (1.0 / 3.0)
        * SPEED_OF_LIGHT**2
        * _GPC_IN_METERS**3
        * SECONDS_PER_YEAR
    )
)

ISCO_ALPHA: float = SPEED_OF_LIGHT**3 / (
    6.0 ** (3.0 / 2.0) * math.pi * _GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG
)
"""Schwarzschild-ISCO cutoff coefficient in Hz solar-mass."""


class PopulationFunction(Protocol):
    """Vectorized one-dimensional population callback.

    Callback arguments are positional, so implementations may name their
    parameters freely.
    """

    def __call__(
        self, redshift: jax.Array, hyperparameters: Mapping[str, ArrayLike], /
    ) -> jax.Array: ...


class JointMassFunction(Protocol):
    """Vectorized ordered joint component-mass probability-density callback.

    Callback arguments are positional, so implementations may name their
    parameters freely.
    """

    def __call__(
        self,
        mass_1: jax.Array,
        mass_2: jax.Array,
        hyperparameters: Mapping[str, ArrayLike],
        /,
    ) -> jax.Array: ...


def _cumulative_mass_moment_grid(
    hyperparameters: Mapping[str, ArrayLike],
    joint_mass_prior_fn: JointMassFunction,
    *,
    component_mass_min: float,
    component_mass_max: float,
    mass_ratio_quadrature_order: int,
    n_interp_grid: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Tabulate the cumulative source-frame chirp-mass moment in total mass.

    The ordered component masses are written in total mass and mass ratio,

    .. math::

        m_1 = \frac{M}{1+q}, \qquad m_2 = \frac{qM}{1+q}.

    Multiplying :math:`\mathcal{M}^{5/3}` by the coordinate Jacobian gives the
    fixed-total-mass integrand

    .. math::

        g(M,q;\theta) = p(m_1,m_2\mid\theta)
            \frac{qM^{8/3}}{(1+q)^4}.

    For component masses in :math:`[m_{\min},m_{\max}]`, the allowed mass-ratio
    interval at fixed ``M`` is :math:`[q_-(M),1]`, where

    .. math::

        q_-(M) =
        \begin{cases}
        m_{\min}/(M-m_{\min}), & 2m_{\min}\le M\le m_{\min}+m_{\max},\\
        M/m_{\max}-1, & m_{\min}+m_{\max}\le M\le2m_{\max}.
        \end{cases}

    The mass-ratio integral is evaluated first,

    .. math::

        H(M;\theta)=\int_{q_-(M)}^1 g(M,q;\theta)\,dq,

    and sampled on a uniform total-mass grid. A cumulative trapezoid gives

    .. math::

        C(u;\theta)=\int_{2m_{\min}}^{
        \operatorname{clip}(u,2m_{\min},2m_{\max})}H(M;\theta)\,dM.

    The odd grid size places the support transition
    :math:`M=m_{\min}+m_{\max}` exactly at the central node.
    """
    total_mass = jnp.linspace(
        2.0 * component_mass_min,
        2.0 * component_mass_max,
        n_interp_grid,
        dtype=jnp.float64,
    )
    transition = component_mass_min + component_mass_max
    mass_ratio_lower = jnp.where(
        total_mass <= transition,
        component_mass_min / (total_mass - component_mass_min),
        total_mass / component_mass_max - 1.0,
    )

    mass_ratio, mass_ratio_weights = mapped_gauss_legendre_rule(
        mass_ratio_quadrature_order, mass_ratio_lower, 1.0
    )

    total_mass_grid = total_mass[..., None]
    one_plus_mass_ratio = 1.0 + mass_ratio
    mass_1 = total_mass_grid / one_plus_mass_ratio
    mass_2 = total_mass_grid * mass_ratio / one_plus_mass_ratio
    joint_mass_density = jnp.broadcast_to(
        jnp.asarray(
            joint_mass_prior_fn(mass_1, mass_2, hyperparameters),
            dtype=jnp.float64,
        ),
        mass_ratio.shape,
    )
    fixed_total_mass_integrand = (
        joint_mass_density
        * mass_ratio
        * total_mass_grid ** (8.0 / 3.0)
        / one_plus_mass_ratio**4
    )
    mass_ratio_integral = jnp.sum(
        mass_ratio_weights * fixed_total_mass_integrand,
        axis=-1,
    )
    return total_mass, cumulative_trapezoid(mass_ratio_integral, total_mass)


def _uniform_phi(total_mass: ArrayLike, component_mass: float) -> jax.Array:
    r"""Evaluate the shared total-mass antiderivative.

    .. math::

        \phi(M,x)=\frac{M^{11/3}}{44}-\frac{3x^2}{10}M^{5/3}
            +\frac{x^3}{2}M^{2/3}.
    """
    total_mass = jnp.asarray(total_mass, dtype=jnp.float64)
    return (
        total_mass ** (11.0 / 3.0) / 44.0
        - (3.0 / 10.0) * component_mass**2 * total_mass ** (5.0 / 3.0)
        + 0.5 * component_mass**3 * total_mass ** (2.0 / 3.0)
    )


def _uniform_cumulative_mass_moment(
    total_mass: ArrayLike,
    *,
    component_mass_min: float,
    component_mass_max: float,
) -> jax.Array:
    r"""Evaluate the exact cumulative mass moment of a uniform mass prior.

    For a density that is uniform on the ordered component-mass triangle,

    .. math::

        p(m_1,m_2)=\frac{2}{(m_{\max}-m_{\min})^2},
        \qquad m_{\min}\le m_2\le m_1\le m_{\max},

    the fixed-total-mass integrand of :func:`_cumulative_mass_moment_grid`
    reduces to :math:`pM^{8/3}q(1+q)^{-4}`, whose mass-ratio antiderivative is
    elementary. Substituting the two branches of :math:`q_-(M)`, for which
    :math:`(1+q_-)^{-1}` is :math:`1-m_{\min}/M` below the support transition
    :math:`T=m_{\min}+m_{\max}` and :math:`m_{\max}/M` above it, leaves only
    powers of ``M``, so the total-mass integral closes as well. Both branches
    share the antiderivative :math:`\phi` of :func:`_uniform_phi`, entering as
    :math:`+\phi(M,m_{\min})` below :math:`T` and :math:`-\phi(M,m_{\max})`
    above it. Writing :math:`\tilde M=\operatorname{clip}(M,2m_{\min},
    2m_{\max})`,

    .. math::

        C(M)/p=
        \begin{cases}
        \phi(\tilde M,m_{\min})-\phi(2m_{\min},m_{\min}),
            & \tilde M\le T,\\
        \phi(T,m_{\min})-\phi(2m_{\min},m_{\min})
            +\phi(T,m_{\max})-\phi(\tilde M,m_{\max}),
            & \tilde M\ge T.
        \end{cases}

    This is the exact counterpart of the array tabulated by
    :func:`_cumulative_mass_moment_grid`, evaluated pointwise at arbitrary
    ``total_mass`` rather than looked up between grid nodes. Clipping before
    the branch selection keeps both branches finite wherever they are traced,
    so reverse-mode gradients stay free of ``NaN`` and an infinite cutoff maps
    onto the full mass moment.
    """
    lower = 2.0 * component_mass_min
    transition = component_mass_min + component_mass_max
    total_mass = jnp.asarray(total_mass, dtype=jnp.float64)
    clipped = jnp.clip(total_mass, lower, 2.0 * component_mass_max)
    origin = _uniform_phi(lower, component_mass_min)
    below_transition = _uniform_phi(clipped, component_mass_min) - origin
    above_transition = (
        _uniform_phi(transition, component_mass_min)
        - origin
        + _uniform_phi(transition, component_mass_max)
        - _uniform_phi(clipped, component_mass_max)
    )
    density = 2.0 / (component_mass_max - component_mass_min) ** 2
    moment = density * jnp.where(
        clipped <= transition, below_transition, above_transition
    )
    # Below the support the cancellation is phi(2 m_min) - phi(2 m_min), which
    # XLA is free to reassociate into a nonzero rounding residual, so the empty
    # integral is pinned here rather than left to floating point.
    return jnp.where(total_mass <= lower, 0.0, moment)


@require_x64
def uniform_prior_mass_moments(
    frequencies: jax.Array,
    *,
    z_min: float,
    z_max: float,
    component_mass_min: float,
    component_mass_max: float,
    alpha: float = ISCO_ALPHA,
    redshift_quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate exact mass moments for a uniform component-mass prior.

    This is the closed-form counterpart of
    :func:`precompute_cumulative_mass_moments` for a density that is uniform on
    the ordered component-mass triangle. Both the mass-ratio and the total-mass
    integrals are elementary there, so the cumulative moment is evaluated
    directly at every cutoff,

    .. math::

        U_{fa}=\frac{\alpha}{f_f(1+z_a)}, \qquad M_{fa}=C(U_{fa}),

    with no mass-ratio quadrature, no total-mass grid, and no interpolation.
    The result is exact to floating-point round-off and carries none of the
    ``n_interp_grid`` discretization error of the generic path.

    The returned array is accepted unchanged by
    :func:`analytic_spectral_density_from_mass_moments`; pass the same
    ``redshift_quadrature_order`` to both so the redshift nodes agree.

    Parameters
    ----------
    frequencies:
        One-dimensional observer-frame frequency array in Hz. Values must be
        finite and strictly positive.
    z_min, z_max:
        Redshift integration bounds.
    component_mass_min, component_mass_max:
        Shared source-frame component-mass bounds in solar masses. They also fix
        the normalized density ``2 / (component_mass_max - component_mass_min)**2``.
    alpha:
        Source-frame cutoff coefficient in Hz solar-mass, defining
        ``f_max = alpha / M``. Use ``inf`` for no cutoff.
    redshift_quadrature_order:
        Gauss-Legendre order of the redshift nodes.

    Returns
    -------
    jax.Array
        Cumulative mass moments of shape ``(frequency, redshift_node)``. Values
        are exactly zero where the cutoff falls below ``2 * component_mass_min``
        and exactly the full mass moment where it lies above
        ``2 * component_mass_max``.
    """
    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    redshift, _ = mapped_gauss_legendre_rule(redshift_quadrature_order, z_min, z_max)
    total_mass_upper = alpha / (frequencies[:, None] * (1.0 + redshift[None, :]))
    return _uniform_cumulative_mass_moment(
        total_mass_upper,
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
    )


@require_x64
def precompute_cumulative_mass_moments(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, ArrayLike],
    joint_mass_prior_fn: JointMassFunction,
    *,
    z_min: float,
    z_max: float,
    component_mass_min: float,
    component_mass_max: float,
    alpha: float = ISCO_ALPHA,
    mass_ratio_quadrature_order: int = 64,
    n_interp_grid: int = 2049,
    redshift_quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate fixed mass moments at every frequency-redshift cutoff.

    For redshift quadrature nodes :math:`z_a`, this function constructs

    .. math::

        U_{fa}=\frac{\alpha}{f_f(1+z_a)}, \qquad M_{fa}=C(U_{fa}),

    and linearly interpolates :math:`C(U_{fa})` from a uniform total-mass grid.
    The result has shape ``(frequency, redshift_node)`` and can be reused while
    sampling cosmology or the redshift distribution with a fixed mass
    population. Values below the physical total-mass support are exactly zero;
    values above it are exactly the full mass moment.
    """
    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    redshift, _ = mapped_gauss_legendre_rule(redshift_quadrature_order, z_min, z_max)
    total_mass_upper = alpha / (frequencies[:, None] * (1.0 + redshift[None, :]))
    total_mass, cumulative_mass_moment = _cumulative_mass_moment_grid(
        hyperparameters,
        joint_mass_prior_fn,
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        mass_ratio_quadrature_order=mass_ratio_quadrature_order,
        n_interp_grid=n_interp_grid,
    )
    return jnp.interp(total_mass_upper, total_mass, cumulative_mass_moment)


@require_x64
def analytic_spectral_density_from_mass_moments(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, ArrayLike],
    merger_rate_fn: PopulationFunction,
    cumulative_mass_moments: jax.Array,
    *,
    z_min: float,
    z_max: float,
    redshift_quadrature_order: int = 64,
) -> jax.Array:
    r"""Evaluate the strain PSD from fixed cumulative mass moments.

    ``cumulative_mass_moments[f, a]`` must have been prepared at the same
    frequency ``f`` and redshift quadrature node ``z_a`` used here. The spectrum
    is the matrix-vector contraction

    .. math::

        S_h(f_f;\theta)=\frac{C_{\rm astro}}{H_0} f_f^{-7/3}
        \sum_a w_a\frac{R(z_a;\theta)}
        {E(z_a;\Omega_m)(1+z_a)^{4/3}}M_{fa}.

    This stage contains no mass-population evaluation or cumulative lookup, so
    a fixed ``(F, Z)`` array can be reused throughout an MCMC over cosmology or
    the redshift distribution.
    """
    frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
    cumulative_mass_moments = jnp.asarray(cumulative_mass_moments, dtype=jnp.float64)
    redshift, redshift_weights = mapped_gauss_legendre_rule(
        redshift_quadrature_order, z_min, z_max
    )
    merger_rate = jnp.broadcast_to(
        jnp.asarray(merger_rate_fn(redshift, hyperparameters), dtype=jnp.float64),
        redshift.shape,
    )
    expansion = normalized_hubble_parameter(
        redshift, jnp.asarray(hyperparameters["Omega_m"], dtype=jnp.float64)
    )
    redshift_integrand = merger_rate / (expansion * (1.0 + redshift) ** (4.0 / 3.0))
    redshift_mass_moments = cumulative_mass_moments @ (
        redshift_weights * redshift_integrand
    )

    coefficient = _ASTROPHYSICAL_STRAIN_COEFFICIENT / hubble_constant_si(
        jnp.asarray(hyperparameters["H0"], dtype=jnp.float64)
    )
    return coefficient * frequencies ** (-7.0 / 3.0) * redshift_mass_moments


@require_x64
def analytic_spectral_density(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, ArrayLike],
    merger_rate_fn: PopulationFunction,
    joint_mass_prior_fn: JointMassFunction,
    *,
    z_min: float,
    z_max: float,
    component_mass_min: float,
    component_mass_max: float,
    alpha: float = ISCO_ALPHA,
    quadrature_order: int = 64,
    n_interp_grid: int = 2049,
) -> jax.Array:
    r"""Evaluate the analytic inspiral-only one-sided strain PSD ``S_h(f)``.

    This convenience API performs both stages in one call: it tabulates the
    cumulative mass moment, interpolates it at the frequency-redshift cutoffs,
    and contracts those fixed mass moments with the redshift-dependent merger
    rate and cosmology. When the mass population is fixed during MCMC, use
    :func:`precompute_cumulative_mass_moments` and
    :func:`analytic_spectral_density_from_mass_moments` separately instead.

    Parameters
    ----------
    frequencies:
        One-dimensional observer-frame frequency array in Hz. Values must be
        finite and strictly positive.
    hyperparameters:
        Population/cosmology parameters supplied unchanged to every callback.
        ``H0`` (km s^-1 Mpc^-1) and ``Omega_m`` are required.
    merger_rate_fn:
        ``fn(redshift, hyperparameters)`` returning the absolute source-frame
        merger-rate density in Gpc^-3 yr^-1.
    joint_mass_prior_fn:
        ``fn(mass_1, mass_2, hyperparameters)`` returning a normalized ordered
        density with respect to ``dmass_1 dmass_2``.
    z_min, z_max:
        Redshift integration bounds.
    component_mass_min, component_mass_max:
        Shared source-frame component-mass bounds in solar masses.
    alpha:
        Source-frame cutoff coefficient in Hz solar-mass, defining
        ``f_max = alpha / M``. Use ``inf`` for no cutoff.
    quadrature_order:
        Shared Gauss-Legendre order for redshift and mass ratio.
    n_interp_grid:
        Odd number of uniform total-mass interpolation nodes. The default gives
        2048 intervals; use 1025, 2049, and 4097 for grid-refinement checks.

    Returns
    -------
    jax.Array
        One-sided strain spectral density in Hz^-1, with the same shape as
        ``frequencies``.
    """
    cumulative_mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        hyperparameters,
        joint_mass_prior_fn,
        z_min=z_min,
        z_max=z_max,
        component_mass_min=component_mass_min,
        component_mass_max=component_mass_max,
        alpha=alpha,
        mass_ratio_quadrature_order=quadrature_order,
        n_interp_grid=n_interp_grid,
        redshift_quadrature_order=quadrature_order,
    )
    return analytic_spectral_density_from_mass_moments(
        frequencies,
        hyperparameters,
        merger_rate_fn,
        cumulative_mass_moments,
        z_min=z_min,
        z_max=z_max,
        redshift_quadrature_order=quadrature_order,
    )
