r"""A hand-written restatement of the BNS Madau-Dickinson population's physics.

This used to be production code, kept as "the single source of truth for the
density formula" and shared by the proposal and the target so the two could
never drift. That job now belongs to one NumPyro model declaration, evaluated
by both sides -- so keeping the formula in the library would be the very
duplication the consolidation removed.

It survives *here*, as a test oracle, because that is what it was always
actually worth: an independent restatement the class-based path is checked
against. An expected value produced by the code under test proves nothing.

Written to match ``InterpolatedDistribution.log_prob``'s operation order
deliberately -- one ``log`` of the ratio rather than a difference of two logs.
That is what lets the comparison be bit-exact instead of merely close, and the
same order is what keeps a catalog that is its own proposal at exactly zero log
weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import distance_and_volume_grid
from astrogwb.distributions.rates import madau_dickinson_rate

__all__ = ["reference_merger_rate_distance_and_logprob"]


def reference_merger_rate_distance_and_logprob(
    params: Mapping[str, Any],
    redshift: ArrayLike,
    *,
    redshift_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Total merger rate, luminosity distance, and redshift log-pdf.

    Builds the cosmology and density tables on ``redshift_grid``, then evaluates
    :math:`\log p(z \mid \theta)` at ``redshift``. The density
    :math:`p(z) \propto \psi(z)/(1+z) \times \mathrm{d}V_c/\mathrm{d}z` is
    normalized by its trapezoidal integral; interpolating the *complete
    unnormalized* density and dividing by that integral makes the interpolant's
    own integral exactly equal the normalization. Samples outside
    ``redshift_grid`` interpolate to zero density, hence ``-inf``.

    ``params`` must include ``H0``, ``Omega_m``, ``gamma``, ``kappa``,
    ``z_peak`` and ``local_merger_rate`` (in ``Gpc^-3 yr^-1``). Returns
    ``(total_merger_rate, luminosity_distance, logpdf)``; the rate is in
    mergers per second and the other two have the shape of ``redshift``.
    """
    redshift = jnp.asarray(redshift)

    luminosity_distance_grid, dvc_dz_grid = distance_and_volume_grid(
        redshift_grid,
        hubble_constant=params["H0"],
        omega_m=params["Omega_m"],
    )
    rate_grid = madau_dickinson_rate(
        redshift_grid,
        params["gamma"],
        params["kappa"],
        params["z_peak"],
        params["local_merger_rate"],
    )
    unnormalized_pdf_grid = rate_grid / (1.0 + redshift_grid) * dvc_dz_grid
    integral_mpc3 = jnp.trapezoid(unnormalized_pdf_grid, redshift_grid)

    unnormalized_pdf = jnp.interp(
        redshift, redshift_grid, unnormalized_pdf_grid, left=0.0, right=0.0
    )
    logpdf = jnp.log(unnormalized_pdf / integral_mpc3)
    luminosity_distance = jnp.interp(
        redshift,
        redshift_grid,
        luminosity_distance_grid,
        left=luminosity_distance_grid[0],
        right=luminosity_distance_grid[-1],
    )
    total_merger_rate = 1e-9 * integral_mpc3 / SECONDS_PER_YEAR
    return total_merger_rate, luminosity_distance, logpdf
