r"""Redshift distributions whose merger rate lags a formation rate by a delay.

If binaries form at the source-frame rate :math:`\psi(z)` and merge after a
delay :math:`\tau \sim p(\tau)`, the merger rate is a convolution in lookback
time :math:`t_L`, not in redshift:

.. math::

    R_m(z) \propto \int p(\tau)\, \psi\big(z_f(t_L(z) + \tau)\big)\,\mathrm{d}\tau,

where :math:`z_f(t)` inverts :math:`t_L`. :math:`R_m` then takes the place of
:math:`\psi` in :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution`,
so the cosmology, normalization and sampler are unchanged.

The integral runs in the delay's probability variable :math:`u = F(\tau)`,

.. math::

    R_m(z) \propto \int_0^{F(\tau_{\mathrm{avail}}(z))}
        \psi\big(z_f(t_L(z) + F^{-1}(u))\big)\,\mathrm{d}u,
    \qquad
    \tau_{\mathrm{avail}}(z) = t_L(z_{\mathrm{cut}}) - t_L(z),

with an ``n_delay_nodes``-point Gauss-Legendre rule on that interval, for each
merger redshift separately.

- **The substitution** takes :math:`p(\tau)` out of the integrand, so the nodes
  crowd where the delay has its mass -- next to :math:`\tau_{\min}` for
  :math:`p \propto 1/\tau` -- with no bounds to choose. A node set tied to the
  redshift grid cannot do this: at :math:`z \approx 0` a
  :math:`\Delta z = 0.01` step is already :math:`\sim 140` Myr.
- **The moving upper limit** integrates only the delays that fit before the
  formation cut-off :math:`z_{\mathrm{cut}}`. The limit and the nodes move
  continuously with ``H0``, ``Omega_m``, :math:`z` and the delay
  hyperparameters, so the density has no jumps and autodiff sees the moving
  boundary. Masking fixed nodes at the cut-off would drop them one at a time.
- **Gauss-Legendre** converges spectrally for the smooth integrand this leaves:
  the default 48 nodes reach :math:`\sim 10^{-9}` at :math:`z = 0` (the widest
  interval) and better at higher :math:`z`, where the midpoint rule needs
  hundreds of nodes for :math:`10^{-5}`.

An FFT would buy nothing here: the direct evaluation is an
``(n_grid, n_delay_nodes)`` array, cheap under ``jit``, and it would need a
uniform time grid that both under-resolves :math:`\tau_{\min}` and forces two
interpolations.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.cosmology import lookback_time, redshift_at_lookback_time
from astrogwb.distributions.redshift.base import (
    RedshiftDistribution,
    SourceFrameDistributionFn,
)
from astrogwb.utils import mapped_gauss_legendre_rule


class TimeDelayedRedshiftDistribution(RedshiftDistribution):
    r"""Redshift distribution of mergers delayed from a formation rate.

    ``source_frame_distribution`` is read as the *formation* rate
    :math:`\psi(z)`. The delayed merger rate is rescaled so that
    :math:`R_m(0) = \psi(0)`, which keeps ``local_merger_rate`` meaning the
    merger rate today and leaves
    :meth:`~astrogwb.distributions.redshift.base.RedshiftDistribution.total_merger_rate`
    in the same units.

    Formation is cut off above ``maximum_formation_redshift``: :math:`\psi` is
    treated as zero there, and never evaluated. The cut-off enters as the
    integral's upper limit rather than a mask, so it is continuous. That window is independent of
    ``maximum_redshift``, which bounds only the *merger* redshift.

    Parameters
    ----------
    params:
        As for :class:`RedshiftDistribution`: ``H0``, ``Omega_m``, and whatever
        ``source_frame_distribution`` reads. Delay hyperparameters live on
        ``time_delay_distribution`` instead.
    source_frame_distribution:
        The formation-rate shape :math:`\psi(z)`.
    time_delay_distribution:
        Delay distribution in Gyr. Must implement ``cdf`` and ``icdf``, e.g. the
        canonical :math:`p(\tau) \propto \tau^{-1}` as
        :class:`~astrogwb.distributions.delay.PowerLawDelayDistribution` ``(-1.0,
        0.02, 13.0)``. Prefer it to ``numpyro.distributions.DoublyTruncatedPowerLaw``
        when the slope is sampled: that one's gradient in the slope blows up
        next to :math:`-1`.
    n_delay_nodes:
        Gauss-Legendre order of the delay integral.
    maximum_formation_redshift:
        Formation cut-off.
    minimum_redshift, maximum_redshift, n_grid, validate_args:
        As for :class:`RedshiftDistribution`.
    """

    # Both data fields are set before `super().__init__`, which calls
    # `_merger_rate`. The distribution is itself a pytree, so its
    # hyperparameters are traced leaves. The Gauss-Legendre order is a shape,
    # hence static aux data.
    pytree_data_fields = ("time_delay_distribution", "maximum_formation_redshift")
    pytree_aux_fields = ("n_delay_nodes",)

    def __init__(
        self,
        *,
        params: Mapping[str, ArrayLike],
        source_frame_distribution: SourceFrameDistributionFn,
        time_delay_distribution: dist.Distribution,
        n_delay_nodes: int = 48,
        maximum_formation_redshift: float = 20.0,
        minimum_redshift: float = 0.0,
        maximum_redshift: float = 10.0,
        n_grid: int = 1000,
        validate_args: bool | None = None,
    ) -> None:
        self.time_delay_distribution = time_delay_distribution
        self.n_delay_nodes = n_delay_nodes
        self.maximum_formation_redshift = jnp.asarray(maximum_formation_redshift)
        super().__init__(
            params=params,
            source_frame_distribution=source_frame_distribution,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
            validate_args=validate_args,
        )

    def _merger_rate(
        self, redshift: jax.Array, params: Mapping[str, ArrayLike]
    ) -> jax.Array:
        r"""Delayed merger rate :math:`R_m(z)`, rescaled to :math:`R_m(0) = \psi(0)`."""
        hubble_constant, omega_m = params["H0"], params["Omega_m"]
        delay = self.time_delay_distribution
        # Prepend z = 0 for the normalization: the grid need not start there.
        merger_redshift = jnp.concatenate([jnp.zeros(1, redshift.dtype), redshift])
        merger_time = lookback_time(merger_redshift, hubble_constant, omega_m)
        available_delay = (
            lookback_time(self.maximum_formation_redshift, hubble_constant, omega_m)
            - merger_time
        )
        # Clip into the support so `cdf` saturates at 0 and 1 instead of warning.
        available_probability = delay.cdf(
            jnp.clip(
                available_delay,
                getattr(delay.support, "lower_bound", -jnp.inf),
                getattr(delay.support, "upper_bound", jnp.inf),
            )
        )
        # The weights carry the Jacobian F(tau_avail) / 2, so the probability
        # mass of the available delays is built in.
        quantiles, weights = mapped_gauss_legendre_rule(
            self.n_delay_nodes,
            jnp.zeros_like(available_probability),
            available_probability,
            dtype=jnp.result_type(merger_time, float),
        )
        # Every node forms at or before the cut-off, which precedes the age of
        # the universe, so no redshift here is infinite.
        formation_redshift = redshift_at_lookback_time(
            merger_time[:, None] + delay.icdf(quantiles), hubble_constant, omega_m
        )
        rate = jnp.sum(
            weights * self._source_frame_distribution(formation_redshift, params),
            axis=-1,
        )
        local_rate = self._source_frame_distribution(jnp.zeros(()), params)
        return rate[1:] * (local_rate / rate[0])
