r"""Redshift distributions whose merger rate lags a formation rate by a delay.

If binaries form at the source-frame rate :math:`\psi(z)` and merge after a
delay :math:`\tau \sim p(\tau)`, the merger rate is a convolution in lookback
time :math:`t_L`, not in redshift:

.. math::

    R_m(z) \propto \int p(\tau)\, \psi\big(z_f(t_L(z) + \tau)\big)\,\mathrm{d}\tau,

where :math:`z_f(t)` inverts :math:`t_L`. :math:`R_m` then takes the place of
:math:`\psi` in :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution`,
so the cosmology, normalization and sampler are unchanged.

The integral runs over quantile nodes of the delay distribution,
:math:`\tau_j = F^{-1}\big((j + \tfrac12)/n\big)`, which turns it into a plain
mean over :math:`j`. Equal-probability nodes crowd where the delay has its
mass -- next to :math:`\tau_{\min}` for :math:`p \propto 1/\tau` -- which a
node set tied to the redshift grid cannot do: at :math:`z \approx 0` a
:math:`\Delta z = 0.01` step is already :math:`\sim 140` Myr. The nodes also
move smoothly with the delay hyperparameters, so the density has clean
gradients for NUTS. An FFT would buy nothing here: the direct evaluation is an
``(n_grid, n_delay_nodes)`` array, cheap under ``jit``, and it would need a
uniform time grid that still under-resolves :math:`\tau_{\min}`.
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


class TimeDelayedRedshiftDistribution(RedshiftDistribution):
    r"""Redshift distribution of mergers delayed from a formation rate.

    ``source_frame_distribution`` is read as the *formation* rate
    :math:`\psi(z)`. The delayed merger rate is rescaled so that
    :math:`R_m(0) = \psi(0)`, which keeps ``local_merger_rate`` meaning the
    merger rate today and leaves
    :meth:`~astrogwb.distributions.redshift.base.RedshiftDistribution.total_merger_rate`
    in the same units.

    Formation is cut off above ``maximum_formation_redshift``: :math:`\psi` is
    treated as zero there, and never evaluated. That window is independent of
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
        Delay distribution in Gyr. Must implement ``icdf``; see
        :class:`~astrogwb.distributions.time_delay.PowerLawTimeDelayDistribution`.
    n_delay_nodes:
        Number of quantile nodes in the delay integral.
    maximum_formation_redshift:
        Formation cut-off.
    minimum_redshift, maximum_redshift, n_grid, validate_args:
        As for :class:`RedshiftDistribution`.
    """

    # The delay enters only through its quantile nodes, so those -- not the
    # distribution object -- are the data. Both fields are set before
    # `super().__init__`, which calls `_merger_rate`.
    pytree_data_fields = ("delay_nodes", "maximum_formation_redshift")

    def __init__(
        self,
        *,
        params: Mapping[str, ArrayLike],
        source_frame_distribution: SourceFrameDistributionFn,
        time_delay_distribution: dist.Distribution,
        n_delay_nodes: int = 200,
        maximum_formation_redshift: float = 20.0,
        minimum_redshift: float = 0.0,
        maximum_redshift: float = 10.0,
        n_grid: int = 1000,
        validate_args: bool | None = None,
    ) -> None:
        quantiles = (jnp.arange(n_delay_nodes) + 0.5) / n_delay_nodes
        self.delay_nodes = time_delay_distribution.icdf(quantiles)
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
        # Prepend z = 0 for the normalization: the grid need not start there.
        merger_redshift = jnp.concatenate([jnp.zeros(1, redshift.dtype), redshift])
        formation_time = (
            lookback_time(merger_redshift, hubble_constant, omega_m)[:, None]
            + self.delay_nodes[None, :]
        )
        formed = formation_time < lookback_time(
            self.maximum_formation_redshift, hubble_constant, omega_m
        )
        # Double `where`: the cut-off entries are fed a harmless time, so
        # neither psi nor its gradient ever sees an infinite redshift.
        formation_redshift = redshift_at_lookback_time(
            jnp.where(formed, formation_time, 0.0), hubble_constant, omega_m
        )
        formation_rate = jnp.where(
            formed, self._source_frame_distribution(formation_redshift, params), 0.0
        )
        rate = jnp.mean(formation_rate, axis=-1)
        local_rate = self._source_frame_distribution(jnp.zeros(()), params)
        return rate[1:] * (local_rate / rate[0])
