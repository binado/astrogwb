"""The Madau-Dickinson (2017) rate shape, as a redshift ``Distribution``.

.. note::

   The rate shape itself lives in :mod:`astrogwb.distributions.rates`, shared
   verbatim with
   :mod:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation`,
   which remains the reference implementation of the importance-weights
   callback. The two express the *same* density by different routes, so they
   are pinned against each other in ``tests/core/test_distributions.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
from jax.typing import ArrayLike

from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.distributions.redshift.base import RedshiftDistribution


class MadauDickinsonRedshiftDistribution(RedshiftDistribution):
    r"""Redshift distribution corresponding to a Madau-Dickinson merger rate.

    ``params`` must carry ``gamma``, ``kappa`` and ``z_peak`` -- the arguments
    of :func:`~astrogwb.distributions.rates.madau_dickinson_rate` -- on top of
    the ``H0`` / ``Omega_m`` cosmology keys
    :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution` reads.
    """

    def source_frame_distribution(
        self, redshift: ArrayLike, params: Mapping[str, ArrayLike]
    ) -> jax.Array:
        return madau_dickinson_rate(
            redshift,
            params["gamma"],
            params["kappa"],
            params["z_peak"],
        )
