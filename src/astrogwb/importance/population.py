r"""Populations as pytrees of NumPyro distributions, and importance weights between them.

A *population* is the source-parameter model :math:`p(x \mid \theta)` at one
point :math:`\theta` of hyperparameter space: a mapping from source-parameter
name to the NumPyro distribution that parameter follows, plus the
hyperparameters themselves. Two levels of distribution are in play and they
are deliberately named apart. The hyperprior :math:`\pi(\theta)` is what
:func:`~astrogwb.sampling.models.spectral_density_model` calls ``priors``; the
mapping inside :class:`Population` is ``distributions``, conditional on the
sampled :math:`\theta`.

Importance weighting a fixed proposal catalog to a target :math:`\theta` is a
ratio of two such populations evaluated at the catalog's source parameters:

.. math::

    \log w_i = \log p_\mathrm{target}(x_i) - \log p_\mathrm{proposal}(x_i)
             - 2\left[\log d_\mathrm{GW,target}(z_i) - \log d_\mathrm{GW,proposal}(z_i)\right].

The first difference is :meth:`Population.log_prob` on each side. The second
is *not* a density ratio: the catalog's polarization power was computed at one
luminosity distance, and the spectral density scales as :math:`d_L^{-2}`, so
moving :math:`\theta` rescales every source's power. It is also asymmetric on
purpose. The target computes :math:`d_L(z; \theta)` from its own cosmology,
while the proposal reads the distance the waveforms were *actually generated
at* -- the catalog's stored ``luminosity_distance`` -- and applies the GW/EM
ratio at the fiducial propagation parameters. Recomputing the proposal
distance from a cosmology table would disagree with the generated one at
interpolation level, and every weight would carry that bias.

:class:`PopulationTerms` is what a population reduces to at a catalog: the two
``(N,)`` arrays above and the total merger rate. Reducing to explicit arrays is
what makes the proposal side cacheable. The proposal never changes during a
run, so its terms are computed once and closed over as constants. Closing over
the proposal :class:`Population` itself and calling ``log_prob`` inside the
model would not be equivalent: XLA constant-folds only below a size threshold,
so an ``(N,)`` interpolation over a large catalog would rerun every step.

``source_parameters`` is an argument to :meth:`Population.log_prob` rather
than a field. The target population is rebuilt on every sampler step and
describes the model at :math:`\theta`; the catalog is constant data owned by
the catalog. Keeping the arrays out also makes ``jax.vmap`` over
:math:`\theta` trivial. With nothing static left inside, a :class:`NamedTuple`
is enough: NumPyro distributions are already pytrees and nest cleanly.

Evaluate populations only on samples inside their support. Off-support
samples have ``-inf`` log-density on both sides, and ``-inf - (-inf)`` is
``nan``. The paper application's ``truncate_catalog_samples`` restricts a
catalog to the analysis window before anything here is called.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike


class Population(NamedTuple):
    r"""The source-parameter model :math:`p(x \mid \theta)` at one :math:`\theta`.

    Only parameters whose distribution differs between target and proposal
    need an entry: a parameter drawn from the same fixed law on both sides
    contributes zero to every weight. The invariant that matters is that the
    target and proposal populations carry the *same* key set -- a key missing
    from one side silently drops a factor from every weight.
    """

    distributions: Mapping[str, dist.Distribution]
    """Per-parameter distributions, keyed by the catalog's source-parameter names."""

    params: Mapping[str, ArrayLike]
    """The hyperparameters :math:`\\theta` this population was built at.

    Carried because not every hyperparameter parametrizes a distribution: the
    propagation parameters ``xi_0`` / ``xi_n`` enter the GW distance, and
    ``local_merger_rate`` enters the total rate, and neither belongs to any
    source-parameter law.
    """

    def log_prob(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Sum of per-parameter log-densities at ``source_parameters``.

        Every key of :attr:`distributions` must be present in
        ``source_parameters``; a missing one raises :class:`KeyError` rather
        than being skipped, since skipping would drop a factor from the weight.
        """
        log_probs = {
            name: distribution.log_prob(source_parameters[name])
            for name, distribution in self.distributions.items()
        }
        return jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))


class PopulationTerms(NamedTuple):
    """What a :class:`Population` reduces to at a catalog's source parameters."""

    log_prob: jax.Array
    """:meth:`Population.log_prob` at the catalog, shape ``(N,)``."""

    log_gw_distance: jax.Array
    """Log GW luminosity distance per source, shape ``(N,)``.

    ``log d_L,EM + log_gw_em_ratio``; the module docstring explains why the
    two sides compute it differently.
    """

    total_merger_rate: jax.Array
    """Total merger rate in mergers per second (scalar)."""


def importance_log_weights(
    target: PopulationTerms, proposal: PopulationTerms
) -> jax.Array:
    """Log importance weights of a proposal catalog under a target population.

    The one weights formula in the package; see the module docstring for its
    derivation. Identical terms give exactly zero.
    """
    log_prob_ratio = target.log_prob - proposal.log_prob
    log_distance_ratio = target.log_gw_distance - proposal.log_gw_distance
    return log_prob_ratio - 2.0 * log_distance_ratio


class PopulationFn(Protocol):
    """Build the population at sampled hyperparameters.

    The one callable a model needs: it is called inside the sampler trace with
    the values of every sampled site, and everything downstream -- log-probs,
    distances, rates -- is read off the returned :class:`Population`.
    """

    def __call__(self, params: Mapping[str, Any]) -> Population: ...


__all__ = [
    "Population",
    "PopulationFn",
    "PopulationTerms",
    "importance_log_weights",
]
