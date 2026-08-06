"""JAX-touching glue between :class:`~astrogwb_paper.config.mcmc.RunConfig`
and astrogwb's amplitude marginalization.

Builds the fixed quadrature grid for an amplitude-marginalized run from a
``RunConfig``. Imported only from inside functions, the same discipline as
:func:`astrogwb_paper.priors.build_prior`, so importing this module does not
itself initialize the JAX backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    from astrogwb.sampling.amplitude import AmplitudeQuadrature
    from numpyro.distributions import Distribution

    from astrogwb_paper.config.mcmc import RunConfig


def amplitude_grid(
    prior: Distribution, *, num_nodes: int, span_sigma: float
) -> jax.Array:
    """Build a grid covering the prior's support.

    ``Uniform`` priors are gridded on their exact ``[low, high]`` bounds;
    ``Normal`` priors are gridded on ``loc +/- span_sigma * scale``. The grid
    must cover the prior support because :func:`~astrogwb.sampling.amplitude.log_trapezoid`
    integrates the prior over exactly this grid -- narrowing it truncates the
    prior. At the default ``span_sigma=10.0`` the lost Normal tail mass is of
    order ``1e-23``, a constant offset identical for every posterior draw, so
    it does not perturb NUTS.
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist

    if isinstance(prior, dist.Uniform):
        low, high = float(prior.low), float(prior.high)
    elif isinstance(prior, dist.Normal):
        loc, scale = float(prior.loc), float(prior.scale)
        low, high = loc - span_sigma * scale, loc + span_sigma * scale
    else:
        raise TypeError(
            f"unsupported amplitude prior type: {type(prior).__name__}; "
            "amplitude marginalization supports uniform and normal priors"
        )
    return jnp.linspace(low, high, num_nodes)


def build_amplitude_quadrature(config: RunConfig) -> AmplitudeQuadrature:
    """Build the fixed quadrature grid for a marginalized-likelihood config.

    Requires ``config.analysis.amplitude_parameter`` and
    ``config.amplitude_prior`` to be set (i.e. ``config.analysis.likelihood
    == "amplitude_marginalized"``); see
    :class:`~astrogwb_paper.config.mcmc.AnalysisConfig`.
    """
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        amplitude_scalings,
    )
    from astrogwb.sampling.amplitude import make_amplitude_quadrature

    from astrogwb_paper.priors import build_prior

    analysis = config.analysis
    parameter = analysis.amplitude_parameter
    if parameter is None or config.amplitude_prior is None:
        raise ValueError(
            "build_amplitude_quadrature requires an amplitude-marginalized "
            "config (analysis.likelihood == 'amplitude_marginalized')"
        )

    prior = build_prior(config.amplitude_prior)
    grid = amplitude_grid(
        prior,
        num_nodes=analysis.amplitude_num_nodes,
        span_sigma=analysis.amplitude_prior_span_sigma,
    )
    import jax.numpy as jnp

    log_prior = jnp.asarray(prior.log_prob(grid))
    scalings = amplitude_scalings(parameter, config.fiducials[parameter])
    return make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        merger_rate_amplitude=scalings.merger_rate,
        mean_energy_flux_amplitude=scalings.mean_energy_flux,
    )
