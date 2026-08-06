"""JAX-touching glue between :class:`~astrogwb_paper.config.mcmc.RunConfig`
and astrogwb's amplitude marginalization.

Imported only from inside functions, the same discipline as
:func:`astrogwb_paper.priors.build_prior`, so importing this module does not
itself initialize the JAX backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def draw_amplitude_posterior(
    posterior: Any,
    *,
    quadrature: AmplitudeQuadrature,
    rng_key: jax.Array,
    chunk_size: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """Draw one :math:`\\varphi` per posterior sample, chunked to bound memory.

    ``draw_marginalized_parameter`` materializes an ``(..., K)`` array over
    its entire input; for a full chain x draw posterior against a grid with
    enough nodes to resolve a narrow conditional posterior (see
    :func:`~astrogwb_paper.config.mcmc.AnalysisConfig.amplitude_num_nodes`),
    that is a multi-GB intermediate. This loops over flattened ``(chain,
    draw)`` elements in blocks of ``chunk_size`` and concatenates, bounding
    the intermediate to ``chunk_size * K`` regardless of the total posterior
    size.

    ``rng_key`` is split into one subkey per posterior sample *before*
    chunking, so which chunk a sample falls into never changes its subkey:
    the result for a given ``rng_key`` is identical for every choice of
    ``chunk_size`` (this is what makes ``chunk_size`` a pure memory/speed
    knob rather than part of the result), verified in
    ``test_amplitude_config.py``.

    Parameters
    ----------
    posterior:
        ArviZ ``InferenceData.posterior`` group (an ``xarray.Dataset``)
        carrying ``amplitude_mle`` and ``template_optimal_snr``, both shape
        ``(chain, draw)``.
    quadrature:
        The grid :func:`build_amplitude_quadrature` built for this config.
    rng_key:
        PRNG key; one uniform draw is consumed per posterior sample.
    chunk_size:
        Number of flattened ``(chain, draw)`` elements to draw per block.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(phi, effective_nodes)``, both shaped like ``amplitude_mle``.
    """
    import jax
    import jax.numpy as jnp
    from astrogwb.sampling.amplitude import (
        draw_marginalized_parameter,
        quadrature_effective_nodes,
    )

    amplitude_mle = jnp.asarray(posterior["amplitude_mle"].values)
    template_optimal_snr = jnp.asarray(posterior["template_optimal_snr"].values)
    shape = amplitude_mle.shape
    flat_mle = amplitude_mle.reshape(-1)
    flat_snr = template_optimal_snr.reshape(-1)
    n = flat_mle.shape[0]
    subkeys = jax.random.split(rng_key, n)

    draw_one = jax.vmap(
        lambda mle, snr, key: draw_marginalized_parameter(
            mle, snr, quadrature=quadrature, rng_key=key
        )
    )

    phi_chunks = []
    nodes_chunks = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        mle_chunk = flat_mle[start:stop]
        snr_chunk = flat_snr[start:stop]
        phi_chunks.append(draw_one(mle_chunk, snr_chunk, subkeys[start:stop]))
        nodes_chunks.append(
            quadrature_effective_nodes(mle_chunk, snr_chunk, quadrature=quadrature)
        )
    phi = jnp.concatenate(phi_chunks).reshape(shape)
    effective_nodes = jnp.concatenate(nodes_chunks).reshape(shape)
    return phi, effective_nodes
