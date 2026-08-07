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
    import numpy as np
    import xarray as xr
    from astrogwb.sampling.amplitude import AmplitudeQuadrature
    from numpyro.distributions import Distribution

    from astrogwb_paper.config.mcmc import RunConfig

AMPLITUDE_NODE_DIM = "amplitude_node"
"""Dimension name shared by every persisted quadrature array."""

QUADRATURE_FIELDS: dict[str, str] = {
    "amplitude_grid": "grid",
    "amplitude_log_prior": "log_prior",
    "amplitude_merger_rate_scaling": "merger_rate_amplitude",
    "amplitude_mean_energy_flux_scaling": "mean_energy_flux_amplitude",
}
"""Persisted ``constant_data`` name -> :class:`AmplitudeQuadrature` field.

The writer (:func:`quadrature_constant_data`) and the reader
(:func:`load_amplitude_quadrature`) both iterate this single mapping, so they
cannot drift apart as fields are added.
"""


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


def quadrature_constant_data(quadrature: AmplitudeQuadrature) -> dict[str, np.ndarray]:
    """The quadrature arrays to persist alongside a run's posterior.

    Saved into the ``constant_data`` group so re-analysis can recover the grid
    that a chain was actually marginalized against, instead of rebuilding it
    from the config and hoping every input reproduces. The reconstruction step
    (:func:`~astrogwb.sampling.models.amplitude_reconstruction_model`) is only
    exact for *the* grid the model integrated; a silently different one yields
    a wrong marginalized posterior with no visible symptom, because the
    sufficient statistics stay finite and plausible whatever grid you pair
    them with.
    """
    import numpy as np

    return {
        name: np.asarray(getattr(quadrature, field))
        for name, field in QUADRATURE_FIELDS.items()
    }


def quadrature_dims() -> dict[str, list[str]]:
    """ArviZ ``dims`` entries putting every persisted array on one shared axis."""
    return {name: [AMPLITUDE_NODE_DIM] for name in QUADRATURE_FIELDS}


def load_amplitude_quadrature(idata: xr.DataTree) -> AmplitudeQuadrature:
    """Recover the exact quadrature a saved run used.

    Prefer this over calling :func:`build_amplitude_quadrature` on the run's
    config: it reads the grid that was integrated rather than deriving one that
    merely ought to match.

    Raises
    ------
    KeyError
        If ``idata`` predates quadrature persistence, or came from a run that
        was not amplitude-marginalized.
    """
    import jax.numpy as jnp
    from astrogwb.sampling.amplitude import make_amplitude_quadrature

    if "constant_data" not in idata:
        raise KeyError(
            "InferenceData has no constant_data group; it is either not from an "
            "amplitude-marginalized run or predates quadrature persistence"
        )
    constant_data = idata["constant_data"].dataset
    missing = [name for name in QUADRATURE_FIELDS if name not in constant_data]
    if missing:
        raise KeyError(
            f"constant_data is missing persisted quadrature arrays {missing}; "
            "it is either not from an amplitude-marginalized run or predates "
            "quadrature persistence"
        )

    arrays = {
        field: jnp.asarray(constant_data[name].values)
        for name, field in QUADRATURE_FIELDS.items()
    }
    grid = arrays["grid"]
    for field in ("merger_rate_amplitude", "mean_energy_flux_amplitude"):
        if arrays[field].shape != grid.shape:
            raise ValueError(
                f"persisted {field} shape {arrays[field].shape} does not match "
                f"grid shape {grid.shape}"
            )
    # The scaling callables are only consumed at construction time, so replaying
    # the stored values through constant functions reuses the same grid
    # validation (1D, increasing, matching shapes) that the original build got.
    # They ignore their argument because the scalings are already tabulated on
    # exactly the grid being passed in.
    return make_amplitude_quadrature(
        grid=grid,
        log_prior=arrays["log_prior"],
        merger_rate_amplitude=lambda marginalized_parameter: arrays[
            "merger_rate_amplitude"
        ],
        mean_energy_flux_amplitude=lambda marginalized_parameter: arrays[
            "mean_energy_flux_amplitude"
        ],
    )
