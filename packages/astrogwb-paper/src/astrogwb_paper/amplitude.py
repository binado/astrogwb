"""JAX-touching glue between :class:`~astrogwb_paper.config.mcmc.RunConfig`
and astrogwb's amplitude marginalization.

Assembles everything an amplitude-marginalized run needs from a ``RunConfig``.
Imported only from inside functions, so importing this module does not itself
initialize the JAX backend. Priors arrive already materialized (``RunConfig``
carries live distributions; see
:data:`~astrogwb_paper.config.mcmc.PriorDistribution`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import jax
    from astrogwb.sampling.amplitude import AmplitudeFn, MergerRateAmplitudeFn
    from numpyro.distributions import Distribution

    from astrogwb_paper.config.mcmc import RunConfig


class AmplitudeMarginalization(NamedTuple):
    """Everything an amplitude-marginalized run needs, built once from a ``RunConfig``.

    App-side plumbing, not a core type: unlike the ``AmplitudeQuadrature`` it
    replaces, it holds *live* objects -- the prior distribution and the scaling
    callables -- so there is nothing derived in it that could go stale against
    the config it came from. The one array, ``grid``, is a quadrature scheme
    rather than a tabulation of the density.
    """

    parameter: str
    """Name of the marginalized parameter, e.g. ``"H0"``."""

    fiducial: float
    """Reference value defining the template; the amplitude is 1 here."""

    prior: Distribution
    """Prior on the marginalized parameter; also defines the conditional's support."""

    amplitude_fn: AmplitudeFn
    """Absolute total scaling :math:`f(\\varphi) = g_R(\\varphi)\\, g_F(\\varphi)`."""

    merger_rate_fn: MergerRateAmplitudeFn
    """Absolute merger-rate scaling :math:`g_R(\\varphi)`, for the reconstructed rate."""

    grid: jax.Array
    """Quadrature nodes the marginalization integral is evaluated on."""


def build_amplitude_marginalization(config: RunConfig) -> AmplitudeMarginalization:
    """Assemble the amplitude marginalization for a marginalized-likelihood config.

    Requires ``config.analysis.amplitude_parameter`` to be set and its prior to
    live in ``config.priors`` (i.e. ``config.analysis.likelihood
    == "amplitude_marginalized"``); see
    :class:`~astrogwb_paper.config.mcmc.AnalysisConfig`.
    """
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        amplitude_H0_fn,
        amplitude_local_merger_rate_fn,
        merger_rate_H0_fn,
        merger_rate_local_merger_rate_fn,
    )
    from astrogwb.sampling.amplitude import quadrature_grid

    analysis = config.analysis
    parameter = analysis.amplitude_parameter
    if parameter is None or parameter not in config.priors:
        raise ValueError(
            "build_amplitude_marginalization requires an amplitude-marginalized "
            "config (analysis.likelihood == 'amplitude_marginalized')"
        )

    if parameter == "H0":
        amplitude_fn, merger_rate_fn = amplitude_H0_fn, merger_rate_H0_fn
    elif parameter == "local_merger_rate":
        amplitude_fn, merger_rate_fn = (
            amplitude_local_merger_rate_fn,
            merger_rate_local_merger_rate_fn,
        )
    else:
        raise ValueError(f"unsupported amplitude parameter {parameter!r}")

    prior = config.priors[parameter]
    return AmplitudeMarginalization(
        parameter=parameter,
        fiducial=float(config.fiducials[parameter]),
        prior=prior,
        amplitude_fn=amplitude_fn,
        merger_rate_fn=merger_rate_fn,
        grid=quadrature_grid(
            prior,
            num_nodes=analysis.amplitude_num_nodes,
            span_sigma=analysis.amplitude_prior_span_sigma,
        ),
    )
