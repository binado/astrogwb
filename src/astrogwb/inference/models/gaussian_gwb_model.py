r"""The NumPyro shell around a pure likelihood.

NumPyro keeps the priors and the constraining transforms; the likelihood is any
:data:`~astrogwb.inference.protocol.LogLikelihood`, such as an
:class:`~astrogwb.inference.likelihood.ImportanceGaussianLikelihood` bound to a
reference catalog outside inference::

    likelihood = ImportanceGaussianLikelihood.from_catalog(
        reference,
        metadata,
        population=target,
        frequencies=frequencies,
        num_redshift_nodes=32,
        density_sites=("source_frame_mass_1", "source_frame_mass_2"),
        observed=observed,
        network=Network(gaussian_bin_scale(effective_psd, observation_time, frequencies)),
    )
    model = partial(gwb_likelihood_model, likelihood=likelihood, priors=priors)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpyro
import numpyro.distributions as dist

from astrogwb.inference.protocol import LogLikelihood

__all__ = ["gwb_likelihood_model"]


def gwb_likelihood_model(
    *,
    likelihood: LogLikelihood,
    priors: Mapping[str, dist.Distribution],
) -> None:
    """Sample the priors, then add the likelihood as one factor.

    The sites are the priors, the likelihood's ``extras`` as deterministics, and
    a ``log_likelihood`` factor. Extra names must not collide with priors. Use
    ``priors={}`` for likelihood-only evaluation.
    """
    params = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    value, extras = likelihood(params)
    for name, extra in extras.items():
        numpyro.deterministic(name, extra)
    numpyro.factor("log_likelihood", value)
