"""Materialize NumPyro prior distributions from serializable config specs.

This module is the bridge between the validated prior specs in
:mod:`astrogwb_paper.config.mcmc` (:data:`~astrogwb_paper.config.mcmc.PriorSpec`)
and live ``numpyro.distributions`` objects handed to
:func:`astrogwb.sampling.models.spectral_density_model`.

NumPyro/JAX are imported lazily inside :func:`build_prior` so that importing
this module does not itself initialize the JAX backend; callers must run their
runtime configuration (see :func:`astrogwb_paper.runtime.configure_runtime`) before
invoking it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, assert_never

from pydantic import TypeAdapter

from astrogwb_paper.config.mcmc import NormalPrior, PriorSpec, UniformPrior

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

_PRIOR_ADAPTER: TypeAdapter[PriorSpec] = TypeAdapter(PriorSpec)


def build_prior(spec: PriorSpec | Mapping[str, Any]) -> Distribution:
    """Materialize a single prior spec into a ``numpyro`` distribution.

    Supported ``type`` values:

    - ``uniform`` (keys: ``low``, ``high``)
    - ``normal``  (keys: ``loc``, ``scale``)

    Parameters
    ----------
    spec:
        A validated :data:`~astrogwb_paper.config.mcmc.PriorSpec`, or a raw
        mapping such as ``{"type": "uniform", "low": 0.0, "high": 1.0}`` read
        straight from a config file. Mappings are validated before use, so an
        unsupported ``type`` raises here rather than producing a bad prior.

    Returns
    -------
    numpyro.distributions.Distribution
        The constructed prior distribution.
    """
    import numpyro.distributions as dist

    if isinstance(spec, Mapping):
        spec = _PRIOR_ADAPTER.validate_python(spec)

    match spec:
        case UniformPrior():
            return dist.Uniform(low=spec.low, high=spec.high)
        case NormalPrior():
            return dist.Normal(loc=spec.loc, scale=spec.scale)
        case _:  # pragma: no cover - exhaustive over PriorSpec
            assert_never(spec)
