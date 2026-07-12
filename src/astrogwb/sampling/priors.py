"""Materialize NumPyro prior distributions from serializable config specs.

This module is the bridge between the validated, JSON-serializable prior tables
in :mod:`astrogwb.config.mcmc` (plain dicts like ``{"type": "uniform",
"low": 0.0, "high": 1.0}``) and live ``numpyro.distributions`` objects handed
to :func:`astrogwb.sampling.numpyro_model.numpyro_model`.

NumPyro/JAX are imported lazily inside :func:`build_prior` so that importing
this module does not itself initialize the JAX backend; callers must run their
runtime configuration (see :func:`astrogwb.runtime.configure_runtime`) before
invoking it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpyro.distributions import Distribution


def build_prior(spec: dict[str, Any]) -> Distribution:
    """Materialize a single prior spec into a ``numpyro`` distribution.

    Supported ``type`` values:

    - ``uniform`` (keys: ``low``, ``high``)
    - ``normal``  (keys: ``loc``, ``scale``)

    Parameters
    ----------
    spec:
        Prior spec mapping, e.g. ``{"type": "uniform", "low": 0.0, "high": 1.0}``.

    Returns
    -------
    numpyro.distributions.Distribution
        The constructed prior distribution.
    """
    import numpyro.distributions as dist

    kind = str(spec["type"]).lower()
    if kind == "uniform":
        return dist.Uniform(low=float(spec["low"]), high=float(spec["high"]))
    if kind == "normal":
        return dist.Normal(loc=float(spec["loc"]), scale=float(spec["scale"]))
    raise ValueError(f"unsupported prior type: {spec['type']!r}")
