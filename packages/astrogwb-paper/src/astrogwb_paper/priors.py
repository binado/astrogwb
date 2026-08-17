"""Thin compatibility shim over the config module's native prior field types.

``RunConfig.priors`` now holds live ``numpyro`` distributions directly --
see :data:`~astrogwb_paper.config.mcmc.PriorDistribution` for the validating /
serializing ``Annotated`` type. This module keeps
:func:`build_prior` for callers (notebooks) that hold raw spec mappings and
want to materialize them ad hoc. NumPyro/JAX are imported lazily inside
:func:`~astrogwb_paper.config.mcmc.materialize_prior`, so importing this
module does not itself initialize the JAX backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from astrogwb_paper.config.mcmc import materialize_prior

if TYPE_CHECKING:
    from numpyro.distributions import Distribution


def build_prior(spec: Mapping[str, Any] | Distribution) -> Distribution:
    """Materialize a single prior spec into a ``numpyro`` distribution.

    Supported ``type`` values:

    - ``uniform`` (keys: ``low``, ``high``)
    - ``normal``  (keys: ``loc``, ``scale``)

    Parameters
    ----------
    spec:
        A raw mapping such as ``{"type": "uniform", "low": 0.0, "high": 1.0}``
        read straight from a config file (the form notebooks hold), or an
        already-built distribution, passed through unchanged. Unsupported
        ``type`` values raise a ``ValueError``.

    Returns
    -------
    numpyro.distributions.Distribution
        The constructed prior distribution.
    """
    return materialize_prior(spec)
