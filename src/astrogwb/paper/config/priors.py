"""Prior specs: the wire format and its materialization into numpyro.

The wire format is ``{"dist": "<numpyro.distributions class name>", "kwargs":
{...}}``, validated by :class:`astrogwb.distributions.config.DistributionConfig`.

Constructing a prior only wraps Python floats and never evaluates a JAX
operation, so materializing priors leaves the XLA backend uninitialized. That is
what lets :func:`astrogwb.paper.runtime.configure_runtime` still set the host
device count afterwards; ``tests/paper/test_cli.py`` guards it with a
subprocess, since re-running ``set_host_device_count`` after a backend
initialization is a silent no-op.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpyro.distributions as dist
from pydantic import ValidationError

from astrogwb.distributions.config import DistributionConfig


def materialize_prior(value: Any) -> dist.Distribution:
    """Materialize a prior spec into a live ``numpyro`` distribution.

    Accepts a raw mapping (``{"dist": "Uniform", "kwargs": {"low": ...}}``) or
    an already-built distribution (passed through unchanged, so
    re-validation is idempotent).
    """
    if isinstance(value, dist.Distribution):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004
            f"cannot materialize a prior from {type(value).__name__!r}; expected a "
            "spec mapping or a numpyro Distribution"
        )
    try:
        spec = DistributionConfig.model_validate(dict(value))
    except ValidationError as error:
        raise ValueError(f"invalid prior spec {dict(value)!r}: {error}") from None
    return spec.build()


def prior_to_spec(prior: dist.Distribution) -> dict[str, Any]:
    """Serialize a materialized prior back to its wire-format spec.

    Inverse of :func:`materialize_prior`; see
    :meth:`astrogwb.distributions.config.DistributionConfig.from_distribution`.
    """
    return DistributionConfig.from_distribution(prior).model_dump()


__all__ = ["materialize_prior", "prior_to_spec"]
