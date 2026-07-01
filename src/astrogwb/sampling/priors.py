"""Materialize NumPyro prior distributions from serializable config specs.

This module is the bridge between the validated, JSON-serializable prior tables
in :mod:`astrogwb.sampling.config` (plain dicts like ``{"type": "uniform",
"low": 0.0, "high": 1.0}``) and live ``numpyro.distributions`` objects handed
to :func:`astrogwb.sampling.numpyro_model.numpyro_model`.

NumPyro/JAX are imported lazily inside :func:`build_prior` so that importing
this module does not itself initialize the JAX backend; callers must run their
runtime configuration (see ``scripts.run_mcmc.configure_runtime``) before
invoking it.
"""

from __future__ import annotations

from typing import Any

from numpyro.distributions import Distribution


def build_prior(spec: dict[str, Any]) -> Distribution:
    """Materialize a single prior spec into a ``numpyro`` distribution.

    Supported ``type`` values:

    - ``uniform`` (keys: ``low``, ``high``)
    - ``normal``  (keys: ``loc``, ``scale``)
    - ``loguniform`` (keys: ``low``, ``high``)

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
    if kind == "loguniform":
        # numpyro exposes a LogUniform distribution (base distribution of a log
        # transform); fall back to a transformed Uniform if unavailable.
        low, high = float(spec["low"]), float(spec["high"])
        if hasattr(dist, "LogUniform"):
            return dist.LogUniform(low=low, high=high)
        import jax.numpy as jnp
        from numpyro.distributions.transforms import ExpTransform

        return dist.TransformedDistribution(
            dist.Uniform(low=jnp.log(low), high=jnp.log(high)),
            ExpTransform(),
        )
    raise ValueError(f"unsupported prior type: {spec['type']!r}")


def build_priors(specs: dict[str, dict[str, Any]]) -> dict[str, Distribution]:
    """Materialize a ``{name: spec}`` mapping into ``{name: distribution}``.

    Convenience wrapper over :func:`build_prior` for the common case of
    converting a whole ``[priors.*]`` config block at once.

    Parameters
    ----------
    specs:
        Mapping from parameter name to prior spec.

    Returns
    -------
    dict[str, numpyro.distributions.Distribution]
    """
    return {name: build_prior(spec) for name, spec in specs.items()}
