"""Tests for the shared inference-input pipeline that need no catalog.

``build_model`` depends only on a validated config and a spectrum callable,
so the model-building block is exercisable without generating a catalog. The
array-consuming half lives in ``test_inference_pipeline.py`` behind the
``integration`` marker.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from config_fixtures import example_raw
from numpyro import handlers
from numpyro.infer.util import log_density

from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.inference import _fix_model_params, build_model, initial_values


def _marginalized_raw() -> dict:
    raw = example_raw()
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = ["Omega_m"]
    return raw


_MODEL_KWARGS: dict[str, Any] = {
    "observed_spectral_density": jnp.ones(2),
    "scale": jnp.ones(2),
}


def _recording_spectrum_fn(seen: list[Mapping[str, Any]]):
    """A ``SpectralDensityFn`` spy: records the params it was evaluated at.

    It publishes ``total_merger_rate`` because that is the one diagnostic
    ``build_model`` renames on the marginalized path, and nothing else: no
    catalog, no power array, no weights. Staying catalog-free here is exactly
    the property these cases exist to pin.
    """

    def spectral_density_fn(params):
        seen.append(dict(params))
        return jnp.ones(2), {"total_merger_rate": jnp.array(3.0)}

    return spectral_density_fn


def test_initial_values_are_the_sampled_parameter_fiducials() -> None:
    config = build_run_config(example_raw())

    assert initial_values(config) == {
        name: config.fiducials[name] for name in config.sampled_params
    }
    assert set(initial_values(config)).isdisjoint(config.fixed_params)


def test_fix_model_params_excludes_fixed_prior_density_and_trace_site() -> None:
    def model():
        sampled = jnp.asarray(numpyro.sample("sampled", dist.Normal(0.0, 1.0)))
        fixed = jnp.asarray(numpyro.sample("fixed", dist.Normal(10.0, 0.1)))
        numpyro.factor("likelihood", -((sampled - fixed) ** 2))

    fixed_value = jnp.array(2.0)
    wrapped = _fix_model_params(model, {"fixed": fixed_value})
    actual, trace = log_density(
        wrapped,
        (),
        {},
        {"sampled": jnp.array(0.25)},
    )
    expected = dist.Normal(0.0, 1.0).log_prob(0.25) - (0.25 - fixed_value) ** 2

    np.testing.assert_allclose(float(actual), float(expected))
    assert "fixed" not in trace


def test_build_model_default_likelihood_conditions_every_fixed_param() -> None:
    config = build_run_config(example_raw())
    seen: list[Mapping[str, Any]] = []

    model, marginalization = build_model(
        config,
        spectral_density_fn=_recording_spectrum_fn(seen),
    )
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(**_MODEL_KWARGS)

    assert marginalization is None
    assert set(seen[0]) == set(config.priors)
    # Nothing renames the rate on the default path: it is the physical one.
    np.testing.assert_allclose(float(trace["total_merger_rate"]["value"]), 3.0)
    for name, value in config.fixed_params.items():
        assert name not in trace
        np.testing.assert_allclose(float(seen[0][name]), value)
    assert set(config.sampled_params) <= set(trace)


def test_build_model_amplitude_marginalized_conditions_other_fixed_params() -> None:
    config = build_run_config(_marginalized_raw())
    seen: list[Mapping[str, Any]] = []

    model, marginalization = build_model(
        config,
        spectral_density_fn=_recording_spectrum_fn(seen),
    )
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(**_MODEL_KWARGS)

    assert marginalization is not None
    assert set(seen[0]) == set(config.priors)
    # The spectrum was evaluated with H0 pinned, so its rate is the template
    # rate; publishing it as `total_merger_rate` would be indistinguishable
    # from the physical rate `amplitude_reconstruction_model` later writes.
    assert "total_merger_rate" not in trace
    np.testing.assert_allclose(float(trace["template_merger_rate"]["value"]), 3.0)
    assert "H0" not in trace
    np.testing.assert_allclose(seen[0]["H0"], config.fiducials["H0"])
    for name, value in config.fixed_params.items():
        if name == "H0":
            continue
        assert name not in trace
        np.testing.assert_allclose(float(seen[0][name]), value)
    assert set(config.sampled_params) <= set(trace)
