"""Tests for the shared inference-input pipeline that need no catalog.

``build_model`` depends only on a validated config and the importance-weight
closure, so the model-building block -- previously duplicated verbatim between
``cli/run_mcmc.py`` and ``cli/profile_model.py``, and covered by nothing -- is
exercisable here. The array-consuming half lives in
``test_inference_pipeline.py`` behind the ``integration`` marker.
"""

from __future__ import annotations

from astrogwb.sampling.models import (
    amplitude_marginalized_model,
    spectral_density_model,
)
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.inference import build_model, initial_values
from config_fixtures import example_raw


def _marginalized_raw() -> dict:
    raw = example_raw()
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = []
    return raw


def _weights_fn(*args, **kwargs):
    """Stand-in for make_merger_rate_and_log_weights_fn; never called here."""
    raise AssertionError("build_model must not evaluate the weights closure")


def test_initial_values_are_the_sampled_parameter_fiducials() -> None:
    config = build_run_config(example_raw())

    assert initial_values(config) == {
        name: config.fiducials[name] for name in config.sampled_params
    }
    # Constants are pinned by the model, not initialized by NUTS.
    assert set(initial_values(config)).isdisjoint(config.constants)


def test_build_model_default_likelihood() -> None:
    config = build_run_config(example_raw())

    model, marginalization = build_model(
        config, merger_rate_and_log_weights_fn=_weights_fn
    )

    assert marginalization is None
    assert model.func is spectral_density_model
    assert model.keywords["observation_time"] == config.observation_time
    assert model.keywords["average_mode"] == "analytic_inclination"
    assert model.keywords["merger_rate_and_log_weights_fn"] is _weights_fn
    assert model.keywords["constants"] == config.constants
    assert tuple(model.keywords["priors"]) == config.sampled_params


def test_build_model_amplitude_marginalized() -> None:
    config = build_run_config(_marginalized_raw())

    model, marginalization = build_model(
        config, merger_rate_and_log_weights_fn=_weights_fn
    )

    assert marginalization is not None
    assert model.func is amplitude_marginalized_model
    assert model.keywords["amplitude_parameter"] == "H0"
    assert model.keywords["amplitude_fn"] is marginalization.amplitude_fn
    assert model.keywords["amplitude_prior"] is marginalization.prior
    assert model.keywords["amplitude_grid"] is marginalization.grid
    # The marginalized parameter has a prior but no NUTS latent, so it must
    # not reach the model's `priors` projection.
    assert "H0" not in model.keywords["priors"]
    assert "H0" in config.priors
