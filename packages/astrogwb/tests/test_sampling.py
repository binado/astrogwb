from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb.detector import gaussian_bin_scale
from astrogwb.sampling import (
    AmplitudeConditional,
    amplitude_marginalized_model,
    amplitude_reconstruction_model,
    spectral_density_model,
)
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

type _AmplitudePrior = dist.Normal | dist.Uniform


def _condition_without_density(model, fixed_params):
    """Supply fixed sample values without exposing their prior sites."""
    names = list(fixed_params)
    return handlers.block(
        handlers.condition(model, data=fixed_params),
        hide=names,
    )


def test_spectral_density_model_smoke_trace() -> None:
    frequencies = jnp.array([10.0, 30.0])
    polarization_power = jnp.array([[1.0, 2.0], [5.0, 6.0]])
    samples = {"mass_1": jnp.array([20.0, 30.0])}

    def merger_rate_and_log_weights_fn(params, samples):
        return jnp.array(1.0), jnp.zeros(2)

    trace = handlers.trace(
        handlers.seed(
            spectral_density_model,
            rng_seed=0,
        )
    ).get_trace(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=jnp.array([16.0, 48.0]),
        effective_psd=jnp.ones(2),
        observation_time=2.0,
        average_mode="catalog_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
    )

    assert trace["spectral_density_obs"]["fn"].event_shape == (2,)


def test_spectral_density_model_uses_combined_callback() -> None:
    frequencies = jnp.array([10.0, 20.0])
    polarization_power = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    samples = {"sentinel": jnp.array([4.0, 6.0])}

    def merger_rate_and_log_weights_fn(params, samples):
        shared = params["scale"] + samples["sentinel"][0]
        total_merger_rate = shared
        log_weights = jnp.log(jnp.array([shared, shared + 2.0]))
        return total_merger_rate, log_weights

    model = _condition_without_density(
        spectral_density_model,
        {"scale": jnp.array(2.0)},
    )
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=jnp.array([1.0, 2.0]),
        effective_psd=jnp.ones(2),
        observation_time=3.0,
        average_mode="catalog_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors={"scale": dist.Normal(10.0, 0.5)},
    )

    assert "scale" not in trace
    np.testing.assert_allclose(np.asarray(trace["total_merger_rate"]["value"]), 6.0)
    np.testing.assert_allclose(
        np.asarray(trace["importance_relative_ess"]["value"]),
        49.0 / 50.0,
    )


# --------------------------------------------------------------------------- #
# Amplitude-marginalized model
# --------------------------------------------------------------------------- #

FIDUCIAL_RATE = 2.0
"""Reference value of the marginalized parameter; the amplitude is its ratio."""

FREQUENCIES = jnp.array([10.0, 20.0, 30.0, 40.0])
OBSERVATION_TIME = 2.0
NOISE_SCALE = jnp.array([1.0, 1.4, 0.8, 1.2])

EFFECTIVE_PSD = NOISE_SCALE / gaussian_bin_scale(
    jnp.ones(4), FREQUENCIES, OBSERVATION_TIME
)
"""PSD chosen so the per-bin likelihood scale is exactly ``NOISE_SCALE``.

``gaussian_bin_scale`` is linear in the PSD, so dividing by its response to a
unit PSD inverts it. Keeping sigma of order the signal makes the amplitude
posterior broad enough to integrate on a modest numerical grid.
"""


def _linear_callback(
    params: Mapping[str, Any],
    samples: Mapping[str, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """A callback strictly linear in ``local_merger_rate``, as the real one is.

    ``tilt`` only reshapes the importance weights, so it is a genuine shape
    parameter that the amplitude cannot absorb.
    """
    total_merger_rate = jnp.asarray(params["local_merger_rate"])
    log_weights = jnp.asarray(params["tilt"]) * samples["sentinel"]
    return total_merger_rate, log_weights


_MARGINALIZED_KWARGS: dict[str, Any] = {
    "frequencies": FREQUENCIES,
    "polarization_power": jnp.array([[1.0, 2.0], [3.0, 1.5], [2.0, 4.0], [1.0, 1.0]]),
    "samples": {"sentinel": jnp.array([0.2, -0.4])},
    "observed_spectral_density": jnp.array([2.4, 4.1, 5.9, 1.8]),
    "effective_psd": EFFECTIVE_PSD,
    "observation_time": OBSERVATION_TIME,
    "average_mode": "catalog_inclination",
    "merger_rate_and_log_weights_fn": _linear_callback,
    "amplitude_parameter": "local_merger_rate",
    "fiducials": {"local_merger_rate": FIDUCIAL_RATE},
}


def _identity_scaling(marginalized_parameter: jax.Array) -> jax.Array:
    """``g_R`` for ``local_merger_rate``; ``g_F`` is 1, so this is also ``f``."""
    return marginalized_parameter


# The marginalized parameter is `local_merger_rate` itself, so the prior and
# the grid live on rates. The fiducial rate is 2.0, so this covers amplitudes
# A = rate / 2 in [0.1, 2.5].
_AMPLITUDE_PRIOR = dist.Uniform(0.2, 5.0)
_AMPLITUDE_GRID = jnp.linspace(0.2, 5.0, 2001)

_MARGINALIZATION_KWARGS: dict[str, Any] = {
    "amplitude_fn": _identity_scaling,
    "amplitude_prior": _AMPLITUDE_PRIOR,
    "amplitude_grid": _AMPLITUDE_GRID,
}
"""What the inference model needs to define the marginalized direction."""

_RECONSTRUCTION_KWARGS: dict[str, Any] = {
    "amplitude_fn": _identity_scaling,
    "merger_rate_amplitude_fn": _identity_scaling,
    "prior": _AMPLITUDE_PRIOR,
    "fiducial": FIDUCIAL_RATE,
    "grid": _AMPLITUDE_GRID,
}
"""The same direction, spelled the way the reconstruction model takes it."""


def test_amplitude_marginalized_model_registers_expected_sites() -> None:
    trace = handlers.trace(
        handlers.seed(amplitude_marginalized_model, rng_seed=0)
    ).get_trace(
        **_MARGINALIZED_KWARGS,
        **_MARGINALIZATION_KWARGS,
        priors={"tilt": dist.Normal(0.0, 1.0)},
    )

    assert "amplitude_mle" in trace
    assert "template_optimal_snr" in trace
    assert "importance_relative_ess" in trace
    assert "template_merger_rate" in trace
    np.testing.assert_allclose(
        np.asarray(trace["template_merger_rate"]["value"]), FIDUCIAL_RATE
    )
    factor_site = trace["amplitude_marginalized_log_likelihood"]
    assert isinstance(factor_site["fn"], dist.Unit)
    assert np.isfinite(float(factor_site["fn"].log_factor))
    assert "spectral_density_obs" not in trace


def test_amplitude_marginalized_model_pins_the_amplitude_to_its_fiducial() -> None:
    seen: list[Mapping[str, Any]] = []

    def recording_callback(params, samples):
        seen.append(dict(params))
        return _linear_callback(params, samples)

    handlers.trace(handlers.seed(amplitude_marginalized_model, rng_seed=0)).get_trace(
        **{
            **_MARGINALIZED_KWARGS,
            "merger_rate_and_log_weights_fn": recording_callback,
        },
        **_MARGINALIZATION_KWARGS,
        priors={"tilt": dist.Normal(0.0, 1.0)},
    )

    assert float(seen[0]["local_merger_rate"]) == FIDUCIAL_RATE


def test_amplitude_marginalized_model_rejects_a_sampled_amplitude() -> None:
    with pytest.raises(ValueError, match="cannot also be sampled"):
        handlers.seed(amplitude_marginalized_model, rng_seed=0)(
            **_MARGINALIZED_KWARGS,
            **_MARGINALIZATION_KWARGS,
            priors={
                "tilt": dist.Normal(0.0, 1.0),
                "local_merger_rate": dist.Uniform(1.0, 3.0),
            },
        )


def test_amplitude_marginalized_model_accepts_a_pre_sliced_frequency_grid() -> None:
    frequencies = jnp.array([10.0, 30.0])
    kwargs = {
        **_MARGINALIZED_KWARGS,
        "frequencies": frequencies,
        "polarization_power": jnp.array([[1.0, 2.0], [2.0, 4.0]]),
        "observed_spectral_density": jnp.array([2.4, 5.9]),
        "effective_psd": EFFECTIVE_PSD[jnp.array([True, False, True, False])],
    }

    model = _condition_without_density(
        amplitude_marginalized_model,
        {"tilt": jnp.array(0.3)},
    )
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(
        **kwargs,
        **_MARGINALIZATION_KWARGS,
        priors={"tilt": dist.Normal(0.0, 1.0)},
    )

    assert np.isfinite(float(trace["amplitude_mle"]["value"]))
    assert np.isfinite(float(trace["template_optimal_snr"]["value"]))


def _marginalization_from_rate_prior(
    prior: _AmplitudePrior, num: int = 40_001
) -> dict[str, Any]:
    """The marginalized direction induced by a prior on the rate.

    Identity scaling anchored at ``FIDUCIAL_RATE`` makes the amplitude
    ``A = rate / FIDUCIAL_RATE``, so the marginalized parameter is the rate
    itself -- exactly the variable the general model is integrated over below,
    which is what cancels the Jacobian.
    """
    if isinstance(prior, dist.Uniform):
        low, high = float(prior.low), float(prior.high)
        grid = jnp.linspace(low, high, num)
    else:
        loc, scale = float(prior.loc), float(prior.scale)
        grid = jnp.linspace(loc - 40.0 * scale, loc + 40.0 * scale, num)
    return {
        "amplitude_fn": _identity_scaling,
        "amplitude_prior": prior,
        "amplitude_grid": grid,
    }


@pytest.mark.parametrize(
    "rate_prior",
    [dist.Uniform(1.0, 3.0), dist.Normal(2.0, 0.8)],
    ids=["uniform", "normal"],
)
def test_amplitude_marginalized_model_matches_the_general_model(
    rate_prior: _AmplitudePrior,
) -> None:
    """Numerically marginalize the general model and compare the log densities.

    Both models are given the *same* prior on the same variable, the rate --
    the marginalized model states it on the marginalized parameter, the
    general model on its sample site. (Equivalently: ``rate_prior`` is the
    pushforward under ``rate = FIDUCIAL_RATE * amplitude`` of a
    ``Uniform(0.5, 1.5)`` / ``Normal(1.0, 0.4)`` prior on the amplitude.)
    Integrating the general model over ``rate`` rather than ``amplitude``
    cancels the Jacobian exactly, so the two log densities must agree without
    any leftover constant -- which is what makes this a joint check on the
    factor term, the normalizations, and the reference injection.
    """
    tilt = 0.35
    general_kwargs = {
        key: value
        for key, value in _MARGINALIZED_KWARGS.items()
        if key not in ("amplitude_parameter", "fiducials")
    }

    def general_log_density(rate: float) -> float:
        value, _ = log_density(
            spectral_density_model,
            (),
            {
                **general_kwargs,
                "priors": {
                    "local_merger_rate": rate_prior,
                    "tilt": dist.Normal(0.0, 1.0),
                },
            },
            {"local_merger_rate": jnp.asarray(rate), "tilt": jnp.asarray(tilt)},
        )
        return float(value)

    if isinstance(rate_prior, dist.Uniform):
        low, high = float(rate_prior.low), float(rate_prior.high)
    else:
        loc, scale = float(rate_prior.loc), float(rate_prior.scale)
        low, high = loc - 40.0 * scale, loc + 40.0 * scale
    rates = np.linspace(low, high, 20_001)
    densities = np.exp([general_log_density(rate) for rate in rates])
    numerical = float(np.log(np.trapezoid(densities, rates)))

    marginalized, _ = log_density(
        amplitude_marginalized_model,
        (),
        {
            **_MARGINALIZED_KWARGS,
            **_marginalization_from_rate_prior(rate_prior),
            "priors": {"tilt": dist.Normal(0.0, 1.0)},
        },
        {"tilt": jnp.asarray(tilt)},
    )

    np.testing.assert_allclose(float(marginalized), numerical, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Amplitude reconstruction model
# --------------------------------------------------------------------------- #


def _reconstruction_samples() -> dict[str, jax.Array]:
    """Statistics shaped ``(chain, draw)``, as ``mcmc.get_samples(group_by_chain=True)``."""
    return {
        "amplitude_mle": jnp.array([[0.9, 1.0, 1.1], [1.2, 0.8, 1.0]]),
        "template_optimal_snr": jnp.array([[25.0, 30.0, 35.0], [40.0, 45.0, 50.0]]),
        "template_merger_rate": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }


def _reconstruction_draws(
    statistics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    draws = Predictive(
        partial(
            amplitude_reconstruction_model,
            amplitude_parameter="local_merger_rate",
            **_RECONSTRUCTION_KWARGS,
        ),
        num_samples=1,
        return_sites=[
            "local_merger_rate",
            "total_merger_rate",
            "quadrature_effective_nodes",
        ],
    )(jax.random.key(0), **statistics)
    return {name: values[0] for name, values in draws.items()}


def test_amplitude_reconstruction_model_returns_chain_draw_sites() -> None:
    draws = _reconstruction_draws(_reconstruction_samples())

    for name in (
        "local_merger_rate",
        "total_merger_rate",
        "quadrature_effective_nodes",
    ):
        assert draws[name].shape == (2, 3), name
        assert bool(jnp.all(jnp.isfinite(draws[name]))), name

    phi = draws["local_merger_rate"]
    assert bool(jnp.all(phi >= float(_AMPLITUDE_GRID[0])))
    assert bool(jnp.all(phi <= float(_AMPLITUDE_GRID[-1])))


def test_amplitude_reconstruction_model_computes_deterministics_from_inputs() -> None:
    samples = _reconstruction_samples()
    draws = _reconstruction_draws(samples)

    # total_merger_rate must be template_merger_rate * g_R(phi) with the
    # input template_merger_rate, elementwise.
    expected_rate = (
        np.asarray(samples["template_merger_rate"])
        * np.asarray(draws["local_merger_rate"])
        / FIDUCIAL_RATE
    )
    np.testing.assert_allclose(
        np.asarray(draws["total_merger_rate"]), expected_rate, rtol=1e-6
    )

    expected_nodes = np.asarray(
        AmplitudeConditional(
            samples["amplitude_mle"],
            samples["template_optimal_snr"],
            amplitude_fn=_identity_scaling,
            prior=_AMPLITUDE_PRIOR,
            fiducial=FIDUCIAL_RATE,
            grid=_AMPLITUDE_GRID,
        ).effective_nodes
    )
    np.testing.assert_allclose(
        np.asarray(draws["quadrature_effective_nodes"]), expected_nodes, rtol=1e-6
    )


def test_amplitude_reconstruction_model_registers_only_generated_sites() -> None:
    trace = handlers.trace(
        handlers.seed(
            partial(
                amplitude_reconstruction_model,
                amplitude_parameter="local_merger_rate",
                **_RECONSTRUCTION_KWARGS,
            ),
            rng_seed=0,
        )
    ).get_trace(**_reconstruction_samples())

    for statistic in _reconstruction_samples():
        assert statistic not in trace
    assert trace["local_merger_rate"]["type"] == "sample"
    assert trace["total_merger_rate"]["type"] == "deterministic"
    assert trace["quadrature_effective_nodes"]["type"] == "deterministic"


def test_amplitude_reconstruction_model_raises_on_a_missing_statistic() -> None:
    samples = _reconstruction_samples()
    del samples["template_optimal_snr"]

    with pytest.raises(TypeError, match="template_optimal_snr"):
        _reconstruction_draws(samples)
