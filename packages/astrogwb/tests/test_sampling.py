from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb.detector import gaussian_bin_scale
from astrogwb.sampling import (
    AmplitudeQuadrature,
    amplitude_marginalized_model,
    make_amplitude_quadrature,
    spectral_density_model,
)
from numpyro import handlers
from numpyro.infer.util import log_density

type _AmplitudePrior = dist.Normal | dist.Uniform


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

    trace = handlers.trace(
        handlers.seed(
            spectral_density_model,
            rng_seed=0,
        )
    ).get_trace(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=jnp.array([1.0, 2.0]),
        effective_psd=jnp.ones(2),
        observation_time=3.0,
        average_mode="catalog_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        constants={"scale": jnp.array(2.0)},
    )

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


_QUADRATURE = make_amplitude_quadrature(
    grid=jnp.linspace(0.1, 2.5, 2001),
    log_prior=jnp.full(2001, -jnp.log(2.4)),
    scaling=lambda marginalized_parameter: marginalized_parameter,
)


def test_amplitude_marginalized_model_registers_expected_sites() -> None:
    trace = handlers.trace(
        handlers.seed(amplitude_marginalized_model, rng_seed=0)
    ).get_trace(
        **_MARGINALIZED_KWARGS,
        quadrature=_QUADRATURE,
        priors={"tilt": dist.Normal(0.0, 1.0)},
    )

    assert "amplitude_mle" in trace
    assert "template_optimal_snr" in trace
    assert "importance_relative_ess" in trace
    assert "total_merger_rate" in trace
    np.testing.assert_allclose(
        np.asarray(trace["total_merger_rate"]["value"]), FIDUCIAL_RATE
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
        quadrature=_QUADRATURE,
        priors={"tilt": dist.Normal(0.0, 1.0)},
        constants={"local_merger_rate": 99.0},
    )

    assert float(seen[0]["local_merger_rate"]) == FIDUCIAL_RATE


def test_amplitude_marginalized_model_rejects_a_sampled_amplitude() -> None:
    with pytest.raises(ValueError, match="cannot also be sampled"):
        handlers.seed(amplitude_marginalized_model, rng_seed=0)(
            **_MARGINALIZED_KWARGS,
            quadrature=_QUADRATURE,
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

    trace = handlers.trace(
        handlers.seed(amplitude_marginalized_model, rng_seed=0)
    ).get_trace(
        **kwargs,
        quadrature=_QUADRATURE,
        priors={},
        constants={"tilt": 0.3},
    )

    assert np.isfinite(float(trace["amplitude_mle"]["value"]))
    assert np.isfinite(float(trace["template_optimal_snr"]["value"]))


def _quadrature_from_amplitude_prior(
    prior: _AmplitudePrior, num: int = 40_001
) -> AmplitudeQuadrature:
    """Build the physical-parameter grid an amplitude prior would induce.

    Identity scaling, so the marginalized parameter is the amplitude itself.
    ``log_prior`` is a normalized density on the grid (caller's contract).
    """
    if isinstance(prior, dist.Uniform):
        low, high = float(prior.low), float(prior.high)
        grid = jnp.linspace(low, high, num)
        log_prior = jnp.full_like(grid, -jnp.log(high - low))
    else:
        loc, scale = float(prior.loc), float(prior.scale)
        grid = jnp.linspace(loc - 40.0 * scale, loc + 40.0 * scale, num)
        log_prior = dist.Normal(loc, scale).log_prob(grid)
    return make_amplitude_quadrature(
        grid=grid,
        log_prior=log_prior,
        scaling=lambda marginalized_parameter: marginalized_parameter,
    )


@pytest.mark.parametrize(
    ("amplitude_prior", "rate_prior"),
    [
        (dist.Uniform(0.5, 1.5), dist.Uniform(1.0, 3.0)),
        (dist.Normal(1.0, 0.4), dist.Normal(2.0, 0.8)),
    ],
    ids=["uniform", "normal"],
)
def test_amplitude_marginalized_model_matches_the_general_model(
    amplitude_prior: _AmplitudePrior,
    rate_prior: _AmplitudePrior,
) -> None:
    """Numerically marginalize the general model and compare the log densities.

    ``rate_prior`` is the pushforward of ``amplitude_prior`` under
    ``rate = FIDUCIAL_RATE * amplitude``. Integrating the general model over
    ``rate`` rather than ``amplitude`` cancels the Jacobian exactly, so the two
    log densities must agree without any leftover constant -- which is what
    makes this a joint check on the factor term, the normalizations, and the
    reference injection.
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
            "quadrature": _quadrature_from_amplitude_prior(amplitude_prior),
            "priors": {"tilt": dist.Normal(0.0, 1.0)},
        },
        {"tilt": jnp.asarray(tilt)},
    )

    np.testing.assert_allclose(float(marginalized), numerical, rtol=1e-3, atol=1e-3)
