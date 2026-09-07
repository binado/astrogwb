"""Contract tests for ``LogPosterior``: dict-native calls, the constrained vs.
unconstrained Jacobian, single-compilation reuse across differing data, and
``evaluate_grid``.

Uses a small analytic ``spectral_density_fn``, the same pattern
``tests/core/test_spectral_sampling.py`` already uses, so these tests stay
fast and need no catalog. As in production (``astrogwb.paper.inference.build_model``),
``spectral_density_fn`` and ``priors`` are baked into the model with
``functools.partial`` -- they are static, not part of the per-call
``model_kwargs`` a ``LogPosterior`` sweeps over. Only ``observed_spectral_density``
and ``scale`` vary between calls.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.infer.util import log_density

from astrogwb.sampling import (
    LogPosterior,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
)

OBSERVED = jnp.array([1.4, 2.0, 3.2])
SCALE = jnp.array([0.7, 0.9, 1.2])

PRIORS = {"h0": dist.Uniform(50.0, 90.0), "tilt": dist.Normal(0.0, 1.0)}


def _identity_amplitude(marginalized_parameter: jax.Array) -> jax.Array:
    return marginalized_parameter


def _analytic(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict[str, Any]]:
    shape = jnp.array([1.0, 1.5, 2.0]) + params["tilt"] * jnp.array([0.1, -0.2, 0.3])
    return jnp.asarray(params["h0"]) * shape, {}


def _data_kwargs() -> dict[str, Any]:
    return {"observed_spectral_density": OBSERVED, "scale": SCALE}


def _model(spectral_density_fn=_analytic, priors=PRIORS):
    return partial(
        gwb_spectral_density_model,
        spectral_density_fn=spectral_density_fn,
        priors=priors,
    )


# --------------------------------------------------------------------------- #
# 1. Dict in, scalar out
# --------------------------------------------------------------------------- #
def test_call_accepts_a_params_dict_and_differentiates() -> None:
    kwargs = _data_kwargs()
    lp = LogPosterior(_model(), template_kwargs=kwargs)
    params = {"h0": jnp.array(70.0), "tilt": jnp.array(0.3)}

    value = lp(params, **kwargs)
    assert value.shape == ()
    assert np.isfinite(value)

    gradient = jax.grad(lp)(params, **kwargs)
    assert set(gradient) == set(params)
    assert all(np.isfinite(g) for g in gradient.values())


# --------------------------------------------------------------------------- #
# 2. The two spaces differ by the Jacobian
# --------------------------------------------------------------------------- #
def test_potential_and_constrained_density_differ_by_the_jacobian() -> None:
    """``h0``'s support is bounded, so its bijector is not the identity.

    ``potential(u) = -log p(x) - log|dx/du|`` (see
    ``numpyro.infer.util.potential_energy`` / ``_unconstrain_reparam``), so
    ``-potential(u) - log p(x)`` isolates the log-Jacobian contributed by
    every transformed site. With ``tilt`` on the real line (identity
    bijector, zero Jacobian), only ``h0``'s ``Uniform`` transform contributes.
    """
    kwargs = _data_kwargs()
    lp = LogPosterior(_model(), template_kwargs=kwargs)

    unconstrained = lp.init_params
    constrained = lp.constrain(unconstrained, **kwargs)
    potential = lp.potential(unconstrained, **kwargs)
    density = lp(constrained, **kwargs)

    transform = dist.biject_to(PRIORS["h0"].support)
    expected_log_det = transform.log_abs_det_jacobian(
        unconstrained["h0"], constrained["h0"]
    )
    np.testing.assert_allclose(-potential - density, expected_log_det, rtol=1e-10)


# --------------------------------------------------------------------------- #
# 3. One compilation across differing data
# --------------------------------------------------------------------------- #
def test_call_retraces_once_per_data_shape_not_per_value() -> None:
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        # A scalar prediction broadcasts against any `scale`/`observed` shape,
        # so a shape change here is driven purely by the call-time data.
        prediction = jnp.asarray(params["h0"]) * (1.0 + 0.1 * params["tilt"])
        return prediction, {}

    kwargs = _data_kwargs()
    lp = LogPosterior(_model(counting_spectrum), template_kwargs=kwargs)
    params = {"h0": jnp.array(70.0), "tilt": jnp.array(0.3)}

    calls.clear()  # `initialize_model` traced the model plainly during __init__
    lp(params, **kwargs)
    assert len(calls) == 1

    lp(params, **{**kwargs, "scale": SCALE * 2.0})
    assert len(calls) == 1, "same shape/dtype must reuse the compiled program"

    lp(params, **{**kwargs, "observed_spectral_density": jnp.zeros(3)})
    assert len(calls) == 1

    lp(
        params,
        observed_spectral_density=jnp.concatenate([OBSERVED, OBSERVED]),
        scale=jnp.concatenate([SCALE, SCALE]),
    )
    assert len(calls) == 2, "a different array shape must trigger exactly one retrace"


# --------------------------------------------------------------------------- #
# 4. evaluate_grid
# --------------------------------------------------------------------------- #
def test_evaluate_grid_1d_matches_a_naive_python_loop() -> None:
    kwargs = _data_kwargs()
    lp = LogPosterior(_model(), template_kwargs=kwargs)
    h0_grid = jnp.linspace(55.0, 85.0, 7)

    result = lp.evaluate_grid({"h0": h0_grid}, fixed={"tilt": jnp.array(0.2)}, **kwargs)
    assert result.shape == (7,)

    naive = jnp.stack(
        [lp({"h0": h0, "tilt": jnp.array(0.2)}, **kwargs) for h0 in h0_grid]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_evaluate_grid_2d_shape_and_axis_order() -> None:
    kwargs = _data_kwargs()
    lp = LogPosterior(_model(), template_kwargs=kwargs)
    h0_grid = jnp.linspace(55.0, 85.0, 5)
    tilt_grid = jnp.linspace(-1.0, 1.0, 4)

    result = lp.evaluate_grid(
        {"h0": h0_grid, "tilt": tilt_grid}, chunk_size=6, **kwargs
    )
    assert result.shape == (5, 4)
    assert bool(jnp.all(jnp.isfinite(result)))

    naive = jnp.stack(
        [
            jnp.stack([lp({"h0": h0, "tilt": tilt}, **kwargs) for tilt in tilt_grid])
            for h0 in h0_grid
        ]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_evaluate_grid_cache_reuses_compilation_across_data_and_fixed_values() -> None:
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        return _analytic(params)

    kwargs = _data_kwargs()
    lp = LogPosterior(_model(counting_spectrum), template_kwargs=kwargs)
    # 6 points with chunk_size=2 divides evenly: `jax.lax.map` traces its
    # mapped function twice whenever the grid size isn't a multiple of
    # `chunk_size` (once for the batched part, once for the remainder), which
    # would masquerade as a second "retrace" here. See `jax.lax.map`'s
    # docstring ("If the axis is not divisible by the batch size...").
    h0_grid = jnp.linspace(55.0, 85.0, 6)

    calls.clear()
    lp.evaluate_grid(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, chunk_size=2, **kwargs
    )
    assert len(calls) == 1

    # A different `fixed` *value* must not retrace.
    lp.evaluate_grid(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.9)}, chunk_size=2, **kwargs
    )
    assert len(calls) == 1

    # A different `scale` *value* of the same shape must not retrace either.
    lp.evaluate_grid(
        {"h0": h0_grid},
        fixed={"tilt": jnp.array(0.1)},
        chunk_size=2,
        **{**kwargs, "scale": SCALE * 1.5},
    )
    assert len(calls) == 1

    # A different chunk_size is a different cache key and must retrace.
    lp.evaluate_grid(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, chunk_size=3, **kwargs
    )
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# 5. Amplitude-marginalized model
# --------------------------------------------------------------------------- #
def test_call_covers_the_amplitude_marginalized_factor_site() -> None:
    model = partial(
        gwb_amplitude_marginalized_model,
        spectral_density_fn=_analytic,
        priors={"tilt": dist.Normal(0.0, 1.0)},
        amplitude_parameter="h0",
        amplitude_fiducial=70.0,
        amplitude_fn=_identity_amplitude,
        amplitude_prior=dist.Uniform(50.0, 90.0),
    )
    kwargs = _data_kwargs()
    lp = LogPosterior(model, template_kwargs=kwargs)
    params = {"tilt": jnp.array(0.3)}

    value = lp(params, **kwargs)
    assert np.isfinite(value)

    expected, _ = log_density(model, (), kwargs, params)
    np.testing.assert_allclose(value, expected, rtol=1e-10)


# --------------------------------------------------------------------------- #
# 6. Init round-trip
# --------------------------------------------------------------------------- #
def test_constrained_init_params_land_inside_every_priors_support() -> None:
    kwargs = _data_kwargs()
    lp = LogPosterior(_model(), template_kwargs=kwargs)
    constrained = lp.constrain(lp.init_params, **kwargs)

    for name, prior in PRIORS.items():
        assert np.isfinite(prior.log_prob(constrained[name])), name


def test_unconstrain_is_the_inverse_of_constrain() -> None:
    kwargs = _data_kwargs()
    lp = LogPosterior(_model(), template_kwargs=kwargs)

    unconstrained = lp.init_params
    roundtrip = lp.unconstrain(lp.constrain(unconstrained, **kwargs), **kwargs)
    for name, value in unconstrained.items():
        np.testing.assert_allclose(roundtrip[name], value, atol=1e-8)


def test_unconstrain_is_the_inverse_of_constrain_with_no_sampled_sites() -> None:
    model = _model(spectral_density_fn=lambda _params: (OBSERVED, {}), priors={})
    kwargs = _data_kwargs()
    lp = LogPosterior(model, template_kwargs=kwargs)

    assert lp.init_params == {}
    assert lp.constrain(lp.init_params, **kwargs) == {}
    assert lp.unconstrain(lp.init_params, **kwargs) == {}
