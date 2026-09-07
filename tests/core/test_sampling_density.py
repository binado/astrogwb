"""Contract tests for ``LogDensityFn``: grid-evaluation correctness against a
naive Python loop, single-compilation reuse across differing data shapes and
across swept-parameter key sets, and coverage of the amplitude-marginalized
model's ``numpyro.factor`` branch.

Uses a small analytic ``spectral_density_fn``, the same pattern
``tests/core/test_spectral_sampling.py`` already uses, so these tests stay
fast and need no catalog. As in production (``astrogwb.paper.inference.build_model``),
``spectral_density_fn`` and ``priors`` are baked into the model with
``functools.partial`` -- they are static, not part of the per-call
``model_kwargs`` a ``LogDensityFn`` sweeps over. Only ``observed_spectral_density``
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
    LogDensityFn,
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
# Grid correctness, against a naive Python loop over `log_density` directly
# --------------------------------------------------------------------------- #
def test_call_matches_naive_log_density_1d() -> None:
    model = _model()
    kwargs = _data_kwargs()
    lp = LogDensityFn(model)
    h0_grid = jnp.linspace(55.0, 85.0, 7)
    tilt = jnp.array(0.2)

    result = lp({"h0": h0_grid}, fixed={"tilt": tilt}, **kwargs)
    assert result.shape == (7,)

    naive = jnp.stack(
        [log_density(model, (), kwargs, {"h0": h0, "tilt": tilt})[0] for h0 in h0_grid]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_call_shape_and_axis_order_2d() -> None:
    model = _model()
    kwargs = _data_kwargs()
    lp = LogDensityFn(model, chunk_size=6)
    h0_grid = jnp.linspace(55.0, 85.0, 5)
    tilt_grid = jnp.linspace(-1.0, 1.0, 4)

    result = lp({"h0": h0_grid, "tilt": tilt_grid}, **kwargs)
    assert result.shape == (5, 4)
    assert bool(jnp.all(jnp.isfinite(result)))

    naive = jnp.stack(
        [
            jnp.stack(
                [
                    log_density(model, (), kwargs, {"h0": h0, "tilt": tilt})[0]
                    for tilt in tilt_grid
                ]
            )
            for h0 in h0_grid
        ]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


# --------------------------------------------------------------------------- #
# One compilation across differing data shapes
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
    lp = LogDensityFn(_model(counting_spectrum))
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}

    calls.clear()
    lp(grids, fixed=fixed, **kwargs)
    assert len(calls) == 1

    lp(grids, fixed=fixed, **{**kwargs, "scale": SCALE * 2.0})
    assert len(calls) == 1, "same shape/dtype must reuse the compiled program"

    lp(grids, fixed=fixed, observed_spectral_density=jnp.zeros(3), scale=SCALE)
    assert len(calls) == 1

    lp(
        grids,
        fixed=fixed,
        observed_spectral_density=jnp.concatenate([OBSERVED, OBSERVED]),
        scale=jnp.concatenate([SCALE, SCALE]),
    )
    assert len(calls) == 2, "a different array shape must trigger exactly one retrace"


# --------------------------------------------------------------------------- #
# One compilation across differing swept-parameter key sets and `fixed` values
# --------------------------------------------------------------------------- #
def test_call_reuses_compilation_across_grid_key_sets() -> None:
    """A single `jax.jit` object already caches per argument pytree structure.

    Revisiting a previously-seen combination of swept parameter names and
    data shapes is a cache hit with no bookkeeping of our own -- verified
    here by going back to the first grid/shape after an intervening call
    with a *different* key set, and confirming it reuses the first compile
    rather than producing a third one. `chunk_size` stays at its default
    (`None`, unbatched) throughout so `jax.lax.map`'s own remainder-chunk
    retracing (see the batched test below) never conflates with this.
    """
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        return _analytic(params)

    kwargs = _data_kwargs()
    lp = LogDensityFn(_model(counting_spectrum))
    h0_grid = jnp.linspace(55.0, 85.0, 5)
    tilt_grid = jnp.linspace(-1.0, 1.0, 4)

    calls.clear()
    lp({"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, **kwargs)
    assert len(calls) == 1

    # Same grid key set, different `fixed` *value*: no retrace.
    lp({"h0": h0_grid}, fixed={"tilt": jnp.array(0.9)}, **kwargs)
    assert len(calls) == 1

    # Same grid key set, different `scale` *value* of the same shape: no retrace.
    lp(
        {"h0": h0_grid},
        fixed={"tilt": jnp.array(0.1)},
        **{**kwargs, "scale": SCALE * 1.5},
    )
    assert len(calls) == 1

    # A different grid *key set* is a new argument pytree structure: one retrace.
    lp({"h0": h0_grid, "tilt": tilt_grid}, **kwargs)
    assert len(calls) == 2

    # Back to the first key set/shape: reuses the FIRST compile, not a third one.
    lp({"h0": h0_grid}, fixed={"tilt": jnp.array(0.3)}, **kwargs)
    assert len(calls) == 2


def test_batched_evaluation_matches_unbatched() -> None:
    """`chunk_size` only bounds peak memory; the result must not depend on it."""
    kwargs = _data_kwargs()
    h0_grid = jnp.linspace(55.0, 85.0, 6)  # divides evenly by chunk_size=2 below

    unbatched = LogDensityFn(_model())(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, **kwargs
    )
    batched = LogDensityFn(_model(), chunk_size=2)(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, **kwargs
    )
    np.testing.assert_allclose(batched, unbatched, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Amplitude-marginalized model
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
    lp = LogDensityFn(model)
    tilt_grid = jnp.array([-0.3, 0.0, 0.3])

    result = lp({"tilt": tilt_grid}, **kwargs)
    assert result.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(result)))

    naive = jnp.stack(
        [log_density(model, (), kwargs, {"tilt": tilt})[0] for tilt in tilt_grid]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Positional model_args support and JIT caching
# --------------------------------------------------------------------------- #
def _positional_model(
    observed_spectral_density: jax.Array,
    scale: jax.Array,
    *,
    spectral_density_fn=_analytic,
    priors=PRIORS,
) -> None:
    gwb_spectral_density_model(
        spectral_density_fn=spectral_density_fn,
        observed_spectral_density=observed_spectral_density,
        priors=priors,
        scale=scale,
    )


def test_call_matches_naive_log_density_with_model_args() -> None:
    model = partial(_positional_model, spectral_density_fn=_analytic, priors=PRIORS)
    lp = LogDensityFn(model)
    h0_grid = jnp.linspace(55.0, 85.0, 7)
    tilt = jnp.array(0.2)

    result = lp(
        {"h0": h0_grid},
        fixed={"tilt": tilt},
        model_args=(OBSERVED, SCALE),
    )
    assert result.shape == (7,)

    naive = jnp.stack(
        [
            log_density(model, (OBSERVED, SCALE), {}, {"h0": h0, "tilt": tilt})[0]
            for h0 in h0_grid
        ]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_call_retraces_once_per_model_args_shape_not_per_value() -> None:
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        prediction = jnp.asarray(params["h0"]) * (1.0 + 0.1 * params["tilt"])
        return prediction, {}

    model = partial(
        _positional_model, spectral_density_fn=counting_spectrum, priors=PRIORS
    )
    lp = LogDensityFn(model)
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}

    calls.clear()
    lp(grids, fixed=fixed, model_args=(OBSERVED, SCALE))
    assert len(calls) == 1

    lp(grids, fixed=fixed, model_args=(OBSERVED, SCALE * 2.0))
    assert len(calls) == 1, "same shape/dtype must reuse the compiled program"

    lp(
        grids,
        fixed=fixed,
        model_args=(
            jnp.concatenate([OBSERVED, OBSERVED]),
            jnp.concatenate([SCALE, SCALE]),
        ),
    )
    assert len(calls) == 2, "a different array shape must trigger exactly one retrace"
