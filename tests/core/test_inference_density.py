"""Contract tests for ``LogDensityFn``: grid-evaluation correctness against a
naive Python loop, single-compilation reuse across differing data shapes and
and across swept-parameter key sets.

Uses a small analytic ``spectral_density_fn``, the same pattern
``tests/core/test_spectral_inference.py`` already uses, so these tests stay
fast and need no catalog. As in the MCMC entrypoints,
``spectral_density_fn`` and ``priors`` are baked into the model with
``functools.partial`` -- they are static, not part of the per-call
``model_kwargs`` a ``LogDensityFn`` sweeps over. Only ``observed_spectral_density``
and ``scale`` vary between calls.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax.tree_util import Partial
from jax.typing import ArrayLike
from numpyro.infer.util import log_density

from astrogwb.inference import (
    GaussianGWBBatchedLikelihood,
    LogDensityFn,
    gwb_spectral_density_model,
)


@pytest.fixture
def observed() -> jax.Array:
    return jnp.array([1.4, 2.0, 3.2])


@pytest.fixture
def scale() -> jax.Array:
    return jnp.array([0.7, 0.9, 1.2])


@pytest.fixture
def priors() -> dict[str, dist.Distribution]:
    return {"h0": dist.Uniform(50.0, 90.0), "tilt": dist.Normal(0.0, 1.0)}


@pytest.fixture
def data_kwargs(observed: jax.Array, scale: jax.Array) -> dict[str, Any]:
    return {"observed_spectral_density": observed, "scale": scale}


def _analytic(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict[str, Any]]:
    shape = jnp.array([1.0, 1.5, 2.0]) + params["tilt"] * jnp.array([0.1, -0.2, 0.3])
    return jnp.asarray(params["h0"]) * shape, {}


@pytest.fixture
def model_factory(
    priors: dict[str, dist.Distribution],
) -> Callable[..., Callable[..., None]]:
    def _factory(
        spectral_density_fn: Callable[..., Any] = _analytic,
        priors: Mapping[str, dist.Distribution] = priors,
    ) -> Callable[..., None]:
        return partial(
            gwb_spectral_density_model,
            spectral_density_fn=spectral_density_fn,
            priors=priors,
        )

    return _factory


@pytest.fixture
def model(
    model_factory: Callable[..., Callable[..., None]],
) -> Callable[..., None]:
    return model_factory()


@pytest.fixture
def log_density_fn(model: Callable[..., None]) -> LogDensityFn:
    return LogDensityFn(model)


def _positional_model(
    observed_spectral_density: jax.Array,
    scale: jax.Array,
    *,
    priors: Mapping[str, dist.Distribution],
    spectral_density_fn: Callable[..., Any] = _analytic,
) -> None:
    gwb_spectral_density_model(
        spectral_density_fn=spectral_density_fn,
        observed_spectral_density=observed_spectral_density,
        priors=priors,
        scale=scale,
    )


@pytest.fixture
def positional_model_factory(
    priors: dict[str, dist.Distribution],
) -> Callable[..., Callable[..., None]]:
    def _factory(
        spectral_density_fn: Callable[..., Any] = _analytic,
    ) -> Callable[..., None]:
        return partial(
            _positional_model,
            spectral_density_fn=spectral_density_fn,
            priors=priors,
        )

    return _factory


@pytest.fixture
def positional_model(
    positional_model_factory: Callable[..., Callable[..., None]],
) -> Callable[..., None]:
    return positional_model_factory()


def test_call_matches_naive_log_density_1d(
    log_density_fn: LogDensityFn,
    model: Callable[..., None],
    data_kwargs: dict[str, Any],
) -> None:
    h0_grid = jnp.linspace(55.0, 85.0, 7)
    tilt = jnp.array(0.2)

    result = log_density_fn({"h0": h0_grid}, fixed={"tilt": tilt}, **data_kwargs)
    assert result.shape == (7,)

    naive = jnp.stack(
        [
            log_density(model, (), data_kwargs, {"h0": h0, "tilt": tilt})[0]
            for h0 in h0_grid
        ]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_call_shape_and_axis_order_2d(
    model: Callable[..., None],
    data_kwargs: dict[str, Any],
) -> None:
    lp = LogDensityFn(model, chunk_size=6)
    h0_grid = jnp.linspace(55.0, 85.0, 5)
    tilt_grid = jnp.linspace(-1.0, 1.0, 4)

    result = lp({"h0": h0_grid, "tilt": tilt_grid}, **data_kwargs)
    assert result.shape == (5, 4)
    assert bool(jnp.all(jnp.isfinite(result)))

    naive = jnp.stack(
        [
            jnp.stack(
                [
                    log_density(model, (), data_kwargs, {"h0": h0, "tilt": tilt})[0]
                    for tilt in tilt_grid
                ]
            )
            for h0 in h0_grid
        ]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_call_retraces_once_per_data_shape_not_per_value(
    model_factory: Callable[..., Callable[..., None]],
    data_kwargs: dict[str, Any],
    observed: jax.Array,
    scale: jax.Array,
) -> None:
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        # A scalar prediction broadcasts against any `scale`/`observed` shape,
        # so a shape change here is driven purely by the call-time data.
        prediction = jnp.asarray(params["h0"]) * (1.0 + 0.1 * params["tilt"])
        return prediction, {}

    lp = LogDensityFn(model_factory(counting_spectrum))
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}

    calls.clear()
    lp(grids, fixed=fixed, **data_kwargs)
    assert len(calls) == 1

    lp(grids, fixed=fixed, **{**data_kwargs, "scale": scale * 2.0})
    assert len(calls) == 1, "same shape/dtype must reuse the compiled program"

    lp(grids, fixed=fixed, observed_spectral_density=jnp.zeros(3), scale=scale)
    assert len(calls) == 1

    lp(
        grids,
        fixed=fixed,
        observed_spectral_density=jnp.concatenate([observed, observed]),
        scale=jnp.concatenate([scale, scale]),
    )
    assert len(calls) == 2, "a different array shape must trigger exactly one retrace"


def test_call_reuses_compilation_across_grid_key_sets(
    model_factory: Callable[..., Callable[..., None]],
    data_kwargs: dict[str, Any],
    scale: jax.Array,
) -> None:
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

    lp = LogDensityFn(model_factory(counting_spectrum))
    h0_grid = jnp.linspace(55.0, 85.0, 5)
    tilt_grid = jnp.linspace(-1.0, 1.0, 4)

    calls.clear()
    lp({"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, **data_kwargs)
    assert len(calls) == 1

    # Same grid key set, different `fixed` *value*: no retrace.
    lp({"h0": h0_grid}, fixed={"tilt": jnp.array(0.9)}, **data_kwargs)
    assert len(calls) == 1

    # Same grid key set, different `scale` *value* of the same shape: no retrace.
    lp(
        {"h0": h0_grid},
        fixed={"tilt": jnp.array(0.1)},
        **{**data_kwargs, "scale": scale * 1.5},
    )
    assert len(calls) == 1

    # A different grid *key set* is a new argument pytree structure: one retrace.
    lp({"h0": h0_grid, "tilt": tilt_grid}, **data_kwargs)
    assert len(calls) == 2

    # Back to the first key set/shape: reuses the FIRST compile, not a third one.
    lp({"h0": h0_grid}, fixed={"tilt": jnp.array(0.3)}, **data_kwargs)
    assert len(calls) == 2


def test_batched_evaluation_matches_unbatched(
    log_density_fn: LogDensityFn,
    model: Callable[..., None],
    data_kwargs: dict[str, Any],
) -> None:
    """`chunk_size` only bounds peak memory; the result must not depend on it."""
    h0_grid = jnp.linspace(55.0, 85.0, 6)  # divides evenly by chunk_size=2 below

    unbatched = log_density_fn(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, **data_kwargs
    )
    batched = LogDensityFn(model, chunk_size=2)(
        {"h0": h0_grid}, fixed={"tilt": jnp.array(0.1)}, **data_kwargs
    )
    np.testing.assert_allclose(batched, unbatched, rtol=1e-10)


def test_call_matches_naive_log_density_with_model_args(
    positional_model: Callable[..., None],
    observed: jax.Array,
    scale: jax.Array,
) -> None:
    lp = LogDensityFn(positional_model)
    h0_grid = jnp.linspace(55.0, 85.0, 7)
    tilt = jnp.array(0.2)

    result = lp(
        {"h0": h0_grid},
        fixed={"tilt": tilt},
        model_args=(observed, scale),
    )
    assert result.shape == (7,)

    naive = jnp.stack(
        [
            log_density(
                positional_model, (observed, scale), {}, {"h0": h0, "tilt": tilt}
            )[0]
            for h0 in h0_grid
        ]
    )
    np.testing.assert_allclose(result, naive, rtol=1e-10)


def test_call_retraces_once_per_model_args_shape_not_per_value(
    positional_model_factory: Callable[..., Callable[..., None]],
    observed: jax.Array,
    scale: jax.Array,
) -> None:
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        prediction = jnp.asarray(params["h0"]) * (1.0 + 0.1 * params["tilt"])
        return prediction, {}

    lp = LogDensityFn(positional_model_factory(counting_spectrum))
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}

    calls.clear()
    lp(grids, fixed=fixed, model_args=(observed, scale))
    assert len(calls) == 1

    lp(grids, fixed=fixed, model_args=(observed, scale * 2.0))
    assert len(calls) == 1, "same shape/dtype must reuse the compiled program"

    lp(
        grids,
        fixed=fixed,
        model_args=(
            jnp.concatenate([observed, observed]),
            jnp.concatenate([scale, scale]),
        ),
    )
    assert len(calls) == 2, "a different array shape must trigger exactly one retrace"


class Networks(TypedDict):
    scale: jax.Array
    frequency_mask: jax.Array


@pytest.fixture
def networks() -> Networks:
    """Three networks: plain, a different scale, and one with masked bins."""
    scale = jnp.array([[0.7, 0.9, 1.2], [0.4, 1.5, 0.8], [0.6, jnp.inf, 1.0]])
    mask = jnp.array([[True] * 3, [True] * 3, [True, False, True]])
    return {"scale": scale, "frequency_mask": mask}


def _per_network_reference(
    model: Callable[..., None],
    grids: Mapping[str, jax.Array],
    fixed: Mapping[str, ArrayLike],
    observed: jax.Array,
    networks: Networks,
) -> jax.Array:
    lp = LogDensityFn(model)
    return jnp.stack(
        [
            lp(
                grids,
                fixed=fixed,
                observed_spectral_density=observed,
                scale=networks["scale"][k],
                frequency_mask=networks["frequency_mask"][k],
            )
            for k in range(networks["scale"].shape[0])
        ]
    )


@pytest.mark.parametrize(
    ("grids", "fixed"),
    [
        ({"h0": jnp.linspace(55.0, 85.0, 7)}, {"tilt": jnp.array(0.2)}),
        (
            {"h0": jnp.linspace(55.0, 85.0, 5), "tilt": jnp.linspace(-1.0, 1.0, 4)},
            {},
        ),
    ],
    ids=["1d", "2d"],
)
def test_batched_likelihood_matches_log_density_fn_per_network(
    model: Callable[..., None],
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
    grids: dict[str, jax.Array],
    fixed: dict[str, jax.Array],
) -> None:
    batched = GaussianGWBBatchedLikelihood(priors, chunk_size=3)(
        grids,
        spectral_density_fn=_analytic,
        fixed=fixed,
        observed_spectral_density=observed,
        **networks,
    )
    expected = _per_network_reference(model, grids, fixed, observed, networks)
    np.testing.assert_allclose(batched, expected, rtol=1e-10)


def test_batched_likelihood_2d_grids_returns_k_first_in_insertion_order(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    grids = {"tilt": jnp.linspace(-1.0, 1.0, 4), "h0": jnp.linspace(55.0, 85.0, 5)}
    result = GaussianGWBBatchedLikelihood(priors)(
        grids,
        spectral_density_fn=_analytic,
        observed_spectral_density=observed,
        **networks,
    )
    assert result.shape == (3, 4, 5)


def test_batched_likelihood_predicts_once_regardless_of_network_count(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    calls: list[None] = []

    def counting_spectrum(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict]:
        calls.append(None)
        return _analytic(params)

    lp = GaussianGWBBatchedLikelihood(priors)
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}

    lp(
        grids,
        spectral_density_fn=counting_spectrum,
        fixed=fixed,
        observed_spectral_density=observed,
        **networks,
    )
    assert len(calls) == 1, "one trace, however many networks"

    lp(
        grids,
        spectral_density_fn=counting_spectrum,
        fixed=fixed,
        observed_spectral_density=observed,
        scale=networks["scale"] * 2.0,
        frequency_mask=~networks["frequency_mask"],
    )
    assert len(calls) == 1, "values of scale and mask are traced"

    lp(
        grids,
        spectral_density_fn=counting_spectrum,
        fixed=fixed,
        observed_spectral_density=observed,
        scale=networks["scale"][:2],
        frequency_mask=networks["frequency_mask"][:2],
    )
    assert len(calls) == 2, "a new K retraces once"


def test_batched_likelihood_outside_prior_support_is_negative_infinite(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    result = GaussianGWBBatchedLikelihood(priors)(
        {"h0": jnp.array([40.0, 70.0])},
        spectral_density_fn=_analytic,
        fixed={"tilt": jnp.array(0.0)},
        observed_spectral_density=observed,
        **networks,
    )
    assert bool(jnp.all(result[:, 0] == -jnp.inf))
    assert bool(jnp.all(jnp.isfinite(result[:, 1])))


def test_batched_likelihood_traces_a_pytree_spectrum_once_per_shape(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    """A Partial's arrays are traced inputs: new values reuse the compilation."""
    traces: list[None] = []

    def scaled(params: Mapping[str, ArrayLike], *, shape: jax.Array) -> tuple:
        traces.append(None)
        return jnp.asarray(params["h0"]) * shape, {}

    lp = GaussianGWBBatchedLikelihood(priors)
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}
    shapes = (jnp.array([1.0, 1.5, 2.0]), jnp.array([2.0, 1.0, 0.5]))

    results = [
        lp(
            grids,
            spectral_density_fn=Partial(scaled, shape=shape),
            fixed=fixed,
            observed_spectral_density=observed,
            **networks,
        )
        for shape in shapes
    ]

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1])


def test_batched_likelihood_with_a_pytree_spectrum_matches_its_closure(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    shape = jnp.array([1.0, 1.5, 2.0])

    def scaled(params: Mapping[str, ArrayLike], *, shape: jax.Array) -> tuple:
        return jnp.asarray(params["h0"]) * shape, {}

    lp = GaussianGWBBatchedLikelihood(priors)
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}
    fixed = {"tilt": jnp.array(0.3)}

    traced = lp(
        grids,
        spectral_density_fn=Partial(scaled, shape=shape),
        fixed=fixed,
        observed_spectral_density=observed,
        **networks,
    )
    closed = lp(
        grids,
        spectral_density_fn=partial(scaled, shape=shape),
        fixed=fixed,
        observed_spectral_density=observed,
        **networks,
    )

    np.testing.assert_allclose(traced, closed, rtol=1e-12)


def _shot_noise_variance(
    params: Mapping[str, ArrayLike], *, relative_sd: ArrayLike
) -> jax.Array:
    """A per-bin variance with a tilted relative scatter, so modes differ."""
    tilt = jnp.array([1.0, 1.4, 0.7])
    return (jnp.asarray(relative_sd) * tilt * _analytic(params)[0]) ** 2


@pytest.mark.parametrize("direction", ["amplitude", "per_frequency"])
def test_batched_likelihood_with_shot_noise_matches_the_dense_gaussian(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
    direction: Literal["amplitude", "per_frequency"],
) -> None:
    grids = {"h0": jnp.linspace(60.0, 80.0, 3)}
    fixed = {"tilt": jnp.array(0.3)}
    variance_fn = Partial(_shot_noise_variance, relative_sd=jnp.array(0.05))

    batched = GaussianGWBBatchedLikelihood(priors)(
        grids,
        spectral_density_fn=_analytic,
        fixed=fixed,
        observed_spectral_density=observed,
        shot_noise_variance_fn=variance_fn,
        shot_noise_direction=direction,
        **networks,
    )

    expected = np.empty(batched.shape)
    for k in range(networks["scale"].shape[0]):
        keep = np.asarray(networks["frequency_mask"][k])
        scale_k = np.asarray(networks["scale"][k])[keep]
        for g, h0 in enumerate(np.asarray(grids["h0"])):
            params = {"h0": h0, **fixed}
            prediction = np.asarray(_analytic(params)[0])[keep]
            variance = np.asarray(variance_fn(params))[keep]
            if direction == "amplitude":
                weight = prediction**2 / scale_k**2
                scatter = np.sum(weight * np.sqrt(variance) / prediction)
                u = scatter / np.sum(weight) * prediction
            else:
                u = np.sqrt(variance)
            covariance = np.diag(scale_k**2) + np.outer(u, u)
            log_prior = sum(
                float(np.asarray(p.log_prob(params[n]))) for n, p in priors.items()
            )
            expected[k, g] = log_prior + float(
                dist.MultivariateNormal(
                    jnp.asarray(prediction), jnp.asarray(covariance)
                ).log_prob(jnp.asarray(observed)[keep])
            )

    np.testing.assert_allclose(batched, expected, rtol=1e-10)


@pytest.mark.parametrize("direction", ["amplitude", "per_frequency"])
def test_batched_likelihood_with_a_flat_relative_scatter_matches_a_fixed_one(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
    direction: Literal["amplitude", "per_frequency"],
) -> None:
    def flat(params: Mapping[str, ArrayLike]) -> jax.Array:
        return (0.05 * _analytic(params)[0]) ** 2

    lp = GaussianGWBBatchedLikelihood(priors)
    common: dict[str, Any] = {
        "spectral_density_fn": _analytic,
        "fixed": {"tilt": jnp.array(0.3)},
        "observed_spectral_density": observed,
        **networks,
    }
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}

    evaluated = lp(
        grids,
        shot_noise_variance_fn=flat,
        shot_noise_direction=direction,
        **common,
    )
    pinned = lp(grids, amplitude_shot_noise_variance=0.05**2, **common)

    np.testing.assert_allclose(evaluated, pinned, rtol=1e-12)


def test_batched_likelihood_with_zero_shot_noise_is_the_diagonal_gaussian(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    lp = GaussianGWBBatchedLikelihood(priors)
    common: dict[str, Any] = {
        "spectral_density_fn": _analytic,
        "fixed": {"tilt": jnp.array(0.3)},
        "observed_spectral_density": observed,
        **networks,
    }
    grids = {"h0": jnp.linspace(60.0, 80.0, 4)}

    np.testing.assert_allclose(
        lp(grids, amplitude_shot_noise_variance=jnp.zeros(3), **common),
        lp(grids, **common),
        rtol=1e-12,
    )


def test_batched_likelihood_traces_a_pytree_variance_once_per_shape(
    priors: dict[str, dist.Distribution],
    observed: jax.Array,
    networks: Networks,
) -> None:
    traces: list[None] = []

    def counted(params: Mapping[str, ArrayLike], *, relative_sd: jax.Array) -> Any:
        traces.append(None)
        return _shot_noise_variance(params, relative_sd=relative_sd)

    lp = GaussianGWBBatchedLikelihood(priors)
    results = [
        lp(
            {"h0": jnp.linspace(60.0, 80.0, 4)},
            spectral_density_fn=_analytic,
            fixed={"tilt": jnp.array(0.3)},
            observed_spectral_density=observed,
            shot_noise_variance_fn=Partial(counted, relative_sd=jnp.array(sd)),
            **networks,
        )
        for sd in (0.05, 0.2)
    ]

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1])
