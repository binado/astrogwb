"""Per-bin Fisher matrices of the Gaussian spectrum likelihood."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    mock_merger_rate_fn,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro.infer.util import log_density

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.importance.spectral import build_importance_spectrum
from astrogwb.populations import DEFAULT_DENSITY_SITES
from astrogwb.sampling import (
    fisher_matrix_per_bin,
    gwb_spectral_density_model,
    spectral_density_jacobian,
)

FREQUENCIES = jnp.linspace(10.0, 100.0, 7)
SCALE = 0.1 + 0.01 * FREQUENCIES
POWER_LAW = {"amplitude": 2.0, "index": 0.7, "offset": 0.3}


def power_law(
    params: Mapping[str, ArrayLike],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    spectrum = params["amplitude"] * (FREQUENCIES / 10.0) ** params["index"]
    return jnp.asarray(spectrum + params["offset"]), {}


def test_jacobian_and_fisher_match_the_closed_form() -> None:
    names = ("amplitude", "index")
    jacobian = spectral_density_jacobian(power_law, POWER_LAW, names)

    ratio = FREQUENCIES / 10.0
    expected = jnp.stack([ratio**0.7, 2.0 * ratio**0.7 * jnp.log(ratio)], axis=-1)
    np.testing.assert_allclose(jacobian, expected, rtol=1e-12)

    fisher = fisher_matrix_per_bin(power_law, POWER_LAW, names, scale=SCALE)
    assert fisher.shape == (FREQUENCIES.size, 2, 2)
    np.testing.assert_allclose(
        fisher,
        expected[:, :, None] * expected[:, None, :] / SCALE[:, None, None] ** 2,
        rtol=1e-12,
    )


def test_columns_follow_the_requested_order_and_others_stay_fixed() -> None:
    forward = spectral_density_jacobian(power_law, POWER_LAW, ("amplitude", "index"))
    reverse = spectral_density_jacobian(power_law, POWER_LAW, ("index", "amplitude"))
    np.testing.assert_array_equal(forward, reverse[:, ::-1])
    # An integer fiducial is promoted rather than breaking the derivative.
    integer = {**POWER_LAW, "amplitude": 2}
    np.testing.assert_allclose(
        spectral_density_jacobian(power_law, integer, ("amplitude",)),
        forward[:, :1],
    )


def test_an_unknown_parameter_is_rejected() -> None:
    with pytest.raises(KeyError, match="missing"):
        spectral_density_jacobian(power_law, POWER_LAW, ("missing",))


def test_summed_fisher_is_the_hessian_of_the_model_likelihood() -> None:
    names = ("amplitude", "index", "offset")
    observed, _ = power_law(POWER_LAW)
    model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=power_law,
        observed_spectral_density=observed,
        priors={},
        scale=SCALE,
    )

    def negative_log_likelihood(free: dict[str, jax.Array]) -> jax.Array:
        # The model samples nothing, so the parameters enter through the spectrum.
        def at_free(
            params: Mapping[str, ArrayLike],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            return power_law(free)

        conditioned = partial(model, spectral_density_fn=at_free)
        return -log_density(conditioned, (), {}, {})[0]

    free = {name: jnp.asarray(POWER_LAW[name]) for name in names}
    hessian = jax.hessian(negative_log_likelihood)(free)
    expected = jnp.array([[hessian[a][b] for b in names] for a in names])

    fisher = fisher_matrix_per_bin(power_law, POWER_LAW, names, scale=SCALE)
    np.testing.assert_allclose(fisher.sum(axis=0), expected, rtol=1e-10)


def test_masked_bins_contribute_zero_even_with_infinite_scale() -> None:
    names = ("amplitude", "index")
    mask = jnp.arange(FREQUENCIES.size) % 2 == 0
    scale = jnp.where(mask, SCALE, jnp.inf)
    masked = fisher_matrix_per_bin(
        power_law, POWER_LAW, names, scale=scale, frequency_mask=mask
    )
    full = fisher_matrix_per_bin(power_law, POWER_LAW, names, scale=SCALE)
    assert np.all(np.isfinite(masked))
    np.testing.assert_array_equal(masked[~mask], 0.0)
    np.testing.assert_allclose(masked[mask], full[mask], rtol=1e-12)


def test_importance_jacobian_matches_finite_differences(
    mock_catalog_factory: Callable[..., PolarizationPowerCatalog],
) -> None:
    spectrum, _ = build_importance_spectrum(
        mock_catalog_factory(num_sources=256),
        source_model=mock_target_model(),
        merger_rate_fn=mock_merger_rate_fn(),
        density_sites=DEFAULT_DENSITY_SITES,
    )
    names = ("H0", "xi_0", "gamma")
    jacobian = spectral_density_jacobian(spectrum, FIDUCIALS, names)
    assert np.all(np.isfinite(jacobian))

    for column, name in enumerate(names):
        step = 1e-5 * abs(FIDUCIALS[name])
        upper = spectrum({**FIDUCIALS, name: FIDUCIALS[name] + step})[0]
        lower = spectrum({**FIDUCIALS, name: FIDUCIALS[name] - step})[0]
        finite_difference = (upper - lower) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian[:, column],
            finite_difference,
            rtol=1e-6,
            atol=1e-8 * float(jnp.max(jnp.abs(finite_difference))),
            err_msg=name,
        )
