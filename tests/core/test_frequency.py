from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.frequency import (
    apply_frequency_mask,
    frequency_mask,
    noise_weighted_inner_product,
    uniform_frequency_grid,
    uniform_grid_spacing,
)


def test_frequency_mask_includes_both_bounds() -> None:
    frequencies = jnp.array([5.0, 10.0, 20.0, 30.0])

    mask = frequency_mask(frequencies, fmin=10.0, fmax=20.0)

    np.testing.assert_array_equal(np.asarray(mask), [False, True, True, False])


def test_frequency_mask_leaves_open_bounds_unconstrained() -> None:
    frequencies = jnp.array([5.0, 10.0, 20.0, 30.0])

    np.testing.assert_array_equal(
        np.asarray(frequency_mask(frequencies, fmax=20.0)),
        [True, True, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(frequency_mask(frequencies, fmin=10.0)),
        [False, True, True, True],
    )
    np.testing.assert_array_equal(
        np.asarray(frequency_mask(frequencies)), [True, True, True, True]
    )


def test_apply_frequency_mask_compresses_leading_axis() -> None:
    """The ``(F, N)`` layout is why ``axis=0`` is the default."""
    mask = jnp.array([True, False, True, False])
    frequencies = jnp.array([5.0, 10.0, 20.0, 30.0])
    polarization_power = jnp.arange(12.0).reshape(4, 3)

    band_frequencies, band_power = apply_frequency_mask(
        mask, frequencies, polarization_power
    )

    np.testing.assert_array_equal(np.asarray(band_frequencies), [5.0, 20.0])
    assert band_power.shape == (2, 3)
    np.testing.assert_array_equal(
        np.asarray(band_power), np.asarray(polarization_power)[[0, 2]]
    )


def test_apply_frequency_mask_need_not_select_a_contiguous_run() -> None:
    """Gappy bands are legal: df comes from the catalog, never from the band."""
    mask = jnp.array([True, False, True, True])
    frequencies = jnp.array([5.0, 10.0, 15.0, 20.0])

    (band,) = apply_frequency_mask(mask, frequencies)

    np.testing.assert_array_equal(np.asarray(band), [5.0, 15.0, 20.0])


def test_noise_weighted_inner_product_matches_explicit_sum() -> None:
    a = jnp.array([1.3, 1.7, 4.6, 2.8])
    b = jnp.array([1.0, 2.0, 4.0, 3.0])
    psd = jnp.array([0.5, 0.4, 0.8, 0.6])
    df = 2.5

    actual = noise_weighted_inner_product(a, b, psd, df)
    expected = df * np.sum(np.asarray(a) * np.asarray(b) / np.asarray(psd) ** 2)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_noise_weighted_inner_product_scales_linearly_with_df() -> None:
    a = jnp.array([1.3, 1.7, 4.6, 2.8])
    b = jnp.array([1.0, 2.0, 4.0, 3.0])
    psd = jnp.array([0.5, 0.4, 0.8, 0.6])

    baseline = noise_weighted_inner_product(a, b, psd, 1.0)
    scaled = noise_weighted_inner_product(a, b, psd, 3.0)

    np.testing.assert_allclose(
        np.asarray(scaled), 3.0 * np.asarray(baseline), rtol=1e-6
    )


def test_noise_weighted_inner_product_broadcasts_over_leading_axes() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0, 2.0])
    df = 2.0

    actual = noise_weighted_inner_product(a, a, psd, df)

    expected = df * np.sum(np.asarray(a) ** 2 / np.asarray(psd) ** 2, axis=-1)
    assert actual.shape == (2,)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_noise_weighted_inner_product_honors_the_axis_keyword() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0])
    df = 2.0

    actual = noise_weighted_inner_product(a, a, psd[:, None], df, axis=0)

    expected = df * np.sum(np.asarray(a) ** 2 / np.asarray(psd)[:, None] ** 2, axis=0)
    assert actual.shape == (3,)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_uniform_frequency_grid_includes_tolerant_endpoint() -> None:
    frequencies = uniform_frequency_grid(0.1, 0.3, 0.1)
    assert frequencies.dtype == np.float64
    assert frequencies.shape == (3,)
    np.testing.assert_array_equal(frequencies, [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    ("minimum", "maximum", "df"),
    [(2.0, 1.0, 0.5), (1.0, 2.0, 0.0), (1.0, 2.0, -1.0)],
)
def test_uniform_frequency_grid_rejects_invalid_settings(
    minimum: float, maximum: float, df: float
) -> None:
    with pytest.raises(ValueError):
        uniform_frequency_grid(minimum, maximum, df)


def test_uniform_grid_spacing_reads_off_ripples_grid_shape() -> None:
    """Exact for ``arange(n) * delta_f``, the shape Ripple's grid takes."""
    assert uniform_grid_spacing(np.arange(9) * 0.5) == 0.5


def test_uniform_grid_spacing_tolerates_the_analytic_grids_last_ulp() -> None:
    """The tolerance regression test.

    ``f_min + df * arange(n)`` carries about 1 ulp of the *largest* frequency
    in each gap, unlike Ripple's bit-identical ``arange(n) * delta_f``. A
    tolerance scaled to ``df`` itself, rather than to the frequencies, would
    reject exactly this -- the package's own analytic catalogs.
    """
    frequencies = 2.0 + 0.25 * np.arange(16377)
    assert frequencies[-1] == 4096.0
    assert uniform_grid_spacing(frequencies) == 0.25


def test_uniform_grid_spacing_rejects_a_non_uniform_grid() -> None:
    with pytest.raises(ValueError, match="not uniform"):
        uniform_grid_spacing(np.array([1.0, 2.0, 3.5, 4.5]))


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        (np.array([1.0]), "at least two bins"),
        (np.array([3.0, 2.0, 1.0]), "strictly increasing"),
        (np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_uniform_grid_spacing_rejects_malformed_grids(
    frequencies: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        uniform_grid_spacing(frequencies)
