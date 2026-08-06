from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrogwb.frequency import (
    apply_frequency_mask,
    frequency_mask,
    noise_weighted_inner_product,
)


def test_frequency_mask_bounds() -> None:
    freqs = jnp.array([5.0, 10.0, 20.0, 30.0])

    mask = frequency_mask(freqs, fmin=10.0, fmax=20.0)

    np.testing.assert_array_equal(
        np.asarray(mask), np.array([False, True, True, False])
    )


def test_apply_frequency_mask_slices_multiple_1d_arrays() -> None:
    mask = jnp.array([False, True, True, False])
    freqs = jnp.array([5.0, 10.0, 20.0, 30.0])
    spectrum = jnp.array([1.0, 2.0, 3.0, 4.0])

    masked_freqs, masked_spectrum = apply_frequency_mask(mask, freqs, spectrum)

    np.testing.assert_array_equal(np.asarray(masked_freqs), np.array([10.0, 20.0]))
    np.testing.assert_array_equal(np.asarray(masked_spectrum), np.array([2.0, 3.0]))


def test_apply_frequency_mask_honors_leading_frequency_axis() -> None:
    mask = jnp.array([True, False, True])
    power = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    (masked_power,) = apply_frequency_mask(mask, power)

    np.testing.assert_array_equal(
        np.asarray(masked_power), np.array([[1.0, 2.0], [5.0, 6.0]])
    )


def test_apply_frequency_mask_returns_empty_tuple_without_arrays() -> None:
    assert apply_frequency_mask(jnp.array([True, False])) == ()


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
