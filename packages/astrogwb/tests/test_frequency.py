from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.frequency import (
    apply_frequency_mask,
    frequency_mask,
    frequency_spacing,
    noise_weighted_inner_product,
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


@pytest.mark.parametrize("asarray", [np.asarray, jnp.asarray])
def test_frequency_spacing_is_namespace_neutral(asarray: Callable[[Any], Any]) -> None:
    freqs = asarray([10.0, 20.0, 40.0])

    np.testing.assert_allclose(np.asarray(frequency_spacing(freqs)), 15.0)


@pytest.mark.parametrize("asarray", [np.asarray, jnp.asarray])
def test_frequency_mask_is_namespace_neutral(asarray: Callable[[Any], Any]) -> None:
    freqs = asarray([5.0, 10.0, 20.0, 30.0])

    mask = frequency_mask(freqs, fmin=10.0, fmax=20.0)

    np.testing.assert_array_equal(
        np.asarray(mask), np.array([False, True, True, False])
    )


@pytest.mark.parametrize("asarray", [np.asarray, jnp.asarray])
def test_apply_frequency_mask_is_namespace_neutral(
    asarray: Callable[[Any], Any],
) -> None:
    mask = asarray([False, True, True, False])
    freqs = asarray([5.0, 10.0, 20.0, 30.0])
    spectrum = asarray([1.0, 2.0, 3.0, 4.0])

    masked_freqs, masked_spectrum = apply_frequency_mask(mask, freqs, spectrum)

    np.testing.assert_array_equal(np.asarray(masked_freqs), np.array([10.0, 20.0]))
    np.testing.assert_array_equal(np.asarray(masked_spectrum), np.array([2.0, 3.0]))


@pytest.mark.parametrize("asarray", [np.asarray, jnp.asarray])
def test_noise_weighted_inner_product_is_namespace_neutral(
    asarray: Callable[[Any], Any],
) -> None:
    a = asarray([1.3, 1.7, 4.6, 2.8])
    b = asarray([1.0, 2.0, 4.0, 3.0])
    psd = asarray([0.5, 0.4, 0.8, 0.6])
    df = 2.5

    actual = noise_weighted_inner_product(a, b, psd, df)

    expected = df * np.sum(np.asarray(a) * np.asarray(b) / np.asarray(psd) ** 2)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
    if asarray is jnp.asarray:
        assert isinstance(actual, jax.Array)
    else:
        assert not isinstance(actual, jax.Array)
        assert isinstance(actual, np.generic | np.ndarray)


@pytest.mark.parametrize("a_asarray", [np.asarray, jnp.asarray])
@pytest.mark.parametrize("b_asarray", [np.asarray, jnp.asarray])
def test_noise_weighted_inner_product_follows_a_namespace_with_mixed_operands(
    a_asarray: Callable[[Any], Any],
    b_asarray: Callable[[Any], Any],
) -> None:
    """b, psd, and df are coerced into a's namespace, never the reverse."""
    a = a_asarray([1.3, 1.7, 4.6, 2.8])
    b = b_asarray([1.0, 2.0, 4.0, 3.0])
    psd = b_asarray([0.5, 0.4, 0.8, 0.6])
    df = b_asarray(2.5)

    actual = noise_weighted_inner_product(a, b, psd, df)

    expected = 2.5 * np.sum(np.asarray(a) * np.asarray(b) / np.asarray(psd) ** 2)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
    if a_asarray is jnp.asarray:
        assert isinstance(actual, jax.Array)
    else:
        assert not isinstance(actual, jax.Array)
        assert isinstance(actual, np.generic | np.ndarray)


def test_noise_weighted_inner_product_honors_the_axis_keyword() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0])
    df = 2.0

    actual = noise_weighted_inner_product(a, a, psd[:, None], df, axis=0)

    expected = df * np.sum(np.asarray(a) ** 2 / np.asarray(psd)[:, None] ** 2, axis=0)
    assert actual.shape == (3,)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
