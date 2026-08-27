from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.frequency import frequency_slice, noise_weighted_inner_product


def test_frequency_slice_includes_both_bounds() -> None:
    frequencies = np.array([5.0, 10.0, 20.0, 30.0])

    actual = frequency_slice(frequencies, fmin=10.0, fmax=20.0)

    assert actual == slice(1, 3)
    np.testing.assert_array_equal(frequencies[actual], [10.0, 20.0])


@pytest.mark.parametrize(
    ("fmin", "fmax", "expected"),
    [
        (None, 20.0, slice(0, 3)),
        (10.0, None, slice(1, 4)),
        (None, None, slice(0, 4)),
        (11.0, 29.0, slice(2, 3)),
    ],
)
def test_frequency_slice_handles_open_and_off_bin_bounds(
    fmin: float | None, fmax: float | None, expected: slice
) -> None:
    frequencies = np.array([5.0, 10.0, 20.0, 30.0])

    assert frequency_slice(frequencies, fmin=fmin, fmax=fmax) == expected


def test_frequency_slice_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="fmin .* must be <= fmax"):
        frequency_slice(np.array([10.0, 20.0]), fmin=20.0, fmax=10.0)


def test_frequency_slice_rejects_empty_band() -> None:
    with pytest.raises(ValueError, match="contains no bins"):
        frequency_slice(np.array([10.0, 20.0]), fmin=30.0, fmax=40.0)


def test_frequency_slice_rejects_empty_or_nonmonotonic_grid() -> None:
    with pytest.raises(ValueError, match="at least one bin"):
        frequency_slice(np.array([]))
    with pytest.raises(ValueError, match="strictly increasing"):
        frequency_slice(np.array([10.0, 30.0, 20.0]))


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
