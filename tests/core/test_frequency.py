from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.frequency import (
    apply_frequency_mask,
    bin_widths,
    frequency_mask,
    noise_weighted_inner_product,
    uniform_frequency_grid,
    validate_frequency_grid,
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
    """Gappy bands are legal: bin widths come from the full grid, never the band."""
    mask = jnp.array([True, False, True, True])
    frequencies = jnp.array([5.0, 10.0, 15.0, 20.0])

    (band,) = apply_frequency_mask(mask, frequencies)

    np.testing.assert_array_equal(np.asarray(band), [5.0, 15.0, 20.0])


def test_noise_weighted_inner_product_matches_explicit_sum() -> None:
    a = jnp.array([1.3, 1.7, 4.6, 2.8])
    b = jnp.array([1.0, 2.0, 4.0, 3.0])
    psd = jnp.array([0.5, 0.4, 0.8, 0.6])
    df = 2.5
    frequencies = 10.0 + df * jnp.arange(4)

    actual = noise_weighted_inner_product(a, b, psd, frequencies)
    expected = df * np.sum(np.asarray(a) * np.asarray(b) / np.asarray(psd) ** 2)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_noise_weighted_inner_product_conjugates_its_first_argument() -> None:
    """The Hermitian product: ``(a|b) = (b|a)^*``, and real for ``a == b``."""
    a = jnp.array([1.0 + 1.0j, 2.0 - 0.5j])
    b = jnp.array([0.5 - 2.0j, 1.5 + 1.0j])
    psd = jnp.array([0.5, 0.4])
    df = 2.0
    frequencies = 10.0 + df * jnp.arange(2)

    actual = noise_weighted_inner_product(a, b, psd, frequencies)
    reversed_ = noise_weighted_inner_product(b, a, psd, frequencies)

    expected = df * np.sum(
        np.conj(np.asarray(a)) * np.asarray(b) / np.asarray(psd) ** 2
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(actual), np.conj(np.asarray(reversed_)), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(noise_weighted_inner_product(a, a, psd, frequencies)),
        df * np.sum(np.abs(np.asarray(a)) ** 2 / np.asarray(psd) ** 2),
        rtol=1e-6,
    )


def test_noise_weighted_inner_product_scales_linearly_with_grid_spacing() -> None:
    a = jnp.array([1.3, 1.7, 4.6, 2.8])
    b = jnp.array([1.0, 2.0, 4.0, 3.0])
    psd = jnp.array([0.5, 0.4, 0.8, 0.6])

    baseline = noise_weighted_inner_product(a, b, psd, 1.0 * jnp.arange(4))
    scaled = noise_weighted_inner_product(a, b, psd, 3.0 * jnp.arange(4))

    np.testing.assert_allclose(
        np.asarray(scaled), 3.0 * np.asarray(baseline), rtol=1e-6
    )


def test_noise_weighted_inner_product_broadcasts_over_leading_axes() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0, 2.0])
    df = 2.0
    frequencies = df * jnp.arange(3)

    actual = noise_weighted_inner_product(a, a, psd, frequencies)

    expected = df * np.sum(np.asarray(a) ** 2 / np.asarray(psd) ** 2, axis=-1)
    assert actual.shape == (2,)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)


def test_noise_weighted_inner_product_honors_the_axis_keyword() -> None:
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    psd = jnp.array([0.5, 1.0])
    df = 2.0
    frequencies = df * jnp.arange(2)

    actual = noise_weighted_inner_product(a, a, psd[:, None], frequencies, axis=0)

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


def test_noise_weighted_inner_product_weights_each_bin_by_its_own_width() -> None:
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    psd = jnp.ones(4)
    frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    actual = noise_weighted_inner_product(a, a, psd, frequencies)

    # Widths on this grid are [1, 1.5, 3, 4].
    np.testing.assert_allclose(np.asarray(actual), 1 * 1 + 1.5 * 4 + 3 * 9 + 4 * 16)


def test_bin_widths_of_a_uniform_grid_is_its_spacing_in_every_bin() -> None:
    """Exact for ``arange(n) * delta_f``, the shape Ripple's grid takes."""
    widths = bin_widths(np.arange(9) * 0.5)

    np.testing.assert_allclose(np.asarray(widths), np.full(9, 0.5))


@pytest.mark.parametrize("stride", [2, 4, 8])
def test_bin_widths_of_a_subsampled_uniform_grid_is_stride_times_spacing(
    stride: int,
) -> None:
    """Coarsening ``frequencies`` coarsens the widths with no width to keep in step."""
    fine_spacing = 0.25
    frequencies = 2.0 + fine_spacing * np.arange(257)

    widths = bin_widths(frequencies[::stride])

    np.testing.assert_allclose(np.asarray(widths), stride * fine_spacing)


def test_bin_widths_of_a_geometric_grid_uses_midpoint_edges() -> None:
    widths = bin_widths(np.array([1.0, 2.0, 4.0, 8.0]))

    np.testing.assert_allclose(np.asarray(widths), [1.0, 1.5, 3.0, 4.0])


def test_bin_widths_of_a_two_bin_grid_is_the_one_gap_in_both() -> None:
    widths = bin_widths(np.array([3.0, 5.0]))

    np.testing.assert_allclose(np.asarray(widths), [2.0, 2.0])


def test_bin_widths_sums_to_the_grid_span_plus_half_an_end_gap_each_side() -> None:
    frequencies = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    total = float(jnp.sum(bin_widths(frequencies)))

    assert total == pytest.approx((16.0 - 1.0) + (1.0 + 8.0) / 2)


def test_bin_widths_midpoint_sum_converges_at_second_order() -> None:
    """Halving the log step quarters the quadrature error on a smooth integrand.

    The covered range is fixed by the widths themselves, so the exact integral
    of ``1/f`` is ``log(b / a)`` over the outer bin edges. A one-sided ``diff``
    width would converge at first order and fail this ratio.
    """

    def error(num_bins: int) -> float:
        grid = np.geomspace(10.0, 1000.0, num_bins)
        widths = np.asarray(bin_widths(grid))
        gaps = np.diff(grid)
        lower, upper = grid[0] - gaps[0] / 2, grid[-1] + gaps[-1] / 2
        return abs(np.sum(widths / grid) - np.log(upper / lower))

    ratio = error(200) / error(400)

    assert 3.0 < ratio < 5.0


def test_bin_widths_can_be_traced_under_jit() -> None:
    frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])

    traced = jax.jit(bin_widths)(frequencies)

    np.testing.assert_allclose(np.asarray(traced), np.asarray(bin_widths(frequencies)))


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        (np.array([1.0]), "at least two bins"),
        (np.array([3.0, 2.0, 1.0]), "strictly increasing"),
        (np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_bin_widths_rejects_malformed_grids(
    frequencies: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bin_widths(frequencies)


def test_validate_frequency_grid_accepts_a_non_uniform_grid() -> None:
    grid = validate_frequency_grid(np.geomspace(1.0, 100.0, 8))

    assert grid.dtype == np.float64
    assert grid.shape == (8,)


def test_validate_frequency_grid_accepts_a_single_bin() -> None:
    assert validate_frequency_grid([5.0]).shape == (1,)


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        (np.array([1.0, 3.0, 2.0]), "strictly increasing"),
        (np.array([1.0, 1.0, 2.0]), "strictly increasing"),
        (np.array([1.0, np.nan, 2.0]), "finite"),
        (np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_validate_frequency_grid_rejects_malformed_grids(
    frequencies: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_frequency_grid(frequencies)
