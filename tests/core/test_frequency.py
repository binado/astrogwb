from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.frequency import (
    FrequencySpacing,
    bin_widths,
    frequency_grid,
    frequency_mask,
    log_frequency_grid,
    noise_weighted_inner_product,
    uniform_frequency_grid,
)


def test_frequency_mask_includes_both_bounds() -> None:
    frequencies = jnp.array([5.0, 10.0, 20.0, 30.0])

    mask = frequency_mask(frequencies, fmin=10.0, fmax=20.0)

    np.testing.assert_array_equal(np.asarray(mask), [False, True, True, False])


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


def test_uniform_frequency_grid_includes_tolerant_endpoint() -> None:
    frequencies = uniform_frequency_grid(0.1, 0.3, 0.1)
    assert frequencies.dtype == np.float64
    assert frequencies.shape == (3,)
    np.testing.assert_array_equal(frequencies, [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    ("spacing", "builder"),
    [("linear", uniform_frequency_grid), ("log", log_frequency_grid)],
)
def test_frequency_grid_builds_the_named_spacing(
    spacing: FrequencySpacing, builder: Callable[[float, float, float], np.ndarray]
) -> None:
    np.testing.assert_array_equal(
        frequency_grid(spacing, 2.0, 200.0, 1.0), builder(2.0, 200.0, 1.0)
    )


def test_log_grid_is_geometric_with_df_wide_first_bin() -> None:
    frequencies = log_frequency_grid(2.0, 2048.0, 0.5)

    assert frequencies[0] == 2.0
    assert frequencies[-1] == 2048.0
    ratios = frequencies[1:] / frequencies[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
    assert frequencies[1] - frequencies[0] <= 0.5


def test_bin_widths_of_a_subsampled_uniform_grid_is_stride_times_spacing() -> None:
    """Coarsening ``frequencies`` coarsens the widths with no width to keep in step."""
    stride = 4
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
