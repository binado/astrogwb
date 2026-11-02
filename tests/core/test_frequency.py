from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.frequency import (
    bin_widths,
    frequency_grid,
    frequency_mask,
    log_frequency_grid,
    loglinear_frequency_grid,
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


def test_loglinear_grid_is_uniform_then_geometric_with_continuous_spacing() -> None:
    frequencies = loglinear_frequency_grid(2.0, 2048.0, 1.0, 100.0)

    np.testing.assert_array_equal(
        frequencies[frequencies <= 100.0], uniform_frequency_grid(2.0, 100.0, 1.0)
    )
    above = frequencies[frequencies >= 100.0]
    ratios = above[1:] / above[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
    # The first geometric gap does not exceed df and is within one geometric
    # step of it.
    gaps = np.diff(frequencies)
    last_linear = np.flatnonzero(frequencies == 100.0)[0] - 1
    assert gaps[last_linear] == 1.0
    assert gaps[last_linear + 1] <= 1.0
    assert gaps[last_linear + 1] > 1.0 - 1.0 / 100.0


def test_loglinear_grid_turns_at_the_last_linear_point_below_the_request() -> None:
    frequencies = loglinear_frequency_grid(2.0, 2048.0, 4.0, 101.0)

    # 2 + 4k <= 101 gives k = 24, so the effective turnover is 98.
    assert 98.0 in frequencies
    np.testing.assert_array_equal(frequencies[:25], 2.0 + 4.0 * np.arange(25))
    assert frequencies[25] > 98.0
    assert frequencies[25] - 98.0 <= 4.0


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


@pytest.mark.parametrize(
    ("df", "turnover", "resolution", "expected", "linear_count"),
    [
        (2.0, 4.0, np.log(2.0), [2.0, 4.0, 8.0, 16.0], 2),
        (2.0, 5.0, np.log(2.0), [2.0, 4.0, 8.0, 16.0], 2),
        (2.0, 4.0, 2.0, [2.0, 4.0, 16.0], 2),
        (8.0, 4.0, 3.0, [2.0, 16.0], 1),
        (1.0, 4.0, np.log(2.0), [2.0, 3.0, 4.0, 8.0, 16.0], 3),
    ],
)
def test_loglinear_independent_tail_has_exact_expected_grid(
    df: float,
    turnover: float,
    resolution: float,
    expected: list[float],
    linear_count: int,
) -> None:
    grid = loglinear_frequency_grid(2.0, 16.0, df, turnover, dlogf=resolution)
    np.testing.assert_array_equal(grid, expected)
    default = loglinear_frequency_grid(2.0, 16.0, df, turnover)
    np.testing.assert_array_equal(grid[:linear_count], default[:linear_count])
    np.testing.assert_array_equal(
        frequency_grid("loglinear", 2.0, 16.0, df, turnover, dlogf=resolution),
        grid,
    )
    assert np.all(np.diff(grid) > 0.0)


@pytest.mark.parametrize(("turnover", "start", "linear_count"), [(101.0, 98.0, 25)])
@pytest.mark.parametrize(("resolution", "tail_steps"), [(0.1, 31), (0.3, 11), (4.0, 1)])
def test_loglinear_tail_aligns_endpoint_with_bounded_uniform_log_steps(
    turnover: float,
    start: float,
    linear_count: int,
    resolution: float,
    tail_steps: int,
) -> None:
    grid = loglinear_frequency_grid(2.0, 2048.0, 4.0, turnover, dlogf=resolution)
    assert grid.size == linear_count + tail_steps
    assert grid[0] == 2.0
    assert grid[-1] == 2048.0
    np.testing.assert_array_equal(grid[:linear_count], np.arange(2.0, 99.0, 4.0))
    steps = np.log(grid[linear_count:] / grid[linear_count - 1 : -1])
    expected_step = np.log(2048.0 / start) / tail_steps
    slack = 64 * np.finfo(np.float64).eps
    assert np.all(np.abs(steps - expected_step) <= slack)
    assert np.all(steps <= resolution + slack)
    assert np.all(np.diff(grid) > 0.0)
