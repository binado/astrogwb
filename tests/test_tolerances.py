"""Unit tests for the convergence checks in :mod:`tolerances`."""

from __future__ import annotations

import numpy as np
import pytest
from tolerances import assert_converges


def test_assert_converges_passes_for_second_order_1d_and_scalar_oracles() -> None:
    spacings = (0.4, 0.2, 0.1)
    oracle = np.array([2.0, 5.0, 10.0])
    ladder = [oracle * (1.0 + 0.5 * h**2) for h in spacings]
    assert_converges(ladder, oracle, spacings=spacings, order=2.0)

    scalar_oracle = np.array(3.0)
    scalar_ladder = [scalar_oracle * (1.0 + 0.25 * h**2) for h in spacings]
    assert_converges(scalar_ladder, scalar_oracle, spacings=spacings, order=2.0)


def test_assert_converges_short_circuits_when_finest_matches_oracle() -> None:
    spacings = (0.4, 0.2, 0.1)
    oracle = np.array([1.0, 4.0, 9.0])
    identical = [oracle.copy() for _ in spacings]
    assert_converges(identical, oracle, spacings=spacings, order=2.0)


def test_assert_converges_self_convergence_passes_and_rejects_identical_levels() -> (
    None
):
    spacings = (0.4, 0.2, 0.1, 0.05)
    oracle = np.array([1.5, 3.0, 6.0])
    ladder = [oracle * (1.0 + 0.5 * h**2) for h in spacings]
    assert_converges(ladder, spacings=spacings, order=2.0)

    identical = [oracle.copy() for _ in spacings]
    with pytest.raises(AssertionError, match="Non-positive convergence errors"):
        assert_converges(identical, spacings=spacings, order=2.0)


def test_assert_converges_rejects_constant_bias() -> None:
    spacings = (0.4, 0.2, 0.1, 0.05)
    oracle = np.array([1.0, 2.0, 4.0])
    biased = [oracle * (1.0 + 1e-2 + 0.05 * h**2) for h in spacings]
    with pytest.raises(AssertionError):
        assert_converges(biased, oracle, spacings=spacings, order=2.0)


def test_assert_converges_rejects_lower_order_than_required() -> None:
    spacings = (0.4, 0.2, 0.1)
    oracle = np.array([1.0, 2.0, 3.0])
    first_order = [oracle * (1.0 + 0.1 * h) for h in spacings]
    with pytest.raises(AssertionError, match="Observed convergence order"):
        assert_converges(first_order, oracle, spacings=spacings, order=2.0)


def test_assert_converges_handles_zero_entries_in_oracle() -> None:
    spacings = (0.4, 0.2, 0.1)
    mixed_oracle = np.array([0.0, 2.0, 4.0])
    good = [np.array([0.0, 2.0, 4.0]) * (1.0 + 0.5 * h**2) for h in spacings]
    assert_converges(good, mixed_oracle, spacings=spacings, order=2.0)

    bad = [
        np.array([1e-3 * h**2, 2.0 * (1.0 + 0.5 * h**2), 4.0 * (1.0 + 0.5 * h**2)])
        for h in spacings
    ]
    with pytest.raises(AssertionError, match="zero-valued oracle entries"):
        assert_converges(bad, mixed_oracle, spacings=spacings, order=2.0)

    zero_oracle = np.zeros(3)
    zero_ladder = [np.full(3, 0.5 * h**2) for h in spacings]
    assert_converges(zero_ladder, zero_oracle, spacings=spacings, order=2.0)


def test_assert_converges_validates_inputs_shapes_dtypes_and_finiteness() -> None:
    spacings = (0.4, 0.2, 0.1)
    oracle = np.array([1.0, 2.0])
    ladder = [oracle * (1.0 + 0.5 * h**2) for h in spacings]

    with pytest.raises(ValueError, match="strictly decreasing"):
        assert_converges(ladder, oracle, spacings=(0.2, 0.2, 0.1), order=2.0)

    with pytest.raises(ValueError, match="at least 3 levels"):
        assert_converges(ladder[:2], oracle, spacings=spacings[:2], order=2.0)

    with pytest.raises(AssertionError):
        assert_converges(
            [ladder[0], np.array([1.0, 2.0, 3.0]), ladder[2]],
            oracle,
            spacings=spacings,
            order=2.0,
        )

    f32_ladder = [arr.astype(np.float32) for arr in ladder]
    with pytest.raises(AssertionError):
        assert_converges(f32_ladder, oracle, spacings=spacings, order=2.0)
    assert_converges(
        f32_ladder, oracle, spacings=spacings, order=2.0, promote_dtypes=True
    )

    with pytest.raises(AssertionError, match="Non-finite"):
        assert_converges(
            [ladder[0], np.array([np.nan, 2.0]), ladder[2]],
            oracle,
            spacings=spacings,
            order=2.0,
        )
    with pytest.raises(AssertionError, match="Non-finite"):
        assert_converges(ladder, np.array([np.inf, 2.0]), spacings=spacings, order=2.0)
