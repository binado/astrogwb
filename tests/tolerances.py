"""Dtype-aware default tolerances for array comparisons in tests.

The defaults follow scipy's ``xp_assert_close``: the relative tolerance is
``sqrt(eps) * 4`` for the inexact dtype under test and the absolute tolerance
is zero, so a comparison cannot pass only because the arrays are small. The
tolerance may be a number or a ``{dtype: value}`` mapping, as in
``jax.test_util.tolerance``, and the less precise of the two dtypes governs.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

type Tolerance = float | Mapping[type | np.dtype, float] | None


def default_rtol(dtype: np.dtype) -> float:
    """Return the default relative tolerance for ``dtype``.

    Parameters
    ----------
    dtype
        Dtype of one of the compared arrays.

    Returns
    -------
    float
        ``sqrt(eps) * 4`` for floating and complex dtypes, and zero for the
        rest, which must match exactly.
    """
    if dtype.kind in "fc":
        return float(np.finfo(dtype).eps) ** 0.5 * 4
    return 0.0


def tolerance(dtype: np.dtype, tol: Tolerance = None) -> float:
    """Resolve a tolerance for ``dtype``.

    Parameters
    ----------
    dtype
        Dtype of one of the compared arrays.
    tol
        A number used for every dtype, a mapping from dtype to value, or
        ``None`` for the default. A dtype missing from the mapping gets the
        default.

    Returns
    -------
    float
        The relative tolerance to use for ``dtype``.
    """
    if tol is None:
        return default_rtol(dtype)
    if isinstance(tol, Mapping):
        table = {np.dtype(key): value for key, value in tol.items()}
        return table.get(dtype, default_rtol(dtype))
    return tol


def assert_close(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: Tolerance = None,
    atol: float = 0.0,
    promote_dtypes: bool = False,
) -> None:
    """Assert that two arrays have the same shape and dtype and are close.

    Parameters
    ----------
    actual, desired
        Arrays to compare.
    rtol
        Relative tolerance as accepted by :func:`tolerance`. When the dtypes
        differ the larger resolved value is used. Omit it to use the dtype
        default.
    atol
        Absolute tolerance. It is zero unless a test asks for one.
    promote_dtypes
        Compare after promoting both arrays to their common dtype instead of
        requiring equal dtypes. Use it against references stored at lower
        precision, such as LAL's float32 response tensors. The tolerance still
        comes from the original dtypes, so the less precise array governs.

    Raises
    ------
    AssertionError
        If the shapes differ, the dtypes differ and ``promote_dtypes`` is
        false, or the values are not close.
    """
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert actual.shape == desired.shape, (actual.shape, desired.shape)
    rtol = max(tolerance(actual.dtype, rtol), tolerance(desired.dtype, rtol))
    if promote_dtypes:
        common = np.result_type(actual, desired)
        actual, desired = actual.astype(common), desired.astype(common)
    assert actual.dtype == desired.dtype, (actual.dtype, desired.dtype)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def _relative_rms(diff: np.ndarray, scale: np.ndarray) -> float:
    """Compute the root-mean-square magnitude of ``diff / scale``."""
    return float(np.sqrt(np.mean(np.abs(diff / scale) ** 2)))


def _error_ladder(
    arrays: Sequence[np.ndarray],
    target: np.ndarray | None,
    steps: np.ndarray,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative RMS errors and matching step sizes for ``assert_converges``."""
    if target is None:
        scale = np.abs(arrays[-1])
        zero = scale == 0.0
        diffs = [arrays[i] - arrays[i + 1] for i in range(len(arrays) - 1)]
        if np.all(zero):
            errors = [float(np.sqrt(np.mean(np.abs(d) ** 2))) for d in diffs]
        else:
            errors = [_relative_rms(d[~zero], scale[~zero]) for d in diffs]
        return np.array(errors, dtype=np.float64), steps[:-1]

    scale = np.abs(target)
    zero = scale == 0.0
    if np.all(zero):
        errors = [float(np.sqrt(np.mean(np.abs(a) ** 2))) for a in arrays]
        return np.array(errors, dtype=np.float64), steps

    if np.any(zero):
        exact_mask = np.abs(arrays[-1][zero]) <= atol
        assert np.all(exact_mask), (
            "Finest approximation exceeded atol at zero-valued oracle entries: "
            f"max abs={float(np.max(np.abs(arrays[-1][zero]))):.3e}, atol={atol}"
        )
    errors = [_relative_rms(a[~zero] - target[~zero], scale[~zero]) for a in arrays]
    return np.array(errors, dtype=np.float64), steps


def assert_converges(
    approximations: Sequence[ArrayLike],
    desired: ArrayLike | None = None,
    *,
    spacings: Sequence[float],
    order: float,
    order_rtol: float = 0.25,
    rtol: Tolerance = None,
    atol: float = 0.0,
    promote_dtypes: bool = False,
) -> None:
    """Assert that a grid-refined sequence converges at the expected order.

    Parameters
    ----------
    approximations
        Sequence of at least three array approximations ordered from coarsest
        to finest grid spacing.
    desired
        Optional exact reference array. When provided, relative RMS errors are
        measured against ``desired``; when ``None``, successive differences
        between adjacent refinement levels are used instead.
    spacings
        Strictly decreasing positive grid spacings aligned with
        ``approximations``.
    order
        Expected algebraic convergence rate :math:`p` such that error scales
        as :math:`O(h^p)`.
    order_rtol
        Relative slack below ``order`` allowed on the fitted log-log slope,
        in ``[0, 1)``.
    rtol
        Relative tolerance as accepted by :func:`tolerance` for short-circuiting
        when the finest approximation already matches ``desired``.
    atol
        Absolute tolerance used alongside ``rtol`` for short-circuiting and for
        exact-zero entries of ``desired``.
    promote_dtypes
        Compare after promoting all arrays to their common dtype instead of
        requiring equal dtypes. The short-circuit tolerance still comes from
        the original dtypes.

    Raises
    ------
    ValueError
        If ``approximations``, ``spacings``, ``order``, or ``order_rtol`` do
        not form a valid refinement ladder.
    AssertionError
        If shapes, finiteness, or dtypes fail validation, or if the error does
        not decrease at least at ``order * (1 - order_rtol)``.
    """
    arrays = [np.asarray(item) for item in approximations]
    steps = np.asarray(spacings, dtype=np.float64)
    if len(arrays) < 3:
        raise ValueError(
            f"assert_converges requires at least 3 levels; got {len(arrays)}"
        )
    if steps.ndim != 1 or steps.size != len(arrays):
        raise ValueError(
            f"spacings must be 1D with length {len(arrays)}; got shape {steps.shape}"
        )
    if (
        not np.all(np.isfinite(steps))
        or not np.all(steps > 0.0)
        or not np.all(np.diff(steps) < 0.0)
    ):
        raise ValueError(
            f"spacings must be finite, positive, and strictly decreasing; got {steps.tolist()}"
        )
    if not np.isfinite(order) or order <= 0.0:
        raise ValueError(f"order must be finite and positive; got {order}")
    if not np.isfinite(order_rtol) or not (0.0 <= order_rtol < 1.0):
        raise ValueError(f"order_rtol must lie in [0, 1); got {order_rtol}")

    expected_shape = arrays[0].shape
    for idx, arr in enumerate(arrays):
        assert arr.shape == expected_shape, (idx, arr.shape, expected_shape)
        assert np.all(np.isfinite(arr)), f"Non-finite values at level {idx}"

    target: np.ndarray | None = None
    if desired is not None:
        target = np.asarray(desired)
        assert target.shape == expected_shape, (target.shape, expected_shape)
        assert np.all(np.isfinite(target)), "Non-finite values in desired reference"
        resolved_rtol = max(
            tolerance(target.dtype, rtol),
            *(tolerance(arr.dtype, rtol) for arr in arrays),
        )
        if promote_dtypes:
            common = np.result_type(*(arr.dtype for arr in arrays), target.dtype)
            arrays = [arr.astype(common) for arr in arrays]
            target = target.astype(common)
        for idx, arr in enumerate(arrays):
            assert arr.dtype == target.dtype, (idx, arr.dtype, target.dtype)
        if np.allclose(arrays[-1], target, rtol=resolved_rtol, atol=atol):
            return
    else:
        if promote_dtypes:
            common = np.result_type(*(arr.dtype for arr in arrays))
            arrays = [arr.astype(common) for arr in arrays]
        for idx, arr in enumerate(arrays[1:], start=1):
            assert arr.dtype == arrays[0].dtype, (idx, arr.dtype, arrays[0].dtype)

    errors, error_steps = _error_ladder(arrays, target, steps, atol)
    assert np.all(errors > 0.0), (
        f"Non-positive convergence errors {errors.tolist()} "
        f"at spacings {error_steps.tolist()}"
    )
    assert errors[-1] < errors[-2], (
        f"Finest error did not decrease: errors={errors.tolist()}, "
        f"spacings={error_steps.tolist()}"
    )
    log_h = np.log(error_steps)
    log_e = np.log(errors)
    centered_h = log_h - np.mean(log_h)
    fitted_order = float(
        np.dot(centered_h, log_e - np.mean(log_e)) / np.dot(centered_h, centered_h)
    )
    min_order = order * (1.0 - order_rtol)
    assert fitted_order >= min_order, (
        f"Observed convergence order {fitted_order:.3f} < {min_order:.3f} "
        f"(expected {order} with order_rtol={order_rtol}); "
        f"spacings={error_steps.tolist()}, errors={errors.tolist()}"
    )
