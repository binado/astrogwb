# Tests to study for ineffective tolerances

These tests remain in the suite. This is a source review, not a report of
observed failures or measured numerical errors. The suggested changes below
need a separate decision about the precision each scientific interface promises.

## Correction to the initial audit

The initial audit incorrectly attributed NumPy's `allclose` default absolute
tolerance to `numpy.testing.assert_allclose`. The waveform and forward model
comparisons listed below do **not** accept zero against nonzero power merely
because their powers are tiny. That removal rationale is withdrawn.

| Helper | Default relative tolerance | Default absolute tolerance | Comparison |
| --- | --- | --- | --- |
| `np.testing.assert_allclose` | `1e-7` | `0` | `abs(actual - expected) <= atol + rtol * abs(expected)` |
| `np.allclose` / `np.isclose` | `1e-5` | `1e-8` | Same sum of absolute and relative allowances |
| `pytest.approx` | `1e-6` | `1e-12` | Either allowance may accept the comparison |

For `pytest.approx`, setting only `rel` leaves the default absolute allowance
enabled. Setting only `abs` disables the default relative allowance. For
`assert_allclose`, setting only `atol` leaves the default relative allowance
enabled. These distinctions explain the confirmed findings below.

Sources: [NumPy assertion documentation](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html),
[NumPy comparison documentation](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html),
and [pytest tolerance documentation](https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx).

## Confirmed tolerance masking

An ineffective tolerance here means that another allowance permits materially
larger errors than the small explicit tolerance suggests. It does not mean the
entire test is useless, or that its production function is incorrect.

### `tests/core/test_constants.py`

| Test | Current comparison | Why it needs study |
| --- | --- | --- |
| `test_lal_solar_mass_triple_is_self_consistent` | Three `pytest.approx(..., rel=1e-16)` comparisons | The docstring promises float64 roundoff. For solar mass in seconds, about `4.93e-6`, the relative allowance is about `4.93e-22`, but the absolute floor is `1e-12`: an effective relative allowance around `2e-7`. For solar mass in meters, about `1477`, the relative allowance is about `1.48e-13`, still below that floor. |
| `test_isco_alpha_is_dimensionless` | First assertion: `pytest.approx(1 / (pi * 6**1.5), rel=1e-15)` | The expected value is about `0.02166`. Its relative allowance is about `2.17e-17`, while the absolute floor is `1e-12`, giving an effective relative allowance around `4.6e-11`. The bounds and the separate approximate `4397.2` frequency check serve different purposes. |

Study whether these identities should use relative comparisons with `abs=0`,
an explicit allowance for rounding, or equality where exact equality is actually
part of the contract. A tabulated constant's accuracy is a separate question
from arithmetic rounding.

### `tests/core/test_utils.py`

| Test | Current comparison | Why it needs study |
| --- | --- | --- |
| `test_cumulative_trapezoid_is_exact_for_linear_samples` | `assert_allclose(..., atol=2e-15)` | The default `rtol=1e-7` remains active. Nonzero expected values are `6`, `24`, and `66`, so permitted errors reach `6.6e-6`, despite the word "exact" and the small absolute allowance. |
| `test_cumulative_trapezoid_broadcasts_over_leading_batch_dimensions` | Row comparisons: `assert_allclose(..., atol=2e-15)` | Nonzero values are approximately order one, so the default relative allowance dominates. The separate first-column comparison against zero has `atol=0` and is exact; that assertion is effective. |
| `test_cumulative_trapezoid_is_jittable_and_differentiable` | Two `pytest.approx(..., rel=2e-15)` comparisons | Expected values are `43.75` and `21.875`; their relative allowances are `8.75e-14` and `4.375e-14`, both below the `1e-12` absolute floor. |
| `test_mapped_rule_is_exact_for_the_highest_representable_degree` | `pytest.approx(expected, rel=2e-14)` for orders `2`, `4`, and `8` | At order `2`, the expected value is `20.234375`, whose relative allowance is about `4.05e-13`, below the absolute floor. This finding concerns only the order `2` case; the other two cases have larger relative allowances. |
| `test_mapped_rule_covers_every_interval_of_a_grid` | `pytest.approx(exp(2) - 1, rel=2e-14)` | Expected value is about `6.389`; the relative allowance is about `1.28e-13`, below the absolute floor. |
| `test_mapped_rule_is_jittable_and_differentiable_in_its_bounds` | Two `pytest.approx(9, rel=2e-15)` comparisons | Both relative allowances are `1.8e-14`, below the absolute floor. |

For NumPy row comparisons, decide whether `rtol=0` is appropriate for the
claimed absolute precision. For pytest comparisons, decide whether `abs=0`
or an explicit absolute allowance expresses the intended precision. Tightening
these settings requires studying dtype and rounding across supported platforms.

## Original candidates retained after correcting the audit

These are an index for further study, **not confirmed ineffective tolerance
tests**. Their omitted `atol` is already zero. In particular, comparisons
against expected zero enforce zero, rather than accepting small nonzero power.

### `tests/core/test_waveform_analytical.py`

All the following comparisons explicitly use `rtol=1e-12`:

- `test_scales_as_inverse_distance_squared`
- `test_scales_as_chirp_mass_to_the_five_thirds`
- `test_scales_as_frequency_to_the_minus_seven_thirds`
- `test_scales_with_inclination_factor`
- `test_missing_inclination_defaults_to_face_on`
- `test_redshift_enters_only_through_detector_frame_masses`
- `test_scalar_source_parameters_match_length_one_arrays`
- `test_extra_gwmock_parameters_are_ignored`
- `test_scalar_and_vector_source_parameters_broadcast_together`
- `test_batched_sources_match_stacked_single_source_calls`
- `test_jit_matches_eager_evaluation`

Possible study questions:

- Ratio checks cannot establish an absolute amplitude normalization: a common
  wrong factor on both sides cancels. An identically zero implementation also
  satisfies a ratio check in isolation. The retained
  `test_power_is_zero_above_the_termination_frequency` asserts positive power
  below the cutoff and rejects globally zero output, so this is a limitation
  of individual assertions rather than an uncovered global-zero failure.
- Default `equal_nan=True` permits NaNs when both sides have them. Whether to
  require finite output explicitly depends on the contract and coverage from
  other tests.
- The `1e-12` requirement needs a numerical justification before describing it
  as brittle or changing it. No platform failure has been established here.

### `tests/core/test_forward_model.py`

Spectral comparisons to revisit from the original audit, all with explicit
`rtol=1e-12` and default `atol=0`:

- `test_spectrum_matches_the_sum_of_per_source_power_over_time`
- `test_batched_power_matches_a_single_generator_call`
- `test_batch_size_does_not_change_the_spectrum`
- `test_missing_inclination_rescales_face_on_power`
- `test_returned_inclination_disables_analytic_rescaling`
- `test_isotropic_inclination_messenger_disables_analytic_rescaling`
- `test_unobserved_poisson_count_is_jittable_and_keeps_static_shapes`
- `test_partial_count_masks_power_but_not_source_capacity`
- `test_count_above_capacity_is_silently_capped`
- `test_jitted_spectrum_matches_eager`
- `test_ripple_spectrum_matches_the_sum_of_per_source_power_over_time`
- `test_ripple_batched_power_matches_a_single_generator_call`
- `test_ripple_batch_size_does_not_change_the_spectrum`
- `test_jitted_scan_matches_eager_for_full_and_ragged_catalogs`
- `test_vmap_over_draws_shares_one_static_event_count`
- `test_ripple_forward_model_is_jittable`

The returned-inclination test also compares JIT and eager results using the
default `rtol=1e-7`. That is a separate precision choice, not an absolute floor
that swamps a small spectrum. Further study should distinguish independent
physical expectations from expected values generated through shared production
code, and justify tolerances for batched reductions and JIT evaluation.

### `tests/core/test_spectra_cache.py`

- `test_batch_size_does_not_change_the_draws` uses `assert_allclose` with its
  default `rtol=1e-7, atol=0`. It rejects zero against nonzero reference power.
  Study whether `1e-7` is the intended precision for batch invariance; there is
  no demonstrated ineffective absolute tolerance here.

### Comparisons already explicit about zero absolute tolerance

- `tests/core/test_waveform_generator.py::test_power_matches_the_gwmock_backend`
  uses `rtol=1e-11, atol=0`.
- `tests/core/test_waveform_generator.py::test_generate_batch_is_traceable_and_values_do_not_recompile`
  and `test_generate_is_the_length_one_batch` use `rtol=1e-13, atol=0`.
- `tests/core/test_gwb_analytic.py` explicitly sets `atol=0` in its numerical
  parity checks.

These are not candidates for removal merely because their values are tiny.
