"""Tests for the NumPyro-native population distributions.

The physics is checked against
:func:`reference_population.reference_merger_rate_distance_and_logprob`, a
hand-written restatement of the same redshift density kept as a test oracle so
the class-based path is compared against something other than itself. The rest
of the module is about JAX plumbing: these classes are auto-registered as
pytrees, and a wrong ``pytree_data_fields`` is invisible to ``ruff``, to ``ty``
and to any test that builds the distribution *inside* a model function.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

# The mock population's fiducials and grid, shared with `test_importance.py`:
# the two tolerance-tight agreement tests below are only licensed because both
# implementations run on the *same* grid, so a second copy of these would let
# them drift apart with no visible symptom.
from astrogwb_mock_population import FIDUCIALS, N_GRID, Z_MAX, Z_MIN, make_redshift_grid
from jax.typing import ArrayLike
from numpyro.distributions.transforms import biject_to
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.distributions.interpolated import InterpolatedDistribution
from astrogwb.distributions.mass import MaxOfTwoNormalsDistribution
from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)

#: Redshifts to evaluate at: interior to the grid, and not on a node.
SAMPLE_REDSHIFTS = jnp.array([0.5, 1.234, 3.7, 12.0, 19.5])


def _distribution(**overrides: float) -> RedshiftDistribution:
    """The Madau-Dickinson specimen at the mock fiducials, on the mock grid."""
    return MadauDickinsonRedshiftDistribution(
        params={**FIDUCIALS, **overrides},
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
    )


def _reference(**overrides: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(total_merger_rate, luminosity_distance, logpdf)`` from the reference model."""
    return reference_merger_rate_distance_and_logprob(
        {**FIDUCIALS, **overrides},
        SAMPLE_REDSHIFTS,
        redshift_grid=make_redshift_grid(),
    )


# --------------------------------------------------------------------------- #
# InterpolatedDistribution
# --------------------------------------------------------------------------- #
def test_interpolated_support_is_the_table_span() -> None:
    x = jnp.linspace(-2.0, 5.0, 64)
    interpolated = InterpolatedDistribution(x, jnp.exp(-(x**2)))

    assert float(interpolated.support.lower_bound) == -2.0
    assert float(interpolated.support.upper_bound) == 5.0


def test_interpolated_normalizes_an_unnormalized_table() -> None:
    """`norm` is the trapezoid integral, and dividing by it is the whole density."""
    x = jnp.linspace(0.0, 1.0, 129)
    interpolated = InterpolatedDistribution(x, 7.5 * (1.0 + x))

    np.testing.assert_allclose(float(interpolated.norm), 7.5 * 1.5, rtol=1e-12)
    density = np.exp(np.asarray(interpolated.log_prob(x)))
    np.testing.assert_allclose(
        float(np.trapezoid(density, np.asarray(x))), 1.0, rtol=1e-12
    )


@pytest.mark.parametrize("value", [0.9, 1.5], ids=["zero-table", "off-table"])
def test_interpolated_log_prob_where_density_vanishes_has_zero_gradient(
    value: float,
) -> None:
    """A vanishing density is ``-inf`` with a zero, not NaN, derivative.

    A delayed merger rate is exactly zero next to its formation cut-off, and
    one importance sample there must not turn the gradient of a weight sum
    into NaN.
    """
    x = jnp.linspace(0.0, 1.0, 11)

    def log_prob(scale: jax.Array) -> jax.Array:
        table = jnp.where(x < 0.8, scale * (1.0 + x), 0.0)
        return InterpolatedDistribution(x, table).log_prob(value)

    assert float(log_prob(jnp.asarray(2.0))) == -math.inf
    assert float(jax.grad(log_prob)(jnp.asarray(2.0))) == 0.0


# --------------------------------------------------------------------------- #
# The Madau-Dickinson density against the reference model
# --------------------------------------------------------------------------- #
def test_grid_is_bit_identical_to_the_shared_redshift_grid() -> None:
    """What licenses the two tolerance-tight agreement tests below."""
    np.testing.assert_array_equal(
        np.asarray(_distribution().x), np.asarray(make_redshift_grid())
    )


def test_log_prob_agrees_with_the_reference_model() -> None:
    # Bit-exact, not a tolerance: `log_prob` forms `log(u) - log(Z)` in the
    # reference's operation order precisely so a catalog that is its own
    # proposal has exactly zero log-weight (see `test_frequency_resolution.py`).
    _, _, reference_logpdf = _reference()
    np.testing.assert_array_equal(
        np.asarray(_distribution().log_prob(SAMPLE_REDSHIFTS)),
        np.asarray(reference_logpdf),
    )


def test_log_prob_agrees_with_the_reference_model_off_the_fiducials() -> None:
    """The agreement is in the formula, not in a coincidence at one parameter point."""
    overrides = {"gamma": 2.7, "kappa": 2.9, "z_peak": 1.9, "H0": 74.0, "Omega_m": 0.27}
    _, _, reference_logpdf = _reference(**overrides)
    np.testing.assert_array_equal(
        np.asarray(_distribution(**overrides).log_prob(SAMPLE_REDSHIFTS)),
        np.asarray(reference_logpdf),
    )


def test_total_merger_rate_is_in_mergers_per_second() -> None:
    """The `1e-9 / SECONDS_PER_YEAR` conversion, bit-identical given the factor order."""
    reference_rate, _, _ = _reference()
    np.testing.assert_allclose(
        float(_distribution().total_merger_rate()),
        float(reference_rate),
        rtol=1e-15,
    )


def test_luminosity_distance_matches_the_reference_model() -> None:
    _, reference_distance, _ = _reference()
    np.testing.assert_allclose(
        np.asarray(_distribution().luminosity_distance(SAMPLE_REDSHIFTS)),
        np.asarray(reference_distance),
        rtol=1e-15,
    )


def test_source_frame_distribution_is_the_rate() -> None:
    distribution = _distribution()
    np.testing.assert_array_equal(
        np.asarray(distribution.source_frame_distribution(SAMPLE_REDSHIFTS, FIDUCIALS)),
        np.asarray(
            madau_dickinson_rate(
                SAMPLE_REDSHIFTS,
                FIDUCIALS["gamma"],
                FIDUCIALS["kappa"],
                FIDUCIALS["z_peak"],
                FIDUCIALS["local_merger_rate"],
            )
        ),
    )


# --------------------------------------------------------------------------- #
# Normalization, CDF and sampling
# --------------------------------------------------------------------------- #
def test_density_integrates_to_unity_on_its_own_grid() -> None:
    """Interpolating the *normalized* table makes the interpolant's own integral
    exactly the normalization."""
    distribution = _distribution()
    grid = np.asarray(distribution.x)
    density = np.exp(np.asarray(distribution.log_prob(distribution.x)))
    np.testing.assert_allclose(float(np.trapezoid(density, grid)), 1.0, rtol=1e-12)


def test_log_prob_is_negative_infinity_off_the_table() -> None:
    """The `left=0.0, right=0.0` contract: no extrapolation, and no exception --
    the importance weights depend on `-inf`."""
    outside = jnp.array([Z_MIN - 0.1, Z_MAX + 0.1])
    assert bool(jnp.all(jnp.isneginf(_distribution().log_prob(outside))))


def test_cdf_grid_spans_zero_to_one_exactly() -> None:
    """The renormalizing divide: `cumsum`-order and `sum`-order reductions do not
    agree to the last bit, so the endpoint is made exact by construction."""
    cdf_grid = _distribution().cdf_grid
    assert float(cdf_grid[0]) == 0.0
    assert float(cdf_grid[-1]) == 1.0
    assert bool(jnp.all(jnp.diff(cdf_grid) > 0.0))


def test_icdf_inverts_the_tabulated_cdf() -> None:
    distribution = _distribution()
    np.testing.assert_allclose(
        np.asarray(distribution.icdf(distribution.cdf_grid)),
        np.asarray(distribution.x),
        rtol=1e-12,
    )


def test_icdf_endpoints_return_the_table_edges() -> None:
    distribution = _distribution()
    assert float(distribution.icdf(jnp.array(0.0))) == float(distribution.x[0])
    assert float(distribution.icdf(jnp.array(1.0))) == float(distribution.x[-1])


def test_sample_is_the_inverse_cdf_of_uniform_draws() -> None:
    """The sampler contract, pinned exactly -- no statistical tolerance at all."""
    distribution = _distribution()
    key = jax.random.PRNGKey(0)

    np.testing.assert_array_equal(
        np.asarray(distribution.sample(key, (1024,))),
        np.asarray(distribution.icdf(jax.random.uniform(key, (1024,)))),
    )


def test_sample_mean_matches_the_density_it_reports() -> None:
    """Inverse-transform draws must follow the density `log_prob` publishes."""
    distribution = _distribution()
    grid = np.asarray(distribution.x)
    density = np.exp(np.asarray(distribution.log_prob(distribution.x)))
    expected = float(np.trapezoid(grid * density, grid))

    draws = np.asarray(distribution.sample(jax.random.PRNGKey(0), (400_000,)))

    # A fixed `rtol` would be unsound here: the Monte Carlo standard error alone
    # is ~0.09% of the mean, so `rtol=2e-3` is a 2.3-sigma assertion -- a coin
    # flip against any change to the grid, the sampler, or the fiducials. Five
    # standard errors *of these draws* is the honest band, and it scales with
    # the sample size instead of silently going stale.
    standard_error = float(draws.std(ddof=1)) / np.sqrt(draws.size)
    assert abs(float(draws.mean()) - expected) < 5.0 * standard_error


def test_support_bijector_round_trips() -> None:
    """`support` used to be `None`, which fails inside NUTS with
    `'NoneType' object has no attribute 'is_discrete'`."""
    transform = biject_to(_distribution().support)
    unconstrained = jnp.array(0.37)
    constrained = transform(unconstrained)

    assert Z_MIN < float(constrained) < Z_MAX
    np.testing.assert_allclose(
        float(transform.inv(constrained)), float(unconstrained), rtol=1e-10
    )


# --------------------------------------------------------------------------- #
# The distributions as pytrees
# --------------------------------------------------------------------------- #

#: The only field that changes when `gamma` moves: the cosmology grids depend on
#: `H0`/`Omega_m` alone, and `x` is fixed by the window.
_MAPPED_FIELDS = ("y",)


def _stack_over_y(
    distributions: list[RedshiftDistribution],
) -> RedshiftDistribution:
    """Hand-stack a batch of specimens along `y`, leaving every other leaf shared."""
    cls = RedshiftDistribution
    fields = cls.gather_pytree_data_fields()
    children, aux = cls.tree_flatten(distributions[0])
    stacked = tuple(
        jnp.stack([cls.tree_flatten(d)[0][index] for d in distributions])
        if field in _MAPPED_FIELDS
        else child
        for index, (field, child) in enumerate(zip(fields, children, strict=True))
    )
    # `Distribution.tree_unflatten` is annotated as returning the base class,
    # but it constructs `cls`; that is what the round-trip test below pins.
    return cast("RedshiftDistribution", cls.tree_unflatten(aux, stacked))


def test_distribution_survives_jit_as_a_pytree_argument() -> None:
    """Without `pytree_data_fields` every array is dropped on flatten and this
    raises `AttributeError: ... has no attribute 'x'`."""
    distribution = _distribution()
    jitted = jax.jit(lambda d, z: d.log_prob(z))(distribution, SAMPLE_REDSHIFTS)
    np.testing.assert_allclose(
        np.asarray(jitted),
        np.asarray(distribution.log_prob(SAMPLE_REDSHIFTS)),
        rtol=1e-15,
    )


def test_distribution_vmaps_over_a_hand_stacked_pytree() -> None:
    """Vary `gamma`, never `H0`: the normalized pdf is H0-independent, so a vmap
    over `H0` returns byte-identical rows and would pass with the pytree broken."""
    gammas = (1.0, 2.0, 3.0)
    distributions = [_distribution(gamma=gamma) for gamma in gammas]
    stacked = _stack_over_y(distributions)

    cls = RedshiftDistribution
    fields = cls.gather_pytree_data_fields()
    aux = cls.tree_flatten(distributions[0])[1]
    # The gathered field order is a set iteration order, so build the specimen
    # by field name rather than positionally.
    in_axes = cls.tree_unflatten(
        aux, tuple(0 if field in _MAPPED_FIELDS else None for field in fields)
    )

    mapped = jax.vmap(lambda d: d.log_prob(SAMPLE_REDSHIFTS), in_axes=(in_axes,))(
        stacked
    )
    assert mapped.shape == (len(gammas), SAMPLE_REDSHIFTS.shape[0])

    for row, distribution in zip(mapped, distributions, strict=True):
        np.testing.assert_allclose(
            np.asarray(row),
            np.asarray(distribution.log_prob(SAMPLE_REDSHIFTS)),
            rtol=1e-15,
        )
    # Three *distinct* rows: identical rows would pass with the pytree broken.
    assert not np.allclose(np.asarray(mapped[0]), np.asarray(mapped[1]))
    assert not np.allclose(np.asarray(mapped[1]), np.asarray(mapped[2]))


def test_distribution_vmaps_over_the_constructor() -> None:
    gammas = jnp.array([1.0, 2.0, 3.0])
    mapped = jax.vmap(
        lambda gamma: _distribution(gamma=gamma).log_prob(SAMPLE_REDSHIFTS)
    )(gammas)

    for row, gamma in zip(mapped, gammas, strict=True):
        np.testing.assert_allclose(
            np.asarray(row),
            np.asarray(_distribution(gamma=float(gamma)).log_prob(SAMPLE_REDSHIFTS)),
            rtol=1e-15,
        )


def test_lazy_fields_are_not_pytree_leaves() -> None:
    """A lazily-materialized field listed as pytree data flattens to `None` before
    first access and to an array afterwards, so the treedef would change under
    the object and every `jax.jit` taking it would retrace."""
    distribution = _distribution()
    jitted = jax.jit(lambda d: d.log_prob(SAMPLE_REDSHIFTS))

    assert len(jax.tree.leaves(distribution)) == 4
    jitted(distribution)
    assert jitted._cache_size() == 1  # ty: ignore[unresolved-attribute]

    # Force every lazy_property to materialize onto the instance.
    _ = distribution.norm, distribution.normalized_y, distribution.cdf_grid

    assert len(jax.tree.leaves(distribution)) == 4
    jitted(distribution)
    assert jitted._cache_size() == 1  # ty: ignore[unresolved-attribute]


def test_aux_data_is_hashable_and_stable() -> None:
    """Aux data is hashed into the jit cache key, so it must not carry arrays."""
    distribution = _distribution()
    cls = RedshiftDistribution
    aux = cls.tree_flatten(distribution)[1]

    assert hash(aux) == hash(cls.tree_flatten(distribution)[1])


def test_redshift_distribution_is_registered_as_a_pytree() -> None:
    """The concrete distribution remains registered as a NumPyro pytree node."""
    distribution = _distribution()
    round_tripped = jax.tree.unflatten(
        jax.tree.structure(distribution), jax.tree.leaves(distribution)
    )

    assert type(round_tripped) is RedshiftDistribution
    assert (
        round_tripped._source_frame_distribution
        is distribution._source_frame_distribution
    )
    np.testing.assert_array_equal(
        np.asarray(round_tripped.log_prob(SAMPLE_REDSHIFTS)),
        np.asarray(distribution.log_prob(SAMPLE_REDSHIFTS)),
    )


def test_validate_args_constructs_and_round_trips() -> None:
    """Forwarding works. The flag is otherwise inert here -- no `arg_constraints`
    and no `@validate_sample` -- and is accepted for API uniformity."""
    validated = MadauDickinsonRedshiftDistribution(
        params=FIDUCIALS,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
        validate_args=True,
    )

    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda d: d.log_prob(SAMPLE_REDSHIFTS))(validated)),
        np.asarray(_distribution().log_prob(SAMPLE_REDSHIFTS)),
        rtol=1e-15,
    )


# --------------------------------------------------------------------------- #
# Window accessors, derived rather than stored
# --------------------------------------------------------------------------- #
def test_redshift_grid_aliases_x_rather_than_copying_it() -> None:
    """One array, one leaf: a second attribute would be a second pytree leaf."""
    distribution = _distribution()
    assert distribution.redshift_grid is distribution.x


def test_window_is_read_back_off_the_grid() -> None:
    """The deleted `pytree_aux_fields` named three attributes `__init__` never
    assigned, so they flattened to `(None, None, None)`."""
    distribution = _distribution()

    assert float(distribution.minimum_redshift) == Z_MIN
    assert float(distribution.maximum_redshift) == Z_MAX
    assert distribution.n_grid == N_GRID
    assert isinstance(distribution.n_grid, int)


# --------------------------------------------------------------------------- #
# Callable source-frame rate law
# --------------------------------------------------------------------------- #
def _doubled_madau_dickinson_rate(
    redshift: ArrayLike, params: Mapping[str, ArrayLike]
) -> jax.Array:
    return 2.0 * madau_dickinson_rate(
        redshift,
        params["gamma"],
        params["kappa"],
        params["z_peak"],
    )


def test_madau_dickinson_factory_returns_redshift_distribution() -> None:
    assert type(_distribution()) is RedshiftDistribution


def test_source_frame_distribution_is_required() -> None:
    with pytest.raises(TypeError, match="source_frame_distribution"):
        RedshiftDistribution(params=FIDUCIALS)


def test_redshift_distribution_uses_supplied_source_frame_callable() -> None:
    distribution = RedshiftDistribution(
        params=FIDUCIALS,
        source_frame_distribution=_doubled_madau_dickinson_rate,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        n_grid=N_GRID,
    )

    np.testing.assert_array_equal(
        np.asarray(distribution.source_frame_distribution(SAMPLE_REDSHIFTS, FIDUCIALS)),
        np.asarray(
            2.0
            * madau_dickinson_rate(
                SAMPLE_REDSHIFTS,
                FIDUCIALS["gamma"],
                FIDUCIALS["kappa"],
                FIDUCIALS["z_peak"],
            )
        ),
    )


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_normalized_density_is_independent_of_the_hubble_constant() -> None:
    """`dV_c/dz` is proportional to `H0^-3`, which cancels in the normalization --
    the fact behind `merger_rate_H0_fn = H0**-3`. Measured -9e-19: float noise,
    not an exact zero, so this is an `atol` assertion."""

    def log_prob_at(name: str, value: float) -> jax.Array:
        return _distribution(**{name: value}).log_prob(jnp.array(1.234))

    d_h0 = jax.grad(lambda h0: log_prob_at("H0", h0))(FIDUCIALS["H0"])
    d_omega_m = jax.grad(lambda om: log_prob_at("Omega_m", om))(FIDUCIALS["Omega_m"])

    assert abs(float(d_h0)) < 1e-12
    # The scale the H0 derivative is small *relative to*: the same density does
    # respond to the other cosmology parameter.
    assert abs(float(d_omega_m)) > 1e-3


def test_total_merger_rate_scales_as_the_inverse_cube_of_the_hubble_constant() -> None:
    """Pins `merger_rate_H0_fn = H0**-3` against the class itself."""

    def rate_at(h0: float) -> jax.Array:
        return _distribution(H0=h0).total_merger_rate()

    h0 = FIDUCIALS["H0"]
    np.testing.assert_allclose(
        float(jax.grad(rate_at)(h0)), -3.0 * float(rate_at(h0)) / h0, rtol=1e-9
    )


# --------------------------------------------------------------------------- #
# MaxOfTwoNormalsDistribution
# --------------------------------------------------------------------------- #
_MASS_MEAN = 1.33
_MASS_SIGMA = 0.09
_MASS_VALUES = jnp.array([1.1, 1.33, 1.5])
_MASS_QUANTILES = jnp.array([0.1, 0.5, 0.9])


def _max_of_two_normals() -> MaxOfTwoNormalsDistribution:
    return MaxOfTwoNormalsDistribution(_MASS_MEAN, _MASS_SIGMA, validate_args=True)


def test_max_of_two_normals_log_prob_matches_the_closed_form() -> None:
    distribution = _max_of_two_normals()
    component = dist.Normal(_MASS_MEAN, _MASS_SIGMA)
    expected = (
        jnp.log(2.0)
        + component.log_prob(_MASS_VALUES)
        + component.log_cdf(_MASS_VALUES)
    )
    np.testing.assert_allclose(
        np.asarray(distribution.log_prob(_MASS_VALUES)),
        np.asarray(expected),
        rtol=0.0,
        atol=0.0,
    )


def test_max_of_two_normals_icdf_matches_the_closed_form() -> None:
    from jax.scipy.special import ndtri

    distribution = _max_of_two_normals()
    expected = _MASS_MEAN + _MASS_SIGMA * ndtri(jnp.sqrt(_MASS_QUANTILES))
    np.testing.assert_allclose(
        np.asarray(distribution.icdf(_MASS_QUANTILES)),
        np.asarray(expected),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(distribution.cdf(distribution.icdf(_MASS_QUANTILES))),
        np.asarray(_MASS_QUANTILES),
        rtol=1e-12,
    )


def test_max_of_two_normals_samples_are_the_max_of_two_standard_normals() -> None:
    distribution = _max_of_two_normals()
    key = jax.random.PRNGKey(0)
    sample_shape = (32,)
    eps = jax.random.normal(key, shape=(2,) + sample_shape)
    np.testing.assert_array_equal(
        distribution.sample(key, sample_shape=sample_shape),
        _MASS_MEAN + _MASS_SIGMA * jnp.max(eps, axis=0),
    )


def test_max_of_two_normals_survives_jit_as_a_pytree_argument() -> None:
    """Without flattening ``loc`` / ``scale`` / ``_normal``, ``jit`` would drop them."""
    distribution = _max_of_two_normals()
    fields = MaxOfTwoNormalsDistribution.gather_pytree_data_fields()
    assert "loc" in fields
    assert "scale" in fields
    assert "_normal" in fields
    jitted = jax.jit(lambda d, x: d.log_prob(x))(distribution, _MASS_VALUES)
    np.testing.assert_allclose(
        np.asarray(jitted),
        np.asarray(distribution.log_prob(_MASS_VALUES)),
        rtol=1e-14,
    )


# --------------------------------------------------------------------------- #
# UniformCosineDistribution
# --------------------------------------------------------------------------- #
@pytest.fixture
def polar_angles() -> jax.Array:
    return jnp.array([0.2, 0.8, math.pi / 2.0, 2.2, 2.9])


@pytest.fixture
def polar_quantiles() -> jax.Array:
    return jnp.array([0.0, 0.1, 0.5, 0.9, 1.0])


def _uniform_cosine() -> UniformCosineDistribution:
    return UniformCosineDistribution(validate_args=True)


def test_uniform_cosine_density_integrates_to_unity() -> None:
    distribution = _uniform_cosine()
    theta = jnp.linspace(0.0, math.pi, 20_001)
    density = jnp.exp(distribution.log_prob(theta))
    np.testing.assert_allclose(
        float(jnp.trapezoid(density, theta)), 1.0, rtol=1e-8, atol=0.0
    )


def test_uniform_cosine_log_prob_is_negative_infinity_off_support() -> None:
    distribution = _uniform_cosine()
    outside = jnp.array([-0.1, math.pi + 0.1])
    with pytest.warns(UserWarning, match="Out-of-support"):
        log_prob = distribution.log_prob(outside)
    assert bool(jnp.all(jnp.isneginf(log_prob)))


def test_uniform_cosine_cdf_is_zero_or_one_off_support() -> None:
    distribution = _uniform_cosine()
    outside = jnp.array([-2.0, -0.1, math.pi + 0.1, 2.0 * math.pi, 10.0])
    expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        np.asarray(distribution.cdf(outside)), np.asarray(expected)
    )
    mixed = jnp.array([-0.1, 0.0, math.pi / 2.0, math.pi, math.pi + 0.1])
    np.testing.assert_allclose(
        np.asarray(distribution.cdf(mixed)),
        np.array([0.0, 0.0, 0.5, 1.0, 1.0]),
        rtol=1e-12,
        atol=0.0,
    )


def test_uniform_cosine_cdf_and_icdf_are_inverses(
    polar_angles: jax.Array, polar_quantiles: jax.Array
) -> None:
    distribution = _uniform_cosine()
    np.testing.assert_allclose(
        np.asarray(distribution.icdf(polar_quantiles)),
        np.asarray(jnp.arccos(1.0 - 2.0 * polar_quantiles)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(distribution.cdf(distribution.icdf(polar_quantiles))),
        np.asarray(polar_quantiles),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(distribution.icdf(distribution.cdf(polar_angles))),
        np.asarray(polar_angles),
        rtol=1e-12,
    )


def test_uniform_cosine_survives_jit_as_a_pytree_argument(
    polar_angles: jax.Array,
) -> None:
    distribution = _uniform_cosine()
    jitted = jax.jit(lambda d, x: d.log_prob(x))(distribution, polar_angles)
    np.testing.assert_allclose(
        np.asarray(jitted),
        np.asarray(distribution.log_prob(polar_angles)),
        rtol=1e-14,
    )
