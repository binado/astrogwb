"""Tests for the NumPyro-native population distributions.

The physics is checked against
:func:`reference_population.reference_merger_rate_distance_and_logprob`, an
independent restatement of the redshift density. The rest checks the pytree
registration, which no linter or in-model test would catch.
"""

from __future__ import annotations

import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

# The mock population's fiducials and grid: the tight agreement tests below
# need both implementations on the same grid.
from astrogwb_mock_population import (
    FIDUCIALS,
    N_GRID,
    Z_MAX,
    Z_MIN,
    make_redshift_grid,
    mock_merger_rate,
)
from reference_population import reference_merger_rate_distance_and_logprob
from tolerances import assert_close, assert_converges

from astrogwb.distributions.interpolated import InterpolatedDistribution
from astrogwb.distributions.mass import (
    MaxOfTwoNormalsDistribution,
    MaxOfTwoUniformsDistribution,
)
from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.rates import total_merger_rate

#: Redshifts to evaluate at: interior to the grid, and not on a node.
SAMPLE_REDSHIFTS = jnp.array([0.5, 1.234, 3.7, 12.0, 19.5])


def _distribution(**overrides: float) -> RedshiftDistribution:
    """The Madau-Dickinson specimen at the mock fiducials, on the mock grid."""
    return RedshiftDistribution(
        source_frame_distribution=mock_merger_rate,
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
def test_interpolated_normalizes_an_unnormalized_table() -> None:
    """`norm` is the trapezoid integral, and dividing by it is the whole density."""
    x = jnp.linspace(0.0, 1.0, 129)
    interpolated = InterpolatedDistribution(x, 7.5 * (1.0 + x))

    assert_close(interpolated.norm, 7.5 * 1.5)
    density = np.exp(np.asarray(interpolated.log_prob(x)))
    assert_close(np.trapezoid(density, np.asarray(x)), 1.0)


# --------------------------------------------------------------------------- #
# The Madau-Dickinson density against the reference model
# --------------------------------------------------------------------------- #
def test_grid_is_bit_identical_to_the_shared_redshift_grid() -> None:
    """What licenses the two tolerance-tight agreement tests below."""
    np.testing.assert_array_equal(
        np.asarray(_distribution().x), np.asarray(make_redshift_grid())
    )


@pytest.mark.parametrize(
    "overrides",
    [{}, {"gamma": 2.7, "kappa": 2.9, "z_peak": 1.9, "H0": 74.0, "Omega_m": 0.27}],
    ids=["fiducial", "off-fiducial"],
)
def test_log_prob_agrees_with_the_reference_model(overrides: dict[str, float]) -> None:
    # Bit-exact: `log_prob` shares the reference's operation order.
    _, _, reference_logpdf = _reference(**overrides)
    np.testing.assert_array_equal(
        np.asarray(_distribution(**overrides).log_prob(SAMPLE_REDSHIFTS)),
        np.asarray(reference_logpdf),
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
    assert_close(np.trapezoid(density, grid), 1.0)


def test_log_prob_is_negative_infinity_off_the_table() -> None:
    """The `left=0.0, right=0.0` contract: no extrapolation, and no exception --
    the importance weights depend on `-inf`."""
    outside = jnp.array([Z_MIN - 0.1, Z_MAX + 0.1])
    assert bool(jnp.all(jnp.isneginf(_distribution().log_prob(outside))))


def test_cdf_grid_spans_zero_to_one_exactly() -> None:
    """The renormalizing divide: `cumsum`-order and `sum`-order reductions do not
    agree to the last bit, so the endpoint is made exact by construction."""
    _cdf_grid = _distribution()._cdf_grid
    assert float(_cdf_grid[0]) == 0.0
    assert float(_cdf_grid[-1]) == 1.0
    assert bool(jnp.all(jnp.diff(_cdf_grid) > 0.0))


def test_icdf_inverts_the_tabulated_cdf() -> None:
    distribution = _distribution()
    assert_close(distribution.icdf(distribution._cdf_grid), distribution.x)


def test_sample_mean_matches_the_density_it_reports() -> None:
    """Inverse-transform draws must follow the density `log_prob` publishes."""
    distribution = _distribution()
    grid = np.asarray(distribution.x)
    density = np.exp(np.asarray(distribution.log_prob(distribution.x)))
    expected = float(np.trapezoid(grid * density, grid))

    draws = np.asarray(distribution.sample(jax.random.PRNGKey(0), (400_000,)))

    # Five standard errors of these draws, not a fixed rtol.
    standard_error = float(draws.std(ddof=1)) / np.sqrt(draws.size)
    assert abs(float(draws.mean()) - expected) < 5.0 * standard_error


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
        assert_close(row, distribution.log_prob(SAMPLE_REDSHIFTS))
    # Three *distinct* rows: identical rows would pass with the pytree broken.
    assert not np.allclose(np.asarray(mapped[0]), np.asarray(mapped[1]))
    assert not np.allclose(np.asarray(mapped[1]), np.asarray(mapped[2]))


def test_materializing_lazy_fields_does_not_retrace() -> None:
    """Our distribution's field declarations must keep its JIT signature stable."""
    distribution = _distribution()
    traces = 0

    def log_prob(d: RedshiftDistribution) -> jax.Array:
        nonlocal traces
        traces += 1
        return d.log_prob(SAMPLE_REDSHIFTS)

    jitted = jax.jit(log_prob)
    before = jitted(distribution)

    # Force every lazy_property to materialize onto the instance.
    _ = distribution.norm, distribution._normalized_y, distribution._cdf_grid

    after = jitted(distribution)
    assert traces == 1
    np.testing.assert_array_equal(before, after)


# --------------------------------------------------------------------------- #
# Window accessors, derived rather than stored
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Callable source-frame rate law
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_normalized_density_is_independent_of_the_hubble_constant() -> None:
    """`dV_c/dz` is proportional to `H0^-3`, which cancels in the normalization."""

    def log_prob_at(name: str, value: float) -> jax.Array:
        return _distribution(**{name: value}).log_prob(jnp.array(1.234))

    d_h0 = jax.grad(lambda h0: log_prob_at("H0", h0))(FIDUCIALS["H0"])
    d_omega_m = jax.grad(lambda om: log_prob_at("Omega_m", om))(FIDUCIALS["Omega_m"])

    assert abs(float(d_h0)) < 1e-12
    # The scale the H0 derivative is small *relative to*: the same density does
    # respond to the other cosmology parameter.
    assert abs(float(d_omega_m)) > 1e-3


def test_total_merger_rate_scales_as_the_inverse_cube_of_the_hubble_constant() -> None:
    """Pins the H0**-3 merger-rate scaling against the class itself."""

    def rate_at(h0: float) -> jax.Array:
        return _distribution(H0=h0).total_merger_rate()

    h0 = FIDUCIALS["H0"]
    assert_close(jax.grad(rate_at)(h0), -3.0 * rate_at(h0) / h0)


# --------------------------------------------------------------------------- #
# MaxOfTwoNormalsDistribution
# --------------------------------------------------------------------------- #
_MASS_MEAN = 1.33
_MASS_SIGMA = 0.09
_MASS_VALUES = jnp.array([1.1, 1.33, 1.5])
_MASS_QUANTILES = jnp.array([0.1, 0.5, 0.9])


def _max_of_two_normals() -> MaxOfTwoNormalsDistribution:
    return MaxOfTwoNormalsDistribution(_MASS_MEAN, _MASS_SIGMA, validate_args=True)


def test_max_of_two_normals_density_integrates_to_unity() -> None:
    distribution = _max_of_two_normals()
    mass = jnp.linspace(
        _MASS_MEAN - 10 * _MASS_SIGMA, _MASS_MEAN + 10 * _MASS_SIGMA, 20_001
    )
    density = jnp.exp(distribution.log_prob(mass))
    assert_close(jnp.trapezoid(density, mass), 1.0)


def test_max_of_two_normals_icdf_inverts_the_cdf() -> None:
    distribution = _max_of_two_normals()
    assert_close(distribution.cdf(distribution.icdf(_MASS_QUANTILES)), _MASS_QUANTILES)


# --------------------------------------------------------------------------- #
# MaxOfTwoUniformsDistribution
# --------------------------------------------------------------------------- #
@pytest.fixture
def max_of_two_uniforms() -> MaxOfTwoUniformsDistribution:
    return MaxOfTwoUniformsDistribution(1.0, 1.5, validate_args=True)


def test_max_of_two_uniforms_log_prob_is_the_scaled_beta_2_1(
    max_of_two_uniforms: MaxOfTwoUniformsDistribution,
) -> None:
    scaled_beta = dist.TransformedDistribution(
        dist.Beta(2.0, 1.0), dist.transforms.AffineTransform(1.0, 1.5)
    )
    values = jnp.array([1.1, 1.75, 2.4])
    assert_close(max_of_two_uniforms.log_prob(values), scaled_beta.log_prob(values))


def test_max_of_two_uniforms_icdf_inverts_the_cdf(
    max_of_two_uniforms: MaxOfTwoUniformsDistribution,
) -> None:
    assert_close(
        max_of_two_uniforms.cdf(max_of_two_uniforms.icdf(_MASS_QUANTILES)),
        _MASS_QUANTILES,
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
    assert_close(jnp.trapezoid(density, theta), 1.0)


def test_uniform_cosine_cdf_is_zero_or_one_off_support() -> None:
    distribution = _uniform_cosine()
    outside = jnp.array([-2.0, -0.1, math.pi + 0.1, 2.0 * math.pi, 10.0])
    expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        np.asarray(distribution.cdf(outside)), np.asarray(expected)
    )
    mixed = jnp.array([-0.1, 0.0, math.pi / 2.0, math.pi, math.pi + 0.1])
    assert_close(distribution.cdf(mixed), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_uniform_cosine_cdf_and_icdf_are_inverses(
    polar_angles: jax.Array, polar_quantiles: jax.Array
) -> None:
    distribution = _uniform_cosine()
    assert_close(distribution.cdf(distribution.icdf(polar_quantiles)), polar_quantiles)
    assert_close(distribution.icdf(distribution.cdf(polar_angles)), polar_angles)


# --------------------------------------------------------------------------- #
# Merger rate off the grid
# --------------------------------------------------------------------------- #
def test_merger_rate_is_pointwise_and_keeps_the_shape_of_its_input() -> None:
    distribution = _distribution()
    redshift = jnp.array([[0.3, 1.0, 4.0], [0.5, 2.0, 7.5]])
    rate = distribution.merger_rate(redshift)
    assert rate.shape == redshift.shape
    assert_close(rate[0, 1], distribution.merger_rate(jnp.asarray(1.0)))


def test_total_merger_rate_by_quadrature_matches_the_distribution() -> None:
    """The quadrature rate and the table-based one integrate the same function."""
    grid_sizes = (129, 257, 513)
    distributions = [
        RedshiftDistribution(
            source_frame_distribution=mock_merger_rate,
            params=FIDUCIALS,
            minimum_redshift=Z_MIN,
            maximum_redshift=Z_MAX,
            n_grid=n,
        )
        for n in grid_sizes
    ]
    quadrature = total_merger_rate(
        Z_MIN,
        Z_MAX,
        distributions[-1].merger_rate,
        FIDUCIALS["H0"],
        FIDUCIALS["Omega_m"],
    )
    # The distribution's own normalization uses a trapezoid rule on the grid,
    # so it converges at second order to the Gauss-Legendre quadrature rate.
    assert_converges(
        [distribution.total_merger_rate() for distribution in distributions],
        quadrature,
        spacings=[1.0 / (n - 1) for n in grid_sizes],
        order=2.0,
    )


# --------------------------------------------------------------------------- #
# Pytree registration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("distribution", "values"),
    [
        (MaxOfTwoNormalsDistribution(_MASS_MEAN, _MASS_SIGMA), _MASS_VALUES),
        (MaxOfTwoUniformsDistribution(1.0, 1.5), jnp.array([1.1, 1.75, 2.4])),
        (UniformCosineDistribution(), jnp.array([0.2, 0.8, math.pi / 2.0, 2.2])),
    ],
    ids=["max-of-two-normals", "max-of-two-uniforms", "uniform-cosine"],
)
def test_distribution_survives_jit_as_a_pytree_argument(
    distribution: dist.Distribution, values: jax.Array
) -> None:
    """Without its data fields flattened, ``jit`` would drop the parameters."""
    jitted = jax.jit(lambda d, x: d.log_prob(x))(distribution, values)
    assert_close(jitted, distribution.log_prob(values))
