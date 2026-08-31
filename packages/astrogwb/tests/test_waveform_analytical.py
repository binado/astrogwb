"""Tests for the closed-form quadrupolar inspiral polarization power.

The scaling tests check exponents, so they are all invariant under a wrong
overall constant: every assertion here is a ratio with the same constant on
both sides. The absolute scale is pinned in ``test_constants.py``, by
:func:`test_gwb_and_waveform_alpha_conventions_agree` for
``SOLAR_MASS_IN_SECONDS``. Nothing covers the amplitude prefactor.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.constants import (
    FACE_ON_INCLINATION_FACTOR,
    ISCO_ALPHA,
    MEAN_INCLINATION_FACTOR,
)
from astrogwb.waveform import (
    inclination_factor,
    inspiral_polarization_power,
    make_catalog,
    termination_frequency,
)


@pytest.fixture
def bns() -> dict[str, jax.Array]:
    """The reference source: a face-on 1.4 + 1.4 Msun binary at z = 0, 100 Mpc."""
    return {
        "source_frame_mass_1": jnp.array([1.4]),
        "source_frame_mass_2": jnp.array([1.4]),
        "redshift": jnp.array([0.008]),
        "luminosity_distance": jnp.array([40.0]),
        "inclination": jnp.array([0.0]),
    }


@pytest.fixture
def bns_power(bns: dict[str, jax.Array]) -> Callable[..., jax.Array]:
    """Power for the reference BNS, with any source parameter overridden."""

    def _power(
        frequencies: jax.Array,
        *,
        alpha: float = ISCO_ALPHA,
        **overrides: jax.Array,
    ) -> jax.Array:
        return inspiral_polarization_power(frequencies, bns | overrides, alpha=alpha)

    return _power


@pytest.fixture
def bns_cutoff(bns: dict[str, jax.Array]) -> Callable[..., float]:
    """Termination frequency of the reference BNS, in Hz."""

    def _cutoff(alpha: float = ISCO_ALPHA) -> float:
        return float(
            termination_frequency(
                bns["source_frame_mass_1"],
                bns["source_frame_mass_2"],
                bns["redshift"],
                alpha=alpha,
            )[0]
        )

    return _cutoff


def test_mean_inclination_factor_is_the_analytic_inclination_constant() -> None:
    """``<g> / g(0)`` must equal the 0.4 in ``astrogwb.gwb.spectral_density``.

    ``spectral_density(..., average_mode="analytic_inclination")`` multiplies
    by a bare 0.4 because the population graphs generate every source face-on.
    That factor is only correct if it is the ratio of the inclination-averaged
    ``g`` to its face-on value -- so this test is what couples the closed form
    here to the magic number over there.

    The average is over ``cos iota`` uniform on ``[-1, 1]``, so integrating
    ``inclination_factor`` over the angle picks up a ``sin iota`` Jacobian and
    a normalization of ``int_0^pi sin iota diota = 2``.
    """
    iota = jnp.linspace(0.0, math.pi, 200_001)
    mean_g = jnp.trapezoid(inclination_factor(iota) * jnp.sin(iota), iota) / 2.0

    np.testing.assert_allclose(mean_g, MEAN_INCLINATION_FACTOR, rtol=1e-9)
    np.testing.assert_allclose(
        MEAN_INCLINATION_FACTOR / inclination_factor(0.0), 0.4, rtol=1e-12
    )


def test_scales_as_inverse_distance_squared(
    bns: dict[str, jax.Array], bns_power: Callable[..., jax.Array]
) -> None:
    frequencies = jnp.array([50.0, 100.0])

    near = bns_power(frequencies)
    far = bns_power(frequencies, luminosity_distance=2.0 * bns["luminosity_distance"])

    np.testing.assert_allclose(far, near / 4.0, rtol=1e-12)


def test_scales_as_chirp_mass_to_the_five_thirds(
    bns: dict[str, jax.Array], bns_power: Callable[..., jax.Array]
) -> None:
    frequencies = jnp.array([50.0, 100.0])

    light = bns_power(frequencies)
    # Doubling both components doubles Mc -- and halves the cutoff, so these
    # frequencies stay well inside the band for both.
    heavy = bns_power(
        frequencies,
        source_frame_mass_1=2.0 * bns["source_frame_mass_1"],
        source_frame_mass_2=2.0 * bns["source_frame_mass_2"],
    )

    np.testing.assert_allclose(heavy, light * 2.0 ** (5.0 / 3.0), rtol=1e-12)


def test_scales_as_frequency_to_the_minus_seven_thirds(
    bns_power: Callable[..., jax.Array],
) -> None:
    power = bns_power(jnp.array([100.0, 200.0]))

    np.testing.assert_allclose(
        power[0, 1], power[0, 0] * 2.0 ** (-7.0 / 3.0), rtol=1e-12
    )


def test_scales_with_inclination_factor(bns_power: Callable[..., jax.Array]) -> None:
    frequencies = jnp.array([100.0])

    face_on = bns_power(frequencies)
    edge_on = bns_power(frequencies, inclination=jnp.array([math.pi / 2.0]))

    np.testing.assert_allclose(
        edge_on, face_on * 0.25 / FACE_ON_INCLINATION_FACTOR, rtol=1e-12
    )


def test_redshift_enters_only_through_detector_frame_masses(
    bns: dict[str, jax.Array], bns_power: Callable[..., jax.Array]
) -> None:
    """z is degenerate with a (1 + z) rescaling of both component masses.

    True of the amplitude and of the cutoff alike, so the comparison covers
    the full band rather than just the low-frequency part.
    """
    frequencies = jnp.linspace(10.0, 3000.0, 400)
    redshift = 0.7

    redshifted = bns_power(frequencies, redshift=jnp.array([redshift]))
    rescaled = bns_power(
        frequencies,
        redshift=jnp.array([0.0]),
        source_frame_mass_1=(1.0 + redshift) * bns["source_frame_mass_1"],
        source_frame_mass_2=(1.0 + redshift) * bns["source_frame_mass_2"],
    )

    np.testing.assert_allclose(redshifted, rescaled, rtol=1e-12)


def test_power_is_zero_above_the_termination_frequency(
    bns_power: Callable[..., jax.Array], bns_cutoff: Callable[..., float]
) -> None:
    cutoff = bns_cutoff()
    frequencies = jnp.array([cutoff * 0.999, cutoff, cutoff * 1.001, 2.0 * cutoff])

    power = bns_power(frequencies)

    assert power[0, 0] > 0.0
    assert power[0, 1] > 0.0  # the cutoff bin itself is inclusive
    assert power[0, 2] == 0.0
    assert power[0, 3] == 0.0


def test_larger_alpha_never_removes_a_surviving_bin(
    bns_power: Callable[..., jax.Array],
) -> None:
    frequencies = jnp.linspace(10.0, 4000.0, 500)

    narrow = bns_power(frequencies)
    wide = bns_power(frequencies, alpha=2.0 * ISCO_ALPHA)

    assert bool(jnp.all(wide[narrow > 0.0] > 0.0))
    assert jnp.count_nonzero(wide) > jnp.count_nonzero(narrow)


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(jnp.array([ISCO_ALPHA]), id="length-one"),
        pytest.param(jnp.array([ISCO_ALPHA, 2.0 * ISCO_ALPHA]), id="vector"),
    ],
)
def test_termination_frequency_requires_scalar_alpha(alpha: jax.Array) -> None:
    with pytest.raises(ValueError, match="alpha must be a scalar"):
        termination_frequency(1.4, 1.3, 0.1, alpha=alpha)


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(jnp.array([ISCO_ALPHA]), id="length-one"),
        pytest.param(jnp.array([ISCO_ALPHA, 2.0 * ISCO_ALPHA]), id="vector"),
    ],
)
def test_inspiral_power_requires_scalar_alpha(
    bns: dict[str, jax.Array], alpha: jax.Array
) -> None:
    with pytest.raises(ValueError, match="alpha must be a scalar"):
        inspiral_polarization_power(jnp.array([100.0]), bns, alpha=alpha)


def test_zero_frequency_bin_is_zero_not_nan(
    bns_power: Callable[..., jax.Array],
) -> None:
    power = bns_power(jnp.array([0.0, 100.0]))

    assert power[0, 0] == 0.0
    assert bool(jnp.all(jnp.isfinite(power)))


def test_scalar_source_parameters_match_length_one_arrays() -> None:
    frequencies = jnp.array([20.0, 100.0, 2000.0])
    scalar_sources = {
        "source_frame_mass_1": 1.4,
        "source_frame_mass_2": 1.3,
        "redshift": 0.2,
        "luminosity_distance": 800.0,
        "inclination": 0.4,
    }
    array_sources = {
        name: jnp.asarray([value]) for name, value in scalar_sources.items()
    }

    scalar_power = inspiral_polarization_power(
        frequencies, scalar_sources, alpha=ISCO_ALPHA
    )
    array_power = inspiral_polarization_power(
        frequencies, array_sources, alpha=ISCO_ALPHA
    )

    assert scalar_power.shape == (1, frequencies.size)
    np.testing.assert_allclose(scalar_power, array_power, rtol=1e-12)


def test_extra_gwmock_parameters_are_ignored(bns: dict[str, jax.Array]) -> None:
    frequencies = jnp.array([50.0, 100.0])
    complete_population = bns | {
        "spin_1x": jnp.array([0.1]),
        "spin_2z": jnp.array([-0.2]),
        "phase": jnp.array([1.2]),
    }

    expected = inspiral_polarization_power(frequencies, bns, alpha=ISCO_ALPHA)
    actual = inspiral_polarization_power(
        frequencies, complete_population, alpha=ISCO_ALPHA
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "missing_key",
    [
        "source_frame_mass_1",
        "source_frame_mass_2",
        "redshift",
        "luminosity_distance",
        "inclination",
    ],
)
def test_missing_required_parameter_raises_named_key_error(
    bns: dict[str, jax.Array], missing_key: str
) -> None:
    incomplete_population = {
        key: value for key, value in bns.items() if key != missing_key
    }

    with pytest.raises(KeyError, match=missing_key):
        inspiral_polarization_power(
            jnp.array([100.0]), incomplete_population, alpha=ISCO_ALPHA
        )


def test_scalar_and_vector_source_parameters_broadcast_together() -> None:
    frequencies = jnp.array([50.0, 100.0])
    mixed_sources = {
        "source_frame_mass_1": 1.4,
        "source_frame_mass_2": jnp.array([1.1, 1.2, 1.3]),
        "redshift": 0.1,
        "luminosity_distance": jnp.array([500.0, 1000.0, 1500.0]),
        "inclination": 0.0,
    }
    vector_sources = {
        name: jnp.broadcast_to(jnp.asarray(value), (3,))
        for name, value in mixed_sources.items()
    }

    actual = inspiral_polarization_power(frequencies, mixed_sources, alpha=ISCO_ALPHA)
    expected = inspiral_polarization_power(
        frequencies,
        vector_sources,
        alpha=ISCO_ALPHA,
    )

    assert actual.shape == (3, frequencies.size)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_batched_sources_match_stacked_single_source_calls() -> None:
    frequencies = jnp.array([0.0, 50.0, 500.0, 5000.0])
    sources = {
        "source_frame_mass_1": jnp.array([1.4, 10.0, 30.0]),
        "source_frame_mass_2": jnp.array([1.3, 8.0, 20.0]),
        "redshift": jnp.array([0.1, 0.5, 1.0]),
        "luminosity_distance": jnp.array([500.0, 2000.0, 8000.0]),
        "inclination": jnp.array([0.0, 0.5, 1.0]),
    }
    batched = inspiral_polarization_power(frequencies, sources, alpha=ISCO_ALPHA)
    independent = jnp.concatenate(
        [
            inspiral_polarization_power(
                frequencies,
                {name: values[index] for name, values in sources.items()},
                alpha=ISCO_ALPHA,
            )
            for index in range(3)
        ],
        axis=0,
    )

    np.testing.assert_allclose(batched, independent, rtol=1e-12)


@pytest.mark.parametrize("frequencies", [jnp.array(100.0), jnp.ones((2, 3))])
def test_frequency_grid_must_be_one_dimensional(frequencies: jax.Array) -> None:
    with pytest.raises(ValueError, match=r"frequencies must have shape \(F,\)"):
        inspiral_polarization_power(
            frequencies,
            {
                "source_frame_mass_1": 1.4,
                "source_frame_mass_2": 1.4,
                "redshift": 0.0,
                "luminosity_distance": 100.0,
                "inclination": 0.0,
            },
            alpha=ISCO_ALPHA,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("source_frame_mass_1", jnp.ones((2, 1)), id="column-vector"),
        pytest.param("inclination", jnp.ones((1, 1, 1)), id="rank-three"),
    ],
)
def test_higher_rank_source_parameters_are_rejected(
    name: str, value: jax.Array
) -> None:
    sources = {
        "source_frame_mass_1": 1.4,
        "source_frame_mass_2": 1.4,
        "redshift": 0.0,
        "luminosity_distance": 100.0,
        "inclination": 0.0,
    }
    sources[name] = value

    with pytest.raises(ValueError, match=rf"^{name} must be a scalar"):
        inspiral_polarization_power(jnp.array([100.0]), sources, alpha=ISCO_ALPHA)


def test_incompatible_source_vector_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        inspiral_polarization_power(
            jnp.array([100.0]),
            {
                "source_frame_mass_1": jnp.ones(2),
                "source_frame_mass_2": jnp.ones(3),
                "redshift": 0.0,
                "luminosity_distance": 100.0,
                "inclination": 0.0,
            },
            alpha=ISCO_ALPHA,
        )


def test_layout_is_catalog_ready() -> None:
    """``(N, F)`` out; ``make_catalog`` takes the transpose to ``(F, N)``."""
    frequencies = np.arange(10.0, 60.0, 10.0)
    sources = {
        "source_frame_mass_1": np.array([1.4, 1.6, 2.0]),
        "source_frame_mass_2": np.array([1.4, 1.3, 1.1]),
        "redshift": np.array([0.1, 0.5, 1.2]),
        "luminosity_distance": np.array([500.0, 2000.0, 8000.0]),
        "inclination": np.array([0.0, 0.5, 1.2]),
    }

    power = inspiral_polarization_power(
        jnp.asarray(frequencies),
        {name: jnp.asarray(value) for name, value in sources.items()},
        alpha=ISCO_ALPHA,
    )

    assert power.shape == (3, frequencies.size)
    assert power.dtype == jnp.float64
    # make_catalog validates the layout for us; a wrong orientation raises.
    catalog = make_catalog(
        frequencies=frequencies,
        polarization_power=np.asarray(power).T,
        source_parameters=sources,
        approximant="analytical-inspiral",
        minimum_frequency=10.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=10.0,
    )
    assert catalog.sizes["sample"] == 3


def test_jit_matches_eager_evaluation() -> None:
    frequencies = jnp.linspace(10.0, 2000.0, 128)
    sources = {
        "source_frame_mass_1": jnp.array([1.4, 2.0]),
        "source_frame_mass_2": jnp.array([1.4, 1.1]),
        "redshift": jnp.array([0.1, 0.9]),
        "luminosity_distance": jnp.array([500.0, 6000.0]),
        "inclination": jnp.array([0.0, 1.0]),
    }

    expected = inspiral_polarization_power(frequencies, sources, alpha=ISCO_ALPHA)
    actual = jax.jit(inspiral_polarization_power)(
        frequencies, sources, alpha=ISCO_ALPHA
    )

    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_gradient_is_finite_across_the_cutoff(bns: dict[str, jax.Array]) -> None:
    """A masked bin must not poison the gradient with the f^(-7/3) blow-up."""
    # Below the band, inside it, and above the ~1570 Hz cutoff.
    frequencies = jnp.array([0.0, 100.0, 5000.0])

    def total_power(luminosity_distance: jax.Array) -> jax.Array:
        return jnp.sum(
            inspiral_polarization_power(
                frequencies,
                bns | {"luminosity_distance": luminosity_distance},
                alpha=ISCO_ALPHA,
            )
        )

    gradient = jax.grad(total_power)(bns["luminosity_distance"])

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(gradient[0]) < 0.0  # power falls with distance
