"""Tests for the closed-form quadrupolar inspiral polarization power.

The scaling tests check exponents; :func:`test_absolute_normalization` is the
one that pins the prefactor against an independently computed number, so a
refactor cannot quietly move it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from astrogwb.waveform import (
    FACE_ON_INCLINATION_FACTOR,
    ISCO_ALPHA,
    MEAN_INCLINATION_FACTOR,
    chirp_mass,
    inclination_factor,
    inspiral_polarization_power,
    make_catalog,
    termination_frequency,
)
from numpy.typing import NDArray

# Polarization power is of order 1e-47 Hz^-2, far below float32's smallest
# normal (~1.2e-38): without x64 the JAX path underflows to all zeros. Same
# setup as `test_cosmology.py` and `astrogwb_paper.runtime`.
jax.config.update("jax_enable_x64", True)

# The reference source: a face-on 1.4 + 1.4 Msun binary at z = 0, 100 Mpc.
BNS_MASS = np.array([1.4])
BNS_REDSHIFT = np.array([0.0])
BNS_DISTANCE = np.array([100.0])


def bns_power(
    frequencies: NDArray[np.float64],
    *,
    mass_1: NDArray[np.float64] = BNS_MASS,
    mass_2: NDArray[np.float64] = BNS_MASS,
    redshift: NDArray[np.float64] = BNS_REDSHIFT,
    luminosity_distance: NDArray[np.float64] = BNS_DISTANCE,
    inclination: NDArray[np.float64] | float = 0.0,
    alpha: float = ISCO_ALPHA,
) -> NDArray[np.float64]:
    """Power for the reference BNS, with any parameter overridden by keyword.

    Written out rather than splatted from a dict so each test states exactly
    which parameter it perturbs, and so the call keeps its declared types.
    """
    return inspiral_polarization_power(
        frequencies,
        mass_1=mass_1,
        mass_2=mass_2,
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        inclination=inclination,
        alpha=alpha,
    )


def bns_cutoff(alpha: float = ISCO_ALPHA) -> float:
    return float(
        termination_frequency(BNS_MASS, BNS_MASS, BNS_REDSHIFT, alpha=alpha)[0]
    )


def test_inclination_factor_closed_form() -> None:
    assert inclination_factor(0.0) == FACE_ON_INCLINATION_FACTOR
    assert inclination_factor(math.pi) == FACE_ON_INCLINATION_FACTOR
    np.testing.assert_allclose(inclination_factor(math.pi / 2.0), 0.25, atol=1e-15)


def test_mean_inclination_factor_is_the_analytic_inclination_constant() -> None:
    """``<g> / g(0)`` must equal the 0.4 in ``astrogwb.gwb.spectral_density``.

    ``spectral_density(..., average_mode="analytic_inclination")`` multiplies
    by a bare 0.4 because the population graphs generate every source face-on.
    That factor is only correct if it is the ratio of the inclination-averaged
    ``g`` to its face-on value -- so this test is what couples the closed form
    here to the magic number over there.
    """
    cos_iota = np.linspace(-1.0, 1.0, 200_001)
    g = ((1.0 + cos_iota**2) / 2.0) ** 2 + cos_iota**2
    mean_g = np.trapezoid(g, cos_iota) / 2.0

    np.testing.assert_allclose(mean_g, MEAN_INCLINATION_FACTOR, rtol=1e-10)
    np.testing.assert_allclose(
        MEAN_INCLINATION_FACTOR / inclination_factor(0.0), 0.4, rtol=1e-12
    )


def test_chirp_mass_matches_the_definition() -> None:
    np.testing.assert_allclose(chirp_mass(BNS_MASS, BNS_MASS), 1.2187707886, rtol=1e-9)
    # Equal masses: Mc = m * 2^(3/5) / 2^(1/5) = m / 2^(1/5).
    np.testing.assert_allclose(
        chirp_mass(np.array([10.0]), np.array([10.0])), 10.0 / 2.0**0.2, rtol=1e-12
    )


def test_absolute_normalization() -> None:
    """Pin the prefactor against a value computed outside this module.

    5/(24 pi^(4/3)) * (G Mc)^(5/3) / (r^2 c^3 f^(7/3)) * g(0) for a face-on
    1.4 + 1.4 Msun binary at 100 Mpc and 100 Hz. The implied strain amplitude
    is |h~| ~ 6.04e-24 Hz^-1, the right order for a BNS at that distance.
    """
    power = bns_power(np.array([100.0]))

    np.testing.assert_allclose(power[0, 0], 3.6515886323e-47, rtol=1e-9)


def test_scales_as_inverse_distance_squared() -> None:
    frequencies = np.array([50.0, 100.0])

    near = bns_power(frequencies)
    far = bns_power(frequencies, luminosity_distance=2.0 * BNS_DISTANCE)

    np.testing.assert_allclose(far, near / 4.0, rtol=1e-12)


def test_scales_as_chirp_mass_to_the_five_thirds() -> None:
    frequencies = np.array([50.0, 100.0])

    light = bns_power(frequencies)
    # Doubling both components doubles Mc -- and halves the cutoff, so these
    # frequencies stay well inside the band for both.
    heavy = bns_power(frequencies, mass_1=2.0 * BNS_MASS, mass_2=2.0 * BNS_MASS)

    np.testing.assert_allclose(heavy, light * 2.0 ** (5.0 / 3.0), rtol=1e-12)


def test_scales_as_frequency_to_the_minus_seven_thirds() -> None:
    power = bns_power(np.array([100.0, 200.0]))

    np.testing.assert_allclose(
        power[1, 0], power[0, 0] * 2.0 ** (-7.0 / 3.0), rtol=1e-12
    )


def test_scales_with_inclination_factor() -> None:
    frequencies = np.array([100.0])

    face_on = bns_power(frequencies)
    edge_on = bns_power(frequencies, inclination=math.pi / 2.0)

    np.testing.assert_allclose(
        edge_on, face_on * 0.25 / FACE_ON_INCLINATION_FACTOR, rtol=1e-12
    )


def test_redshift_enters_only_through_detector_frame_masses() -> None:
    """z is degenerate with a (1 + z) rescaling of both component masses.

    True of the amplitude and of the cutoff alike, so the comparison covers
    the full band rather than just the low-frequency part.
    """
    frequencies = np.linspace(10.0, 3000.0, 400)
    redshift = 0.7

    redshifted = bns_power(frequencies, redshift=np.array([redshift]))
    rescaled = bns_power(
        frequencies,
        mass_1=(1.0 + redshift) * BNS_MASS,
        mass_2=(1.0 + redshift) * BNS_MASS,
    )

    np.testing.assert_allclose(redshifted, rescaled, rtol=1e-12)


def test_termination_frequency_at_isco() -> None:
    np.testing.assert_allclose(bns_cutoff(), 1570.4196, rtol=1e-6)


def test_power_is_zero_above_the_termination_frequency() -> None:
    cutoff = bns_cutoff()
    frequencies = np.array([cutoff * 0.999, cutoff, cutoff * 1.001, 2.0 * cutoff])

    power = bns_power(frequencies)

    assert power[0, 0] > 0.0
    assert power[1, 0] > 0.0  # the cutoff bin itself is inclusive
    assert power[2, 0] == 0.0
    assert power[3, 0] == 0.0


def test_larger_alpha_never_removes_a_surviving_bin() -> None:
    frequencies = np.linspace(10.0, 4000.0, 500)

    narrow = bns_power(frequencies)
    wide = bns_power(frequencies, alpha=2.0 * ISCO_ALPHA)

    assert np.all(wide[narrow > 0.0] > 0.0)
    assert np.count_nonzero(wide) > np.count_nonzero(narrow)


def test_zero_frequency_bin_is_zero_not_nan() -> None:
    power = bns_power(np.array([0.0, 100.0]))

    assert power[0, 0] == 0.0
    assert np.all(np.isfinite(power))


def test_layout_is_catalog_ready() -> None:
    """``(F, N)`` float64 -- drops straight into ``make_catalog``."""
    frequencies = np.arange(10.0, 60.0, 10.0)
    sources = {
        "mass_1": np.array([1.4, 1.6, 2.0]),
        "mass_2": np.array([1.4, 1.3, 1.1]),
        "redshift": np.array([0.1, 0.5, 1.2]),
        "luminosity_distance": np.array([500.0, 2000.0, 8000.0]),
        "inclination": np.array([0.0, 0.5, 1.2]),
    }

    power = inspiral_polarization_power(
        frequencies,
        mass_1=sources["mass_1"],
        mass_2=sources["mass_2"],
        redshift=sources["redshift"],
        luminosity_distance=sources["luminosity_distance"],
        inclination=sources["inclination"],
        alpha=ISCO_ALPHA,
    )

    assert power.shape == (frequencies.size, 3)
    assert power.dtype == np.float64
    # make_catalog validates the layout for us; a wrong orientation raises.
    catalog = make_catalog(
        frequencies=frequencies,
        polarization_power=power,
        source_parameters=sources,
        approximant="analytical-inspiral",
        minimum_frequency=10.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=10.0,
    )
    assert catalog.sizes["sample"] == 3


def test_jax_matches_numpy_and_is_jittable() -> None:
    frequencies = np.linspace(10.0, 2000.0, 128)
    mass_1 = np.array([1.4, 2.0])
    mass_2 = np.array([1.4, 1.1])
    redshift = np.array([0.1, 0.9])
    luminosity_distance = np.array([500.0, 6000.0])
    inclination = np.array([0.0, 1.0])

    expected = inspiral_polarization_power(
        frequencies,
        mass_1=mass_1,
        mass_2=mass_2,
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        inclination=inclination,
        alpha=ISCO_ALPHA,
    )
    actual = jax.jit(inspiral_polarization_power)(
        jnp.asarray(frequencies),
        mass_1=jnp.asarray(mass_1),
        mass_2=jnp.asarray(mass_2),
        redshift=jnp.asarray(redshift),
        luminosity_distance=jnp.asarray(luminosity_distance),
        inclination=jnp.asarray(inclination),
        alpha=ISCO_ALPHA,
    )

    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12)


def test_gradient_is_finite_across_the_cutoff() -> None:
    """A masked bin must not poison the gradient with the f^(-7/3) blow-up."""
    # Below the band, inside it, and above the ~1570 Hz cutoff.
    frequencies = jnp.array([0.0, 100.0, 5000.0])

    def total_power(luminosity_distance: jax.Array) -> jax.Array:
        return jnp.sum(
            inspiral_polarization_power(
                frequencies,
                mass_1=jnp.asarray(BNS_MASS),
                mass_2=jnp.asarray(BNS_MASS),
                redshift=jnp.asarray(BNS_REDSHIFT),
                luminosity_distance=luminosity_distance,
                inclination=0.0,
                alpha=ISCO_ALPHA,
            )
        )

    gradient = jax.grad(total_power)(jnp.asarray(BNS_DISTANCE))

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(gradient[0]) < 0.0  # power falls with distance
