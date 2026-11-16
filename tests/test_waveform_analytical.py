"""Tests for the closed-form quadrupolar inspiral intensity.

The scaling tests check exponents through ratios, so they are invariant under
the overall constant; the amplitude prefactor is not covered.
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
    inspiral_intensity,
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
def bns_intensity(bns: dict[str, jax.Array]) -> Callable[..., jax.Array]:
    """Intensity for the reference BNS, with any source parameter overridden."""

    def _intensity(
        frequencies: jax.Array,
        *,
        alpha: float = ISCO_ALPHA,
        **overrides: jax.Array,
    ) -> jax.Array:
        return inspiral_intensity(frequencies, bns | overrides, alpha=alpha)

    return _intensity


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
    """``MEAN_INCLINATION_FACTOR`` is the isotropic average of ``inclination_factor``.

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
    bns: dict[str, jax.Array], bns_intensity: Callable[..., jax.Array]
) -> None:
    frequencies = jnp.array([50.0, 100.0])

    near = bns_intensity(frequencies)
    far = bns_intensity(
        frequencies, luminosity_distance=2.0 * bns["luminosity_distance"]
    )

    np.testing.assert_allclose(far, near / 4.0, rtol=1e-12)


def test_scales_as_chirp_mass_to_the_five_thirds(
    bns: dict[str, jax.Array], bns_intensity: Callable[..., jax.Array]
) -> None:
    frequencies = jnp.array([50.0, 100.0])

    light = bns_intensity(frequencies)
    # Doubling both components doubles Mc -- and halves the cutoff, so these
    # frequencies stay well inside the band for both.
    heavy = bns_intensity(
        frequencies,
        source_frame_mass_1=2.0 * bns["source_frame_mass_1"],
        source_frame_mass_2=2.0 * bns["source_frame_mass_2"],
    )

    np.testing.assert_allclose(heavy, light * 2.0 ** (5.0 / 3.0), rtol=1e-12)


def test_scales_as_frequency_to_the_minus_seven_thirds(
    bns_intensity: Callable[..., jax.Array],
) -> None:
    intensity = bns_intensity(jnp.array([100.0, 200.0]))

    np.testing.assert_allclose(
        intensity[0, 1], intensity[0, 0] * 2.0 ** (-7.0 / 3.0), rtol=1e-12
    )


def test_scales_with_inclination_factor(
    bns_intensity: Callable[..., jax.Array],
) -> None:
    frequencies = jnp.array([100.0])

    face_on = bns_intensity(frequencies)
    edge_on = bns_intensity(frequencies, inclination=jnp.array([math.pi / 2.0]))

    np.testing.assert_allclose(
        edge_on, face_on * 0.25 / FACE_ON_INCLINATION_FACTOR, rtol=1e-12
    )


def test_redshift_enters_only_through_detector_frame_masses(
    bns: dict[str, jax.Array], bns_intensity: Callable[..., jax.Array]
) -> None:
    """z is degenerate with a (1 + z) rescaling of both component masses.

    True of the amplitude and of the cutoff alike, so the comparison covers
    the full band rather than just the low-frequency part.
    """
    frequencies = jnp.linspace(10.0, 3000.0, 400)
    redshift = 0.7

    redshifted = bns_intensity(frequencies, redshift=jnp.array([redshift]))
    rescaled = bns_intensity(
        frequencies,
        redshift=jnp.array([0.0]),
        source_frame_mass_1=(1.0 + redshift) * bns["source_frame_mass_1"],
        source_frame_mass_2=(1.0 + redshift) * bns["source_frame_mass_2"],
    )

    np.testing.assert_allclose(redshifted, rescaled, rtol=1e-12)


def test_intensity_is_zero_above_the_termination_frequency(
    bns_intensity: Callable[..., jax.Array], bns_cutoff: Callable[..., float]
) -> None:
    cutoff = bns_cutoff()
    frequencies = jnp.array([cutoff * 0.999, cutoff, cutoff * 1.001, 2.0 * cutoff])

    intensity = bns_intensity(frequencies)

    assert intensity[0, 0] > 0.0
    assert intensity[0, 1] > 0.0  # the cutoff bin itself is inclusive
    assert intensity[0, 2] == 0.0
    assert intensity[0, 3] == 0.0


def test_zero_frequency_bin_is_zero_not_nan(
    bns_intensity: Callable[..., jax.Array],
) -> None:
    intensity = bns_intensity(jnp.array([0.0, 100.0]))

    assert intensity[0, 0] == 0.0
    assert bool(jnp.all(jnp.isfinite(intensity)))


def test_batched_sources_match_stacked_single_source_calls() -> None:
    frequencies = jnp.array([0.0, 50.0, 500.0, 5000.0])
    sources = {
        "source_frame_mass_1": jnp.array([1.4, 10.0, 30.0]),
        "source_frame_mass_2": jnp.array([1.3, 8.0, 20.0]),
        "redshift": jnp.array([0.1, 0.5, 1.0]),
        "luminosity_distance": jnp.array([500.0, 2000.0, 8000.0]),
        "inclination": jnp.array([0.0, 0.5, 1.0]),
    }
    batched = inspiral_intensity(frequencies, sources, alpha=ISCO_ALPHA)
    independent = jnp.concatenate(
        [
            inspiral_intensity(
                frequencies,
                {name: values[index] for name, values in sources.items()},
                alpha=ISCO_ALPHA,
            )
            for index in range(3)
        ],
        axis=0,
    )

    np.testing.assert_allclose(batched, independent, rtol=1e-12)


def test_gradient_is_finite_across_the_cutoff(bns: dict[str, jax.Array]) -> None:
    """A masked bin must not poison the gradient with the f^(-7/3) blow-up."""
    # Below the band, inside it, and above the ~1570 Hz cutoff.
    frequencies = jnp.array([0.0, 100.0, 5000.0])

    def total_intensity(luminosity_distance: jax.Array) -> jax.Array:
        return jnp.sum(
            inspiral_intensity(
                frequencies,
                bns | {"luminosity_distance": luminosity_distance},
                alpha=ISCO_ALPHA,
            )
        )

    gradient = jax.grad(total_intensity)(bns["luminosity_distance"])

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(gradient[0]) < 0.0  # intensity falls with distance
