"""A catalog's energy spectrum against ``pygwb.background``.

``compute_Omega_from_CBC_dictionary`` sums LAL IMRPhenomD waveforms through
bilby for the CoBA BNS events of ``references/coba_bns_events.json``; the
library evaluates the same events with Ripple's IMRPhenomD. Both sides read
the events' detector-frame masses and luminosity distances, so no cosmology
enters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import REFERENCES
from numpy.typing import NDArray
from tolerances import assert_close

from astrogwb.background.spectral import omega_gw_from_spectral_density
from astrogwb.waveform import WaveformMetadata
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.factory import waveform_generator


@pytest.fixture(scope="module")
def events() -> dict[str, NDArray[np.float64]]:
    payload = json.loads((REFERENCES / "coba_bns_events.json").read_text())
    return {name: np.asarray(values) for name, values in payload["events"].items()}


@pytest.fixture(scope="module")
def generator(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
) -> WaveformGenerator:
    frequencies = pygwb_reference["cbc_frequencies"]
    return waveform_generator(
        WaveformMetadata(
            approximant="IMRPhenomD",
            minimum_frequency=float(frequencies[0]),
            maximum_frequency=float(frequencies[-1]),
            reference_frequency=float(pygwb_reference["cbc_reference_frequency"]),
            sampling_frequency=float(pygwb_reference["cbc_sampling_frequency"]),
            frequency_resolution=float(frequencies[1] - frequencies[0]),
        )
    )


def test_generator_grid_is_the_reference_grid(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    generator: WaveformGenerator,
) -> None:
    np.testing.assert_array_equal(
        generator.frequencies, pygwb_reference["cbc_frequencies"]
    )


def test_catalog_omega_gw_matches_pygwb(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    hubble_constant: float,
    events: dict[str, NDArray[np.float64]],
    generator: WaveformGenerator,
) -> None:
    intensity = generator.generate_batch(events)
    count = intensity.shape[-1]
    rate = count / float(pygwb_reference["cbc_observation_seconds"])

    omega = omega_gw_from_spectral_density(
        rate * jnp.mean(intensity, axis=-1),
        jnp.asarray(generator.frequencies),
        hubble_constant=hubble_constant,
    )

    assert_close(omega, pygwb_reference["cbc_omega_gw"])
