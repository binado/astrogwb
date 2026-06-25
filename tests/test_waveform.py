from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from astrogwb.waveform import (
    PolarizationPowerCatalog,
    generate_catalog_polarization_power,
    load_polarization_power_catalog,
    save_polarization_power_catalog,
)


def test_polarization_power_catalog_npz_round_trip(tmp_path) -> None:
    catalog = PolarizationPowerCatalog(
        frequencies=jnp.array([10.0, 20.0, 30.0]),
        polarization_power=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        samples={"mass_1": jnp.array([20.0, 30.0]), "redshift": jnp.array([0.1, 0.2])},
    )
    path = tmp_path / "catalog.npz"

    save_polarization_power_catalog(path, catalog)
    actual = load_polarization_power_catalog(path)

    assert isinstance(actual.frequencies, np.ndarray)
    np.testing.assert_allclose(
        np.asarray(actual.frequencies), np.array([10.0, 20.0, 30.0])
    )
    np.testing.assert_allclose(
        np.asarray(actual.polarization_power), np.asarray(catalog.polarization_power)
    )
    np.testing.assert_allclose(
        np.asarray(actual.samples["mass_1"]), np.array([20.0, 30.0])
    )
    np.testing.assert_allclose(
        np.asarray(actual.samples["redshift"]), np.array([0.1, 0.2])
    )


@dataclass
class _Polarizations:
    frequencies: object
    plus: object
    cross: object


class _Backend:
    def generate_fd_polarizations_batch(
        self, approximant, *, sampling_frequency, minimum_frequency, parameters
    ):
        assert approximant == "Toy"
        assert sampling_frequency == 128.0
        assert minimum_frequency == 10.0
        assert parameters["mass_1"].shape == (2,)
        return _Polarizations(
            frequencies=jnp.array([10.0, 20.0, 30.0]),
            plus=jnp.array([[1.0 + 1.0j, 2.0, 3.0], [4.0, 5.0 + 1.0j, 6.0]]),
            cross=jnp.array([[0.0, 1.0, 0.0], [1.0j, 0.0, 2.0]]),
        )


def test_generate_catalog_polarization_power_transposes_raw_power() -> None:
    samples = {"mass_1": jnp.array([20.0, 30.0])}

    actual = generate_catalog_polarization_power(
        samples,
        approximant="Toy",
        sampling_frequency=128.0,
        minimum_frequency=10.0,
        backend=_Backend(),
    )

    expected = np.array(
        [
            [2.0, 17.0],
            [5.0, 26.0],
            [9.0, 40.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(actual.polarization_power), expected)
    assert actual.samples is not samples
    assert actual.samples["mass_1"] is samples["mass_1"]
