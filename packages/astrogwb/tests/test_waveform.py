from __future__ import annotations

import numpy as np
import xarray as xr
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.waveform import (
    apply_gw_distance_to_waveforms,
    make_catalog,
    polarization_power,
)


def _make_catalog(*, with_redshift: bool = True) -> xr.Dataset:
    source_parameters = {
        "mass_1": np.array([20.0, 30.0]),
        "luminosity_distance": np.array([500.0, 5000.0]),
    }
    if with_redshift:
        source_parameters["redshift"] = np.array([0.1, 0.8])
    return make_catalog(
        frequencies=np.array([10.0, 20.0, 30.0]),
        plus=np.array([[1.0 + 1.0j, 2.0, 3.0], [4.0, 5.0 + 1.0j, 6.0]]),
        cross=np.array([[0.0, 1.0, 0.0], [1.0j, 0.0, 2.0]]),
        source_parameters=source_parameters,
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=30.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )


def test_polarization_power_reduces_catalog() -> None:
    catalog = _make_catalog()

    actual = polarization_power(catalog)

    expected = np.array(
        [
            [2.0, 17.0],
            [5.0, 26.0],
            [9.0, 40.0],
        ]
    )
    assert actual.dims == ("frequency", "sample")
    assert actual.shape == (3, 2)  # (nfreq, nsamples)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual.values, expected)


def test_apply_gw_distance_gr_limit_is_identity() -> None:
    # xi_0 = 1 -> xi(z) = 1 everywhere, so polarizations and power are unchanged.
    catalog = _make_catalog()

    corrected = apply_gw_distance_to_waveforms(catalog, xi_0=1.0, xi_n=1.91)

    np.testing.assert_allclose(
        corrected.polarizations.values, catalog.polarizations.values
    )
    np.testing.assert_allclose(
        polarization_power(corrected).values, polarization_power(catalog).values
    )


def test_apply_gw_distance_scales_polarizations_by_inverse_xi() -> None:
    catalog = _make_catalog()
    redshift = catalog.source_parameters.sel(parameter="redshift").values
    xi_0, xi_n = 1.5, 1.91
    xi = np.exp(log_gw_em_ratio(redshift, xi_0=xi_0, xi_n=xi_n))

    corrected = apply_gw_distance_to_waveforms(catalog, xi_0=xi_0, xi_n=xi_n)

    np.testing.assert_allclose(
        corrected.polarizations.values,
        catalog.polarizations.values / xi[None, :, None],
    )
    # Power rescales by 1 / xi^2 per sample (column).
    np.testing.assert_allclose(
        polarization_power(corrected).values,
        polarization_power(catalog).values / xi[None, :] ** 2,
    )


def test_apply_gw_distance_is_pure_and_preserves_metadata() -> None:
    catalog = _make_catalog()

    corrected = apply_gw_distance_to_waveforms(catalog, xi_0=1.5, xi_n=1.91)

    assert corrected is not catalog
    np.testing.assert_array_equal(corrected.frequency.values, catalog.frequency.values)
    np.testing.assert_array_equal(
        corrected.source_parameters.values, catalog.source_parameters.values
    )
    assert corrected.attrs == catalog.attrs
    # The input catalog's arrays must not be modified in place.
    np.testing.assert_allclose(
        catalog.polarizations.values, _make_catalog().polarizations.values
    )


def test_apply_gw_distance_requires_source_parameters() -> None:
    catalog = _make_catalog(with_redshift=False)

    try:
        apply_gw_distance_to_waveforms(catalog, xi_0=1.5, xi_n=1.91)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for missing 'redshift'")
