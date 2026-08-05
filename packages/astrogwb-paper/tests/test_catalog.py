from __future__ import annotations

import numpy as np
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.waveform import polarization_power
from astrogwb_paper.catalog import apply_gw_distance_at_fiducial
from pluscross import WaveformCatalog


def _make_catalog() -> WaveformCatalog:
    return WaveformCatalog(
        frequencies=np.array([10.0, 20.0, 30.0]),
        plus=np.array([[1.0 + 1.0j, 2.0, 3.0], [4.0, 5.0 + 1.0j, 6.0]]),
        cross=np.array([[0.0, 1.0, 0.0], [1.0j, 0.0, 2.0]]),
        source_parameters={
            "mass_1": np.array([20.0, 30.0]),
            "redshift": np.array([0.1, 0.8]),
            "luminosity_distance": np.array([500.0, 5000.0]),
        },
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=30.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )


def test_apply_gw_distance_at_fiducial_is_identity_for_gr() -> None:
    # xi_0 = 1 -> xi(z) = 1 everywhere, so polarizations and power are unchanged.
    catalog = _make_catalog()

    corrected = apply_gw_distance_at_fiducial(catalog, xi_0=1.0, xi_n=1.91)

    np.testing.assert_allclose(corrected.plus, catalog.plus)
    np.testing.assert_allclose(corrected.cross, catalog.cross)
    np.testing.assert_allclose(
        polarization_power(corrected), polarization_power(catalog)
    )


def test_apply_gw_distance_at_fiducial_scales_for_xi0_gt_1() -> None:
    catalog = _make_catalog()
    redshift = catalog.source_parameters["redshift"]
    xi_0, xi_n = 1.5, 1.91
    xi = np.exp(log_gw_em_ratio(redshift, xi_0=xi_0, xi_n=xi_n))

    corrected = apply_gw_distance_at_fiducial(catalog, xi_0=xi_0, xi_n=xi_n)

    np.testing.assert_allclose(corrected.plus, catalog.plus / xi[:, None])
    np.testing.assert_allclose(corrected.cross, catalog.cross / xi[:, None])
    np.testing.assert_allclose(
        polarization_power(corrected),
        polarization_power(catalog) / xi[None, :] ** 2,
    )


def test_apply_gw_distance_at_fiducial_is_pure_and_preserves_metadata() -> None:
    catalog = _make_catalog()

    corrected = apply_gw_distance_at_fiducial(catalog, xi_0=1.5, xi_n=1.91)

    assert corrected is not catalog
    np.testing.assert_array_equal(corrected.frequencies, catalog.frequencies)
    for name in catalog.source_parameters:
        np.testing.assert_array_equal(
            corrected.source_parameters[name], catalog.source_parameters[name]
        )
    assert corrected.approximant == catalog.approximant
    assert corrected.minimum_frequency == catalog.minimum_frequency
    assert corrected.maximum_frequency == catalog.maximum_frequency
    assert corrected.reference_frequency == catalog.reference_frequency
    assert corrected.sampling_frequency == catalog.sampling_frequency
    # The input catalog's arrays must not be modified in place.
    np.testing.assert_allclose(catalog.plus, _make_catalog().plus)
    np.testing.assert_allclose(catalog.cross, _make_catalog().cross)
