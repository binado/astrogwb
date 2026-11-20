"""The Gaussian likelihood against ``pygwb.pe.PowerLawModel``.

Without shot noise the dense likelihood is pygwb's diagonal one in other
units: every bin's residual over its standard deviation is the same number.
The two normalizations differ by the parameter-independent Jacobian of the
change of units, so log-likelihood differences between parameter points are
compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PYGWB_POWER_FRACTION
from jax.typing import ArrayLike
from numpy.typing import NDArray
from tolerances import assert_close

from astrogwb.background.spectral import spectral_density_from_omega_gw
from astrogwb.detector import Detector, pair_names
from astrogwb.inference import DiagonalGaussianLikelihood, SpectralDensityPrediction


@dataclass(frozen=True)
class PowerLawSpectrum:
    """A power-law energy spectrum, a spectral-density model without shot noise."""

    frequencies: jax.Array
    reference_frequency: float
    hubble_constant: float

    def __call__(
        self, params: Mapping[str, ArrayLike], *, shot_noise: bool
    ) -> SpectralDensityPrediction:
        if shot_noise:
            raise ValueError("a power law has no shot noise")
        omega = jnp.asarray(params["omega_ref"]) * (
            self.frequencies / self.reference_frequency
        ) ** jnp.asarray(params["alpha"])
        density = spectral_density_from_omega_gw(
            omega, self.frequencies, hubble_constant=self.hubble_constant
        )
        return SpectralDensityPrediction(density, None, {})


jax.tree_util.register_dataclass(
    PowerLawSpectrum,
    data_fields=["frequencies"],
    meta_fields=["reference_frequency", "hubble_constant"],
)


@pytest.fixture
def likelihood(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
    observation_years: float,
    exact_detectors: tuple[Detector, ...],
) -> DiagonalGaussianLikelihood[PowerLawSpectrum]:
    model = PowerLawSpectrum(
        jnp.asarray(frequencies),
        float(pygwb_reference["reference_frequency"]),
        hubble_constant,
    )
    csd = pygwb_reference["csd"] / PYGWB_POWER_FRACTION
    return DiagonalGaussianLikelihood(
        model,
        exact_detectors,
        frequencies=frequencies,
        observation_time=observation_years,
        observed=dict(zip(pair_names(exact_detectors), csd, strict=True)),
    )


def test_log_likelihood_differences_match_pygwb(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    likelihood: DiagonalGaussianLikelihood[PowerLawSpectrum],
) -> None:
    values = jnp.stack(
        [
            likelihood.log_likelihood({"omega_ref": omega_ref, "alpha": alpha})[0]
            for omega_ref, alpha in pygwb_reference["likelihood_points"]
        ]
    )

    reference = pygwb_reference["likelihood_log_values"]
    assert_close(values - values[0], reference - reference[0])
