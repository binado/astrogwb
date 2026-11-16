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
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PYGWB_POWER_FRACTION
from jax.typing import ArrayLike
from numpy.typing import NDArray

from astrogwb.background.spectral import spectral_density_from_omega_gw
from astrogwb.detector import Detector, pair_bin_scale, pair_terms
from astrogwb.inference import DenseGaussianLikelihood, Network
from astrogwb.inference.model import CatalogModel, Prediction


@dataclass(frozen=True)
class PowerLawSpectrum:
    """A power-law energy spectrum, standing in for a catalog model."""

    frequencies: jax.Array
    reference_frequency: float
    hubble_constant: float

    def predict(
        self, params: Mapping[str, ArrayLike], *, shot_noise: bool
    ) -> Prediction:
        del shot_noise
        omega = jnp.asarray(params["omega_ref"]) * (
            self.frequencies / self.reference_frequency
        ) ** jnp.asarray(params["alpha"])
        mean = spectral_density_from_omega_gw(
            omega, self.frequencies, hubble_constant=self.hubble_constant
        )
        return Prediction(mean=mean, power_moment=None, extras={})


@pytest.fixture
def likelihood(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
    observation_years: float,
    exact_detectors: tuple[Detector, ...],
) -> DenseGaussianLikelihood:
    network = Network.from_pairs(
        pair_bin_scale(frequencies, exact_detectors, observation_years),
        np.stack([term.overlap for term in pair_terms(frequencies, exact_detectors)]),
    )
    model = PowerLawSpectrum(
        jnp.asarray(frequencies),
        float(pygwb_reference["reference_frequency"]),
        hubble_constant,
    )
    return DenseGaussianLikelihood(
        model=cast(CatalogModel, model),
        network=network,
        observed=network.compress(pygwb_reference["csd"] / PYGWB_POWER_FRACTION),
        shot_noise=False,
    )


def test_log_likelihood_differences_match_pygwb(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    likelihood: DenseGaussianLikelihood,
) -> None:
    values = jnp.stack(
        [
            likelihood.log_likelihood({"omega_ref": omega_ref, "alpha": alpha})[0]
            for omega_ref, alpha in pygwb_reference["likelihood_points"]
        ]
    )

    reference = pygwb_reference["likelihood_log_values"]
    np.testing.assert_allclose(
        values - values[0], reference - reference[0], rtol=0.0, atol=1e-4
    )
