from functools import cached_property
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from bilby.core.likelihood import Likelihood

from asgwb.cosmology import Cosmology, get_cosmology
from asgwb.detector import Detector
from asgwb.detector.overlap import pairwise_overlap_reduction_function
from asgwb.prior.intrinsic import IntrinsicPriorDictGenerator
from asgwb.waveform import WaveformGenerator


class FiducialSpectralDensityGenerator(Protocol):
    def __call__(self, frequencies: npt.NDArray) -> npt.NDArray: ...


class SGWBGaussianLikelihood(Likelihood):
    def __init__(
        self,
        waveform_generator: WaveformGenerator,
        detectors: Sequence[Detector],
        fiducial_spectral_density_generator: FiducialSpectralDensityGenerator,
        intrinsic_prior_dict_generator: IntrinsicPriorDictGenerator,
        mc_integral_npoints: int = 256,
        observation_time: float = 1,
        seed: int = 42,
    ):
        super().__init__()
        if mc_integral_npoints <= 0:
            raise ValueError(
                f"mc_integral_npoints must be positive, got {mc_integral_npoints}"
            )
        self.detectors = detectors
        self.waveform_generator = waveform_generator
        self.fiducial_spectral_density_generator = fiducial_spectral_density_generator
        self.intrinsic_prior_dict_generator = intrinsic_prior_dict_generator
        self.mc_integral_npoints = mc_integral_npoints
        self.observation_time = observation_time
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def cosmology(parameters: dict[str, float]) -> Cosmology:
        return get_cosmology(parameters)

    @cached_property
    def frequencies(self) -> npt.NDArray:
        return self.waveform_generator.grid.frequencies

    @cached_property
    def inverse_covariance(self) -> npt.NDArray:
        orf_pair = pairwise_overlap_reduction_function(self.frequencies, self.detectors)
        psds = np.stack([det.psd(self.frequencies) for det in self.detectors], axis=-1)
        psd_prod = np.einsum("ij,ik->ijk", psds, psds)
        inv_psd_prod = 1 / psd_prod
        # orf_pair is shape (ndet, ndet, nf), we want to move the frequency axis to the front
        out_before_sum = np.moveaxis(orf_pair, -1, 0) * inv_psd_prod
        # sum over upper triangle
        ndet = len(self.detectors)
        det_i, det_j = np.triu_indices(ndet, k=1)
        return out_before_sum[:, det_i, det_j].sum(axis=-1)

    @cached_property
    def fiducial_spectral_density(self) -> npt.NDArray:
        return self.fiducial_spectral_density_generator(self.frequencies)

    @cached_property
    def log_2observation_time(self) -> float:
        return np.log(2 * self.observation_time)

    def energy_flux(self, injection: dict[str, float]) -> npt.NDArray:
        # We compute the energy flux for a face-on source at luminosity distance 1Mpc.
        injection.update({"luminosity_distance": 1, "theta_jn": 0})
        _ = injection.pop("redshift", None)
        polarizations = self.waveform_generator.frequency_domain_polarizations(
            injection
        )

        return polarizations.squared_sum()

    def spectral_density(self, parameters: dict[str, float]) -> npt.NDArray:
        """
        Compute the spectral density over the intrinsic parameter prior volume using Monte Carlo integration.
        """
        intrinsic_prior = self.intrinsic_prior_dict_generator(parameters)
        samples = intrinsic_prior.sample(self.mc_integral_npoints)
        sample_keys = list(intrinsic_prior.keys())
        samples_df = pd.DataFrame(samples, columns=sample_keys)
        z = np.asarray(samples_df["redshift"])
        cosmology = self.cosmology(parameters)
        luminosity_distance = cosmology.luminosity_distance(z)
        dgw = self.gravitational_wave_distance(z, luminosity_distance, parameters)

        out = np.zeros_like(self.frequencies)
        for sample, dgw_per_sample in zip(
            samples_df.itertuples(index=False, name=None), dgw, strict=True
        ):
            injection = dict(zip(sample_keys, sample, strict=True))
            flux = self.energy_flux(injection)
            out += flux / dgw_per_sample**2

        out /= self.mc_integral_npoints
        # Factor of 2 / 5 accounts for averaging over inclination angle
        out *= 0.4
        return out

    def omega_gw(self, parameters: dict[str, float]) -> npt.NDArray:
        spectral_density = self.spectral_density(parameters)
        h0 = parameters["H0"]
        f = self.frequencies
        return (4 / 3) * np.pi * f**3 * spectral_density / h0**2

    def gravitational_wave_distance(
        self,
        redshift: npt.NDArray,
        luminosity_distance: npt.NDArray,
        parameters: dict[str, float],
    ) -> npt.NDArray:
        chi0, chin = parameters["chi0"], parameters["chin"]
        chiz = chi0 + (1 - chi0) / (1 + redshift) ** chin
        return chiz * luminosity_distance

    def log_likelihood(self, parameters: dict[str, float] | None = None) -> float:
        """Compute the Gaussian log-likelihood.

        Parameters
        ----------
        parameters : dict[str, float] | None
            Model parameters. This likelihood requires explicit parameters and
            raises a ValueError when None is provided.
        """
        if parameters is None:
            raise ValueError("parameters must be provided")
        spectral_density = self.spectral_density(parameters)
        fiducial_spectral_density = self.fiducial_spectral_density
        integrand = (
            spectral_density - fiducial_spectral_density
        ) ** 2 * self.inverse_covariance
        chi_squared = np.trapezoid(integrand, x=self.frequencies).astype(float).item()
        return -0.5 * chi_squared + self.log_2observation_time
