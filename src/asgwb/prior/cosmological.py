from __future__ import annotations

import operator
from abc import abstractmethod

import numpy as np
from bilby.gw.prior import Cosmological, UniformSourceFrame

from .redshift import (
    AVAILABLE_TIME_DELAY_MODELS,
    TimeDelayPdf,
    madau_dickinson_source_frame_distribution,
    power_law_source_frame_distribution,
    redshift_pdf,
)

DEFAULT_NUM_INTERP = 500


class UniformSourceFramePrior(UniformSourceFrame):
    """Uniform-in-source-frame redshift prior."""

    model_type = "redshift"
    model_name = "uniform_source_frame"


class ParametrizedCosmological(Cosmological):
    """Abstract base for parametrized cosmological redshift priors.

    Subclasses must define:
    - ``model_name``: identifier string
    - ``default_model_parameters``: dict of parameter names to default values
    - ``get_source_frame_distribution(z)``: returns the source-frame merger rate array

    Parameters
    ----------
    *args:
        Passed to ``bilby.gw.prior.Cosmological``.
    num_interp:
        Number of interpolation points for the redshift grid.
    time_delay_fn:
        Optional time-delay model for convolution. Must expose
        ``pdf(time_delay_array)`` and return a normalized PDF.
        Keys in ``AVAILABLE_TIME_DELAY_MODELS`` are accepted string shortcuts.
    **kwargs:
        Model parameters (e.g. ``lamb=2.9``) are extracted and the rest are
        forwarded to ``bilby.gw.prior.Cosmological``.
    """

    model_type = "redshift"
    model_name = ""
    default_model_parameters: dict[str, float] = {}

    def __init__(
        self,
        *args,
        num_interp: int = DEFAULT_NUM_INTERP,
        time_delay_fn: TimeDelayPdf | str | None = None,
        **kwargs,
    ) -> None:
        self.num_interp = num_interp
        self.model_parameters = self._extract_model_parameters(**kwargs)
        if isinstance(time_delay_fn, str):
            try:
                time_delay_fn = AVAILABLE_TIME_DELAY_MODELS[time_delay_fn]
            except KeyError:
                valid_keys = ", ".join(sorted(AVAILABLE_TIME_DELAY_MODELS))
                raise ValueError(
                    f"Unknown time_delay_fn {time_delay_fn!r}. "
                    f"Valid options are: {valid_keys}"
                ) from None
        self.time_delay_fn = time_delay_fn
        for key in self.model_parameters:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    @classmethod
    def _extract_model_parameters(cls, **kwargs) -> dict[str, float]:
        """Merge default_model_parameters with any matching kwargs."""
        params = dict(cls.default_model_parameters)
        for key in params:
            if key in kwargs:
                params[key] = kwargs[key]
        return params

    @abstractmethod
    def get_source_frame_distribution(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_redshift_array_and_pdf(self, normalize: bool = True):
        z_min, z_max = self._minimum["redshift"], self._maximum["redshift"]
        zs = np.linspace(z_min * 0.99, z_max * 1.01, self.num_interp)
        source_frame_distribution = self.get_source_frame_distribution(zs)
        return zs, redshift_pdf(
            zs,
            self.cosmology,
            source_frame_distribution,
            self.time_delay_fn,
            z_min=z_min,
            z_max=z_max,
            normalize=normalize,
        )

    def _get_redshift_arrays(self):
        return self._get_redshift_array_and_pdf(normalize=True)

    def get_redshift_arrays(self, normalize: bool = True):
        """Public access to the redshift grid and PDF arrays."""
        return self._get_redshift_array_and_pdf(normalize=normalize)


class PowerLawRedshiftPrior(ParametrizedCosmological):
    """Redshift prior with a power-law source-frame merger rate: (1+z)^lamb."""

    model_name = "power_law"
    default_model_parameters = {"lamb": 2.9}

    def get_source_frame_distribution(self, z: np.ndarray) -> np.ndarray:
        (lamb,) = (operator.itemgetter("lamb")(self.model_parameters),)
        return power_law_source_frame_distribution(z, lamb)


class MadauDickinsonRedshiftPrior(ParametrizedCosmological):
    """Redshift prior following the Madau-Dickinson star formation rate."""

    model_name = "madau_dickinson"
    default_model_parameters = {"gamma": 2.7, "kappa": 5.7, "z_peak": 2}

    def get_source_frame_distribution(self, z: np.ndarray) -> np.ndarray:
        kappa, gamma, z_peak = operator.itemgetter("kappa", "gamma", "z_peak")(
            self.model_parameters
        )
        return madau_dickinson_source_frame_distribution(z, kappa, gamma, z_peak)
