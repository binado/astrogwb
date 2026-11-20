Inference
=========

.. currentmodule:: astrogwb

Use :class:`~astrogwb.inference.DiagonalGaussianLikelihood` for detector noise
alone and :class:`~astrogwb.inference.DenseGaussianLikelihood` to include
background shot noise. Both take the observation duration in seconds.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~inference.DenseGaussianLikelihood
   ~inference.DiagonalGaussianLikelihood
   ~inference.SpectralDensityModel
   ~inference.SpectralDensityPrediction
   ~inference.grid_log_likelihood
   ~inference.grid_log_prior
