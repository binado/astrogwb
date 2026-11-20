Background spectra and sensitivity
==================================

.. currentmodule:: astrogwb

Analytic spectra
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~background.inspiral_spectral_density

Fixed-mass population calculations
----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~background.analytic.PopulationFunction
   ~background.analytic.JointMassFunction
   ~background.analytic.uniform_prior_mass_moments
   ~background.analytic.precompute_cumulative_mass_moments
   ~background.analytic.analytic_spectral_density_from_mass_moments

Spectral conversions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~background.spectral.omega_gw_from_spectral_density
   ~background.spectral.spectral_density_from_omega_gw
   ~background.spectral.characteristic_strain
   ~background.spectral.characteristic_noise

Signal-to-noise ratio
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~background.snr.spectral_snr_squared_per_bin
   ~background.snr.spectral_snr_squared
   ~background.snr.spectral_snr_squared_per_log_frequency
   ~background.snr.spectral_snr
