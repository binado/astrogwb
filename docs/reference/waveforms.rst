Waveforms
=========

.. currentmodule:: astrogwb

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~waveform.WaveformGenerator
   ~waveform.WaveformConfig
   ~waveform.chirp_mass
   ~waveform.inclination_factor
   ~waveform.inspiral_intensity
   ~waveform.intensity
   ~waveform.quadrupolar_polarizations
   ~waveform.quadrupolar_stokes_factors
   ~waveform.termination_frequency
   ~waveform.waveform_generator

Generator implementations and Stokes conventions
-------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~waveform.analytical.AnalyticInspiralGenerator
   ~waveform.ripple.RippleGenerator

.. autodata:: astrogwb.waveform.intensity.StokesPair

   Supported Stokes parameter pairs for the quadrupolar response.

.. autodata:: astrogwb.waveform.intensity.QUADRUPOLE_STOKES_WEIGHTS

   Stokes weights for a quadrupolar waveform averaged over inclination and polarization angle.
