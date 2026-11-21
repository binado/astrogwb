Configuration and numerical utilities
=====================================

.. currentmodule:: astrogwb

Runtime
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~runtime.configure_runtime

Physical constants
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~constants.GPC_IN_METERS
   ~constants.GRAVITATIONAL_CONSTANT
   ~constants.ISCO_ALPHA
   ~constants.MPC_IN_METERS
   ~constants.MPC_IN_SECONDS
   ~constants.SECONDS_PER_YEAR
   ~constants.SOLAR_MASS_IN_KILOGRAMS
   ~constants.SOLAR_MASS_IN_SECONDS
   ~constants.SPEED_OF_LIGHT

Frequency grids
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~frequency.frequency_grid
   ~frequency.bin_widths
   ~frequency.FrequencySpacing
   ~frequency.frequency_mask

Advanced utilities
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~utils.require_current_version
   ~utils.years_to_seconds
   ~utils.require_x64
   ~utils.stack_pairs
   ~utils.unstack_pairs
   ~utils.numpyro.InverseCDF
   ~utils.numpyro.Sampling
   ~utils.numpyro.sample_and_logprob
   ~utils.numpyro.sample_model
   ~utils.numpyro.sample_sources_qmc
