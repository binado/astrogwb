Configuration and numerical utilities
=====================================

.. currentmodule:: astrogwb

Configuration and runtime
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~config.merge_layers
   ~config.require_toml
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

   ~frequency.FrequencySpacing
   ~frequency.bin_widths
   ~frequency.broadcast_along_axis
   ~frequency.frequency_grid
   ~frequency.frequency_mask
   ~frequency.log_frequency_grid
   ~frequency.uniform_frequency_grid
   ~frequency.validate_frequency_grid

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~utils.require_current_version
   ~utils.years_to_seconds
   ~utils.require_x64
   ~utils.cumulative_trapezoid
   ~utils.gauss_legendre_nodes_weights
   ~utils.stack_pairs
   ~utils.unstack_pairs
   ~utils.io.Arrays
   ~utils.io.atomic_h5
   ~utils.io.load
   ~utils.io.read_tree
   ~utils.io.write
   ~utils.io.write_tree
   ~utils.numpyro.InverseCDF
   ~utils.numpyro.Sampling
   ~utils.numpyro.sample_and_logprob
   ~utils.numpyro.sample_model
   ~utils.numpyro.sample_sources_qmc
