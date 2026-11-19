Public API
==========

The reference uses the package-level import path for re-exported objects.
For command-line usage, see :doc:`../user_guide/simulator-pipeline`.

Package metadata
----------------

.. autodata:: astrogwb.__version__


Background spectra and sensitivity
----------------------------------

.. automodule:: astrogwb.background.analytic
   :members:

.. automodule:: astrogwb.background.spectral
   :members:

.. automodule:: astrogwb.background.snr
   :members:

Reference catalogs and population models
----------------------------------------

.. automodule:: astrogwb.background.importance.catalog
   :members:

.. automodule:: astrogwb.background.importance.spectrum
   :members:

.. automodule:: astrogwb.background.importance.model
   :members:

.. automodule:: astrogwb.background.importance.diagnostics
   :members:

.. automodule:: astrogwb.populations
   :members:
   :imported-members:
   :exclude-members: coba
   :undoc-members:

.. automodule:: astrogwb.populations.coba
   :members:

.. automodule:: astrogwb.populations.mass.gaussian_mass_pair
   :members:

.. automodule:: astrogwb.populations.mass.uniform_mass_pair
   :members:

Probability distributions
-------------------------

.. automodule:: astrogwb.distributions
   :members:
   :imported-members:

.. automodule:: astrogwb.distributions.config
   :members:

.. automodule:: astrogwb.distributions.interpolated
   :members:

.. automodule:: astrogwb.distributions.redshift
   :members:
   :imported-members:

Cosmology and merger rates
--------------------------

.. automodule:: astrogwb.cosmology
   :members:

.. automodule:: astrogwb.rates
   :members:

Waveforms
---------

.. automodule:: astrogwb.waveform
   :members:
   :imported-members:

.. autodata:: astrogwb.waveform.intensity.StokesPair

   Supported Stokes parameter pairs for the quadrupolar response.

.. autodata:: astrogwb.waveform.intensity.QUADRUPOLE_STOKES_WEIGHTS

   Stokes weights for a quadrupolar waveform averaged over inclination and
   polarization angle.

Detectors
---------

.. automodule:: astrogwb.detector
   :members:
   :imported-members:

.. autodata:: astrogwb.detector.DETECTOR_DEFAULT_PATHS
   :no-value:

   Packaged detector configuration layers loaded by default.

.. autodata:: astrogwb.detector.NORMALIZED_ORF_SCALE

   Factor converting the sky-averaged tensor response to the normalized
   overlap reduction function.

.. autodata:: astrogwb.detector.OrientationReference

   Reference direction for relative detector orientations.

.. automodule:: astrogwb.detector.projection
   :members:

Simulation
----------

.. automodule:: astrogwb.background.simulation
   :members:
   :imported-members:

Inference
---------

.. automodule:: astrogwb.inference
   :members:
   :imported-members:

Statistics
----------

.. automodule:: astrogwb.statistics
   :members:
   :imported-members:

Configuration and numerical utilities
-------------------------------------

.. automodule:: astrogwb.config
   :members:

.. automodule:: astrogwb.constants
   :members:
   :undoc-members:

.. automodule:: astrogwb.frequency
   :members:

.. automodule:: astrogwb.runtime
   :members:

.. automodule:: astrogwb.utils
   :members:

.. automodule:: astrogwb.utils.io
   :members:

.. automodule:: astrogwb.utils.numpyro
   :members:

Command-line entry point
------------------------

.. py:currentmodule:: astrogwb.cli

.. py:data:: app
   :type: typer.Typer

   Typer application behind the ``astrogwb`` command.

.. automodule:: astrogwb.cli.simulate
   :members:
