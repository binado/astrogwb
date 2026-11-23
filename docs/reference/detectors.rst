Detectors
=========

.. currentmodule:: astrogwb

Construct a custom detector with ``Detector(name="S1", geometry=geometry,
psd_reference="ET_D_psd")`` and register it with
``register_detector(detector)``. The PSD curve loads on the first call to
``detector.sensitivity(frequencies)``.

Standalone detector dumps include ``name``. Simulation metadata uses the detector
mapping key as the name. A network passed to a calculation must be keyed by the
name of each detector and hold at least two;
:func:`~astrogwb.detector.network_pairs` raises ``ValueError`` otherwise.

A network is a mapping from detector names to detectors, such as
``{name: load_detector(name) for name in names}``.
Pass that mapping directly to detector-network
calculations. Array rows follow :func:`~astrogwb.detector.network_pairs`,
regardless of the mapping's insertion order.

Use :func:`~astrogwb.detector.map_over_pairs` to evaluate a custom observable
and keep its detector-pair keys.

.. code-block:: python

   import numpy as np
   from astrogwb.detector import (
       isotropic_overlap_reduction_function,
       load_detector,
       map_over_pairs,
   )

   frequencies = np.arange(10.0, 101.0)
   network = {name: load_detector(name) for name in ("V1", "H1", "L1")}
   overlaps = map_over_pairs(
       lambda first, second: isotropic_overlap_reduction_function(
           frequencies, first, second
       ),
       network,
   )

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~detector.Detector
   ~detector.DetectorGeometry
   ~detector.Network
   ~detector.Sensitivity
   ~detector.effective_psd
   ~detector.isotropic_overlap_reduction_function
   ~detector.load_detector
   ~detector.load_detectors_from_file
   ~detector.load_sensitivity
   ~detector.load_sensitivity_map
   ~detector.map_over_pairs
   ~detector.network_overlap_reduction_function
   ~detector.network_pairs
   ~detector.point_source_overlap_reduction_function
   ~detector.pair_cross_power_std
   ~detector.pair_response
   ~detector.pair_weights
   ~detector.register_detector

Advanced response and noise calculations
----------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~detector.generalized_overlap_reduction_function
   ~detector.stokes_cross_power
   ~detector.align_detector
   ~detector.relative_orientation
   ~detector.network_cross_power_std
   ~detector.resolve_psd_path

.. autodata:: astrogwb.detector.NORMALIZED_ORF_SCALE

   Factor converting the sky-averaged tensor response to the normalized overlap reduction function.

.. autodata:: astrogwb.detector.OrientationReference

   Reference direction for relative detector orientations.

Projection
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~detector.projection.antenna_pattern
   ~detector.projection.time_delay_from_geocenter
   ~detector.projection.project_polarizations_fd
