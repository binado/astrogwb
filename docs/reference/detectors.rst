Detectors
=========

.. currentmodule:: astrogwb

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~detector.Detector
   ~detector.DetectorConfig
   ~detector.DetectorGeometry
   ~detector.DetectorNetworkConfig
   ~detector.PairTerms
   ~detector.Sensitivity
   ~detector.align_detector
   ~detector.effective_psd
   ~detector.gaussian_bin_scale
   ~detector.generalized_overlap_reduction_function
   ~detector.isotropic_overlap_reduction_function
   ~detector.load_detector
   ~detector.load_network
   ~detector.load_sensitivity
   ~detector.load_sensitivity_map
   ~detector.log_frequency_noise_scale
   ~detector.overlap_reduction_function
   ~detector.pair_bin_scale
   ~detector.pair_names
   ~detector.pair_response
   ~detector.pair_terms
   ~detector.pairwise_overlap_reduction_function
   ~detector.register_detector
   ~detector.register_network
   ~detector.relative_orientation
   ~detector.resolve_network
   ~detector.resolve_psd_path
   ~detector.sample_pair_response
   ~detector.stokes_cross_power

.. autodata:: astrogwb.detector.DETECTOR_DEFAULT_PATHS
   :no-value:

   Packaged detector configuration layers loaded by default.

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
