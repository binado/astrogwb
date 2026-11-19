Quickstart
==========

Load the LIGO Hanford and Livingston detector geometries and compute their
overlap reduction function on a frequency grid.

.. code-block:: python

   import numpy as np

   from astrogwb.detector import isotropic_overlap_reduction_function, load_detector

   frequencies = np.arange(10.0, 101.0)
   hanford = load_detector("H1")
   livingston = load_detector("L1")
   overlap = isotropic_overlap_reduction_function(frequencies, hanford, livingston)
   print(overlap[:3])

The dimensionless overlap gives the response of the detector pair to an
isotropic background at each frequency in Hz. It is normalized to one for
co-located, aligned detectors with orthogonal arms.

For detector noise curves, use :func:`astrogwb.detector.load_sensitivity`.
For a population prediction on a frequency grid, continue to
:doc:`../user_guide/catalog-inference`.
