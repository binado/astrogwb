"""Metadata-keyed generators: polarization-power catalogs and spectral-density draws.

- :mod:`astrogwb.simulators.core` -- content keys, the HDF5 layer and the cache,
  free of every physics package.
- :mod:`astrogwb.simulators.polarization_power` -- per-source waveform power and
  the plain catalog draw; :mod:`astrogwb.gwb.importance` places the draw at
  the redshift window's lower edge.
- :mod:`astrogwb.simulators.spectra` -- forward-model spectral-density draws.
"""
