Simulating detector-pair backgrounds
====================================

Supply a fixed pool of physical events as a mapping of column names to arrays. Each row
must describe one merger from the population whose **observer-frame rate**, in events
per second, is passed to the simulator. Apply any redshift or selection cuts to both the
pool and its rate. The pool size only controls how many distinct mergers are available.

.. code:: python

   from astrogwb.background.simulation import simulate_background
   from astrogwb.detector import load_network, network_pairs, register_network

   register_network("HL", {"H1", "L1"})
   hl = load_network("HL")
   draw = simulate_background(
       key, sources, waveform, networks={"HL": hl},
       observer_rate_hz=0.02, num_segments=1024, segment_seconds=16.0,
   )
   pairs = network_pairs(hl)
   counts = draw.segment_counts
   pair_spectra = draw.spectra["HL"]  # (detector pairs, frequencies)

The pool supplies the waveform inputs (including source-frame masses, redshift,
luminosity distance in Mpc, inclination and coalescence phase) and the projection angles
``right_ascension``, ``declination`` and ``polarization_angle`` in radians. Each segment
gets an independent Poisson count with mean ``observer_rate_hz * segment_seconds``. The
simulator chooses rows uniformly without replacement across all segments and draws each
merger’s geocentric arrival time uniformly relative to its segment center. If the
realized total exceeds capacity, enlarge the pool and repeat with the same key.

Each detector receives both polarizations with its antenna response and geometric delay.
The code uses fixed sidereal geometry (GMST zero) and places the complete waveform in
the assigned segment. Network pairs follow ``network_pairs(hl)`` in the example above. A joint
run uses one event and detector-noise realization, computes each requested pair once,
and reuses shared pair rows. It requires the same definition for a detector name
wherever it appears. Source chunks and segment batches bound working memory.

Command line
------------

``astrogwb simulate`` runs the simulator on a gwmock-pop population catalogue (HDF5 or
CSV, canonical gwmock-pop column names) and writes one HDF5 file:

::

   astrogwb simulate 41-0.h5 events.h5 \
       --waveform-config waveform.toml --detector-config detectors.toml \
       --rate-per-year 1e5 --num-segments 1024 --segment-seconds 16 --seed 41

Pass one complete TOML file per option, with any layers and references resolved
before invoking the CLI. The CLI runs every named network in the detector file's
``[networks]`` table. For example, ``HL = ["H1", "L1"]`` requires both
``[detectors.H1]`` and ``[detectors.L1]`` definitions in that file. To load
that network in Python, use ``load_networks_from_file(Path("detectors.toml"))["HL"]``.

The catalogue must hold ``distance`` (luminosity distance in Mpc, renamed
``luminosity_distance``), the sky columns, and the detector-frame or source-frame
masses; other columns pass through. ``--waveform-config`` is a TOML file of the
``WaveformMetadata`` fields. ``--rate-per-year`` is the observer-frame rate of the
population the catalogue represents; it is converted to ``observer_rate_hz``.
``--incoherent`` skips noise and sums event powers, the noiseless expected signal of the
same Poisson draw. The file stores the ``BackgroundDraw`` (``frequencies``,
``segment_counts`` and each network’s ``pair_spectral_density``) and, as JSON in its
``metadata`` attribute, the ``SimulationConfig`` it was made with, the astrogwb
``version`` (informational) and an integer ``format_version`` that
``BackgroundDraw.read`` keys on, so a patch release does not invalidate existing files.
``BackgroundDraw.write(path, config, **attrs)`` stores extra root attributes in the same
atomic write; ``BackgroundDraw.read_attrs`` reads them back. The on-disk config stores
shared detector definitions once, alongside the network memberships.
``BackgroundDraw.read`` reconstructs the network mappings without looking up any
registrations, and ``network_noise`` rebuilds the noise inputs from those configs.

Normalization and noise
-----------------------

Let :math:`K` be the sum of individual event cross-powers for a detector pair, :math:`C`
the sum of segment cross-powers of coherent strain plus detector noise, and :math:`T`
the number of segments times their duration :math:`\tau`. Empty segments also receive
noise. For a bin of width :math:`\Delta f` the returned pair spectrum in inverse Hz is

.. math::


   Y = \frac{5}{T}\left[(1-r)K+rC\right], \qquad
   r=(\tau\Delta f)^{-1/2}.

Each noisy bin must contain a positive whole number of segment Fourier modes. For an
isotropic unpolarized population, :math:`E[Y_{ab}]=\gamma_{ab} S_h`.
``astrogwb.statistics.cross_correlation_estimator`` combines a network’s rows using its
overlap functions and PSDs.

Use ``incoherent=True`` for a Poisson realization of the individual event powers alone.
To project **all** supplied events deterministically, call ``project_background(sources,
waveform, networks={"HL": hl}, observation_seconds=T)``. Both signal-only calculations return
:math:`5K/T`.

Repeated draws from one stored pool describe realizations conditional on that pool.
Generate independent pools to include population sampling scatter in ensemble
comparisons.
