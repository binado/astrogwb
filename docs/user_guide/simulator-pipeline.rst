Simulating detector-pair backgrounds
====================================

Supply a fixed pool of physical events as a mapping of column names to arrays. Each row
must describe one merger from the population whose **observer-frame rate**, in events
per second, is passed to the simulator. Apply any redshift or selection cuts to both the
pool and its rate. The pool size only controls how many distinct mergers are available.

.. code:: python

   from astrogwb.background.simulation import simulate_background
   from astrogwb.detector import load_network, network_pairs, register_network
   from astrogwb.waveform import waveform_generator

   register_network("HL", {"H1", "L1"})
   hl = load_network("HL")
   generator = waveform_generator(waveform_config)
   draw = simulate_background(
       key, sources, generator, hl,
       poisson_rate=0.02, num_segments=1024, segment_duration=16.0,
   )
   pairs = network_pairs(hl)
   counts = draw.segment_counts
   pair_spectra = draw.pair_spectral_density  # (detector pairs, frequencies)

The pool supplies the waveform inputs (including source-frame masses, redshift,
luminosity distance in Mpc, inclination and coalescence phase) and the projection angles
``right_ascension``, ``declination`` and ``polarization_angle`` in radians. Each segment
gets an independent Poisson count with mean ``poisson_rate * segment_duration``. The
simulator chooses rows uniformly without replacement across all segments and draws each
merger’s geocentric arrival time uniformly relative to its segment center. If the
realized total exceeds capacity, enlarge the pool and repeat with the same key.

Each detector receives both polarizations with its antenna response and geometric delay.
The code uses fixed sidereal geometry (GMST zero) and places the complete waveform in
the assigned segment. The output contains one row per detector pair in
``network_pairs(hl)`` order. Source chunks and segment batches bound working memory.

Command line
------------

``astrogwb simulate`` runs the simulator on a population catalogue (HDF5 or
CSV, canonical gwmock-pop column names) and writes one HDF5 file. HDF5
catalogues are read from ``source_parameters`` by default, a compound dataset
or a group of one dataset per column. Use ``--population-root`` to select a
different dataset or group path, or pass ``root`` to
:func:`~astrogwb.populations.io.read_population_file` in Python:

::

   astrogwb simulate 41.h5 events.h5 \
       --waveform-config waveform.toml --detector-config detectors.toml --network H1,L1 \
       --poisson-rate 0.00317 --num-segments 1024 --segment-duration 16 --seed 41

The output configuration records the population path and selected root.

The CLI passes ``jax.random.key(seed)`` directly to the simulator. Use a
different seed for each independent draw.

Pass one complete TOML file per option, with any layers and references resolved
before invoking the CLI. Use ``--network`` to list the detectors of the network by name. For example,
``--network H1,L1`` requires both ``[detectors.H1]`` and ``[detectors.L1]``
definitions in that file. To build that network in Python, use
``select_network(load_detectors_from_file(Path("detectors.toml")), ["H1", "L1"])``.

The catalogue must hold ``distance`` (luminosity distance in Mpc, renamed
``luminosity_distance``), the sky columns, and the detector-frame or source-frame
masses; other columns pass through. ``--waveform-config`` is a TOML file of the
``WaveformConfig`` fields. ``--poisson-rate`` is the observer-frame rate of the population the catalogue
represents, in events per second.
Each segment includes source interference. Add ``--simulate-detector-noise``
to also simulate detector noise. The HDF5 file stores the ``BackgroundDraw``
as root datasets ``frequencies``, ``segment_counts`` and ``pair_spectral_density``.
Root attributes include the ``SimulationConfig`` as JSON in ``metadata``,
the astrogwb ``version``, and ``format_version = 2``, which the reader checks.
``BackgroundDraw.write(path, config, **attrs)`` stores extra root attributes in the same
atomic write. ``BackgroundDraw.read_attrs`` reads them back. The on-disk config
stores the resolved detector definitions, so ``BackgroundDraw.read`` can
reconstruct the network directly. Use the stored network and frequency grid to evaluate
``effective_psd`` and ``pair_cross_power_std`` with the original observation duration.

Normalization and noise
-----------------------

Let :math:`K` be the sum of individual event cross-powers for a detector pair, :math:`C`
the sum of segment cross-powers of coherent strain plus detector noise, and :math:`T`
the number of segments times their duration :math:`\tau`. With noise enabled,
empty segments also receive noise. For a bin of width :math:`\Delta f` the returned
pair spectrum in inverse Hz is

.. math::


   Y = \frac{5}{T}\left[(1-r)K+rC\right], \qquad
   r=(\tau\Delta f)^{-1/2}.

Each bin must contain a positive whole number of segment Fourier modes. For an
isotropic unpolarized population, :math:`E[Y_{ab}]=\gamma_{ab} S_h`.
``astrogwb.statistics.cross_correlation_estimator`` combines a network’s rows using its
overlap functions and PSDs.

Use ``simulate_detector_noise=True`` to add detector noise from Python. Either
setting includes source interference. For the deterministic sum :math:`5K/T`
of individual powers from **all** supplied events, call
``project_background(sources, generator, hl, observation_seconds=T)``.

Repeated draws from one stored pool describe realizations conditional on that pool.
Generate independent pools to include population sampling scatter in ensemble
comparisons.
