# Simulating detector-pair backgrounds

Supply a fixed pool of physical events as a mapping of column names to arrays.
Each row must describe one merger from the population whose **observer-frame
rate**, in events per second, is passed to the simulator. Apply any redshift
or selection cuts to both the pool and its rate. The pool size only controls
how many distinct mergers are available.

```python
from astrogwb.simulators import simulate_background

draw = simulate_background(
    key, sources, waveform, detectors,
    observer_rate_hz=0.02, num_segments=1024, segment_seconds=16.0,
)
counts = draw.segment_counts
pair_spectra = draw.spectra[network_name]  # (detector pairs, frequencies)
```

The pool supplies the waveform inputs (including source-frame masses,
redshift, luminosity distance in Mpc, inclination and coalescence phase)
and the projection angles `right_ascension`, `declination` and
`polarization_angle` in radians. Each segment gets an independent Poisson
count with mean `observer_rate_hz * segment_seconds`. The simulator chooses
rows uniformly without replacement across all segments and draws each
merger's geocentric arrival time uniformly relative to its segment center.
If the realized total exceeds capacity, enlarge the pool and repeat with
the same key.

Each detector receives both polarizations with its antenna response and
geometric delay. The code uses fixed sidereal geometry (GMST zero) and
places the complete waveform in the assigned segment. Network pairs follow
`detectors.pairs(network_name)`. Source chunks and segment batches bound
working memory.

## Normalization and noise

Let $K$ be the sum of individual event cross-powers for a detector pair,
$C$ the sum of segment cross-powers of coherent strain plus detector noise,
and $T$ the number of segments times their duration $\tau$. Empty segments
also receive noise. For a bin of width $\Delta f$ the returned pair spectrum
in inverse Hz is

$$
Y = \frac{5}{T}\left[(1-r)K+rC\right], \qquad
r=(\tau\Delta f)^{-1/2}.
$$

Each noisy bin must contain a positive whole number of segment Fourier modes.
For an isotropic unpolarized population, $E[Y_{ab}]=\gamma_{ab} S_h$.
`combine_pair_spectra` combines a network's rows using its overlap functions
and PSDs.

Use `incoherent=True` for a Poisson realization of the individual event
powers alone. To project **all** supplied events deterministically, call
`project_background(sources, waveform, detectors, observation_seconds=T)`.
Both signal-only calculations return $5K/T$.

Repeated draws from one stored pool describe realizations conditional on
that pool. Generate independent pools to include population sampling scatter
in ensemble comparisons.
