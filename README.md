# astrogwb

`astrogwb` provides reusable scientific modules for modelling the stochastic
gravitational-wave background from compact-binary populations and performing
Bayesian inference with detector networks.

## Installation

```bash
pip install astrogwb
```

Optional accelerator builds are available as `astrogwb[cuda]` and
`astrogwb[tpu]`. Direct HDF5 catalog serialization is opt-in:

```bash
pip install astrogwb[io]
```

## Library modules

- `astrogwb.constants` holds the SI physical constants and unit conversions
  every other module shares. The tabulated values are the LALSuite literals,
  so results are bit-comparable with LALSimulation and ripple.
- `astrogwb.detector` loads bundled detector geometry and sensitivity data and
  evaluates overlap-reduction functions and effective PSDs.
- `astrogwb.background` provides spectral-density, Omega-GW conversion, and SNR
  calculations. `astrogwb.background.analytic` evaluates the inspiral-only
  background in closed form, truncated on the same dimensionless `alpha` as
  `astrogwb.waveform`.
- `astrogwb.populations` declares source populations as NumPyro models,
  addressed by registered name. One declaration serves both generation and
  density evaluation, so a catalog and the weights that reweight it can never
  disagree about the law behind it.
- `astrogwb.importance` owns importance weighting: `astrogwb.importance.catalog`
  draws fixed-size reference catalogs and their intensity,
  `astrogwb.importance.spectrum` reweights their intrinsic draws to a target
  population and integrates redshift by closed-form quadrature, and
  `astrogwb.importance.diagnostics` reports effective sample sizes.
- `astrogwb.inference` exposes the caller-prepared NumPyro model.
- `astrogwb.catalog` provides an array-native catalog that records the
  population that drew it, and reads and writes it as HDF5 behind the `io`
  extra.
- `astrogwb.waveform` owns intensity generators, reduces raw
  plus/cross polarizations to power, applies GW-distance corrections to plain
  arrays, and provides a closed-form quadrupolar inspiral model. Persistence
  and labelled-array policy stay with applications.
  `astrogwb.waveform` gives the same power in closed form for a
  quadrupolar, inspiral-only binary, truncated at `f = alpha / ((1 + z) M)`
  for a caller-chosen dimensionless `alpha`.

For example:

```python
from astrogwb.detector import load_detector, load_sensitivity

hanford = load_detector("H1")
sensitivity = load_sensitivity("H1")
```

A reference catalog draws intrinsic sources, computes their intensities, and
stores them together with the declaration that produced them:

```python
import jax

from astrogwb.importance.catalog import CatalogMetadata
from astrogwb.importance.spectrum import reference_catalog
from astrogwb.populations import BNSCobaPopulation
from astrogwb.utils.io import write
from astrogwb.waveform import WaveformMetadata

# A population record's fields are its construction settings; build() returns
# hyperparameters -> (redshift distribution, source model).
population = BNSCobaPopulation(
    mass_model="uniform", minimum_redshift=0.01, maximum_redshift=20.0, n_grid=4096
)
params = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 2.35,
    "kappa": 4.4,
    "z_peak": 2.14,
    "local_merger_rate": 240.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}
waveform = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=2.0,
    maximum_frequency=2048.0,
    reference_frequency=2.0,
    sampling_frequency=4096.0,
    frequency_resolution=1.0,
)
metadata = CatalogMetadata(
    waveform=waveform, population=population, fiducials=params, num_samples=1024
)
reference = reference_catalog(metadata, jax.random.key(42))
write("catalog.h5", reference, metadata, seed=42)
```

A spectrum prediction uses the reference catalog, with every source placed at
the lower edge of the redshift window and face-on, so that
a source at any redshift is a rescaling of its reference power. Intrinsic
parameters are reweighted to a target population; redshift is integrated by
closed-form quadrature, so it needs no weights at all.

```python
import numpy as np

from astrogwb.inference import CatalogModel

model = CatalogModel.from_catalog(
    reference,
    metadata,
    population=population.build(),
    frequencies=np.arange(10.0, 1000.0, 1.0),
    num_redshift_nodes=32,
    observation_time=1.0,
    density_sites=("source_frame_mass_1", "source_frame_mass_2"),
)
spectrum, power_moment, extras = model.predict(
    {**params, "H0": 70.0, "xi_0": 1.2, "xi_n": 1.91}
)
```

`spectrum` is the background's mean and `power_moment` the shot-noise moment of
a Poisson catalog over the observation time. A detector network's likelihood of
its per-pair cross powers is the dense Gaussian with covariance
`Sigma + power_moment * R` (elementwise), with `Sigma` the detector noise and
`R` the network's response moment:

```python
from astrogwb.detector import pair_response_moment
from astrogwb.inference import DenseGaussianLikelihood, Network

network = Network.from_pairs(
    scale,  # (P, F) per-pair detector noise, pair_bin_scale
    pair_overlap,  # (P, F) isotropic ORFs
    mask,  # (P, F) usable entries
    response_moment=pair_response_moment(
        jax.random.key(0), 32_768, frequencies, detectors
    ),
)
likelihood = DenseGaussianLikelihood(model, network, network.compress(observed))
log_likelihood, extras = likelihood.log_likelihood(params)
```

## Experiments

Paper notebooks, run configurations, plotting, and cluster workflows live in
the private [`experiments-astrogwb`](https://github.com/binado/experiments-astrogwb) repository. This distribution
contains the scientific library and the `astrogwb` CLI (`pip install
"astrogwb[cli]"`). CLI commands accept external run configuration via
`--config-dir` and detector layers via `--detector-config`.
