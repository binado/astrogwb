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
- `astrogwb.populations` and `astrogwb.importance` provide BNS reference
  catalog generation, including scrambled Sobol draws.
- `astrogwb.inference.CatalogModel` predicts spectra and shot-noise moments
  using an explicit reference catalog, intrinsic density and merger rate.
  The inference module also provides detector-network likelihoods.
- `astrogwb.background.simulation.simulate_background` draws segmented detector-pair spectra
  from a supplied physical event pool and observer-frame rate, and
  `astrogwb simulate` runs it on a gwmock-pop population catalogue. See the
  [simulation API, command and normalization](docs/simulator-pipeline.md).
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

A reference catalog places caller-drawn intrinsic sources at one reference
redshift, face-on, and generates their intensities. The caller supplies the
source model, the proposal log density of the draws and the reference redshift:

```python
import jax
import numpy as np
from numpyro import handlers

from astrogwb.importance.spectrum import REFERENCE_INCLINATION, reference_catalog
from astrogwb.populations import coba
from astrogwb.utils.numpyro import sample_model
from astrogwb.waveform import WaveformMetadata, waveform_generator

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
mass_sites = ("source_frame_mass_1", "source_frame_mass_2")
model = handlers.condition(
    coba.source_model(params, mass_model="uniform"),
    data={"inclination": REFERENCE_INCLINATION},
)
samples = sample_model(model, jax.random.key(42), 1024)
catalog = reference_catalog(
    waveform_generator(waveform),
    samples,
    intrinsic_sites=mass_sites,
    # Joint density of the ordered uniform mass pair.
    proposal_log_prob=np.full(1024, np.log(2.0) - 2.0 * np.log(params["mass_width"])),
    reference_redshift=0.01,
    parameters=params,
)
```

A spectrum prediction uses the reference catalog, with every source placed at
a common reference redshift and face-on, so that
a source at any redshift is a rescaling of its reference power. Intrinsic
parameters are reweighted to a target population; redshift is integrated by
quadrature. The proposal and target log densities must use the same variables
and measure. Here the spin and tidal distributions are shared, so their
densities cancel and only the ordered mass pair enters the ratio.

```python
import jax.numpy as jnp
import numpy as np

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.inference import CatalogModel
from astrogwb.rates import madau_dickinson_rate


def ordered_mass_log_prob(intrinsic, target):
    primary, secondary = (intrinsic[name] for name in mass_sites)
    low, width = target["minimum_mass"], target["mass_width"]
    support = (low <= secondary) & (secondary <= primary) & (primary <= low + width)
    return jnp.where(support, jnp.log(2.0) - 2.0 * jnp.log(width), -jnp.inf)


def merger_rate(redshift, target):
    return madau_dickinson_rate(
        redshift,
        target["gamma"],
        target["kappa"],
        target["z_peak"],
        target["local_merger_rate"],
    )


model = CatalogModel.from_catalog(
    catalog,
    merger_rate=merger_rate,
    log_prob=ordered_mass_log_prob,
    frequencies=np.arange(10.0, 1000.0, 1.0),
    num_redshift_nodes=32,
    minimum_redshift=0.01,
    maximum_redshift=20.0,
    observation_seconds=SECONDS_PER_YEAR,
)
spectrum, power_moment, extras = model.predict(
    {**params, "mass_width": 1.2, "H0": 70.0, "xi_0": 1.2, "xi_n": 1.91}
)
```

When the target intrinsic density matches the proposal throughout the analysis,
pass `target_matches_proposal=True` in place of `log_prob` to cache the
unit-weight mean and power moments.

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
"astrogwb[cli]"`). `astrogwb simulate` takes waveform settings via
`--waveform-config` and detector layers via `--detector-config`.
