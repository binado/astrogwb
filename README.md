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
- `astrogwb.importance.model.CatalogModel` predicts the spectral density and
  the intensity second moment from an explicit reference catalog, intrinsic
  density and merger rate.
- `astrogwb.inference` provides the dense Gaussian detector-network likelihood
  of any model implementing its `SpectralDensityModel` protocol.
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

from astrogwb.importance.model import CatalogModel
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
)
prediction = model(
    {**params, "mass_width": 1.2, "H0": 70.0, "xi_0": 1.2, "xi_n": 1.91},
    shot_noise=True,
)
```

When the target intrinsic density matches the proposal throughout the analysis,
pass `target_matches_proposal=True` in place of `log_prob` to cache the
unit-weight moments.

`prediction.spectral_density` is the background's mean, $S_h = 2\mathcal R\langle I\rangle$,
and `prediction.intensity_second_moment` is $4\mathcal R\langle I I'\rangle$, the
raw second moment that sets the shot noise of a Poisson catalog. A detector
network's likelihood of its per-pair cross powers is the dense Gaussian with
covariance `Sigma + intensity_second_moment * connected_overlap / T`
(elementwise), with `Sigma` the detector noise, `connected_overlap` the
network's sky and polarization average
(`astrogwb.detector.connected_overlap`) and `T` the observation time. The
likelihood builds the network from the detectors; the data are keyed by the
sorted detector-name pair:

```python
from astrogwb.detector import load_detector
from astrogwb.inference import DenseGaussianLikelihood

likelihood = DenseGaussianLikelihood(
    model,
    [load_detector(name) for name in ("E1", "E2", "E3")],  # or a network name
    frequencies=frequencies,
    observation_time=1.0,  # years
    observed={("E1", "E2"): c_12, ("E1", "E3"): c_13, ("E2", "E3"): c_23},
)
log_likelihood, extras = likelihood.log_likelihood(params)
```

Any pytree implementing `SpectralDensityModel` works in place of a catalog
model, for example a `jax.tree_util.Partial` of a module-level function
returning a `SpectralDensityPrediction`.

## Experiments

Paper notebooks, run configurations, plotting, and cluster workflows live in
the private [`experiments-astrogwb`](https://github.com/binado/experiments-astrogwb) repository. This distribution
contains the scientific library and the `astrogwb` CLI (`pip install
"astrogwb[cli]"`). `astrogwb simulate` takes waveform settings via
`--waveform-config` and detector layers via `--detector-config`.
