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
- `astrogwb.gwb` provides spectral-density, Omega-GW conversion, and SNR
  calculations. `astrogwb.gwb.analytic` evaluates the inspiral-only
  background in closed form, truncated on the same dimensionless `alpha` as
  `astrogwb.waveform`.
- `astrogwb.populations` declares source populations as NumPyro models,
  addressed by registered name. One declaration serves both generation and
  density evaluation, so a catalog and the weights that reweight it can never
  disagree about the law behind it.
- `astrogwb.importance` reweights a fixed catalog to a target population and
  contracts it into a spectrum.
- `astrogwb.sampling` exposes the caller-prepared NumPyro model.
- `astrogwb.catalog` provides an array-native catalog that records the
  population that drew it, and reads and writes it as HDF5 behind the `io`
  extra.
- `astrogwb.waveform` owns polarization-power generators, reduces raw
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

A population is drawn, reduced to polarization power, and stored together with
the declaration that produced it:

```python
import jax

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.constants import ISCO_ALPHA
from astrogwb.populations import DEFAULT_DENSITY_SITES, build_population
from astrogwb.utils.sampling import sample_sources
from astrogwb.waveform import AnalyticInspiralGenerator

model_kwargs = {"minimum_redshift": 0.0, "maximum_redshift": 20.0, "n_grid": 4096}
params = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}
source_model = build_population("bns_md_cosmological", **model_kwargs).source_model
source_parameters = sample_sources(
    source_model, jax.random.PRNGKey(42), params, num_samples=1024
)

catalog = PolarizationPowerCatalog.from_generator(
    source_parameters,
    generator=AnalyticInspiralGenerator(
        alpha=ISCO_ALPHA,
        approximant="AnalyticInspiral",
        minimum_frequency=2.0,
        maximum_frequency=2048.0,
        reference_frequency=2.0,
        sampling_frequency=4096.0,
        frequency_resolution=1.0,
    ),
    model_name="bns_md_cosmological",
    model_kwargs=model_kwargs,
    fiducials=params,
    density_sites=DEFAULT_DENSITY_SITES,
    seed=42,
)
catalog.save("catalog.h5")
```

Reweighting it to a target population needs nothing else: the file says what
drew it, so the proposal density is recovered rather than restated. A
population is one registered name that yields both callables, so a target's
source model and merger rate cannot be paired with one another by mistake.

```python
from astrogwb.importance.spectral import build_importance_spectrum

target = build_population("bns_md_modified_propagation", **model_kwargs)
spectrum_fn = build_importance_spectrum(
    PolarizationPowerCatalog.load("catalog.h5"),
    source_model=target.source_model,
    merger_rate_fn=target.merger_rate_fn,
)[0]
spectrum, extras = spectrum_fn({**params, "H0": 70.0, "xi_0": 1.2, "xi_n": 1.91})
```

## The manuscript application

The reproducibility application -- the MCMC runner, catalog generation, campaign
configuration, cluster profiles, and the Snakemake workflow that drives them --
ships inside this package as `astrogwb.paper`, behind an extra:

```bash
pip install "astrogwb[paper]"
```

Installing `astrogwb` alone leaves it inert: the subpackage is present but its
dependencies are not, which is how the one-way dependency (`astrogwb.paper`
may import `astrogwb`, never the reverse) is kept honest.

Its committed assets -- `config/`, `scripts/`, `notebooks/`, `profiles/` and
the `Snakefile` -- live at the root of the source repository and are not part
of the wheel; the application is meant to be run from a checkout, with the
repository root as the working directory. See [`docs/`](docs/) for catalog
generation, running inference, the workflow, and the paper figures.
