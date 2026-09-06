# astrogwb

`astrogwb` provides reusable scientific modules for modelling the stochastic
gravitational-wave background from compact-binary populations and performing
Bayesian inference with detector networks.

## Installation

```bash
pip install astrogwb
```

Optional accelerator builds are available as `astrogwb[cuda]` and
`astrogwb[tpu]`. Install the population-simulation adapter separately when it
is needed:

```bash
pip install astrogwb[simulation]
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
  `astrogwb.waveform.analytical`.
- `astrogwb.importance` defines the reusable importance-weighting protocol and
  compact-binary population model.
- `astrogwb.sampling` exposes the caller-prepared NumPyro model.
- `astrogwb.catalog` provides array-native catalog metadata, validation,
  population simulation, and polarization-power generation.
  `astrogwb.catalog.generator` defines the generator protocol and the
  closed-form inspiral adapter; the optional `gwmock-pop` adapter is imported
  only when `simulate_population` is called.
- `astrogwb.waveform` reduces raw plus/cross polarizations to power, applies
  GW-distance corrections to plain arrays, and provides a closed-form
  quadrupolar inspiral model. Persistence and labelled-array policy stay with
  applications.
  `astrogwb.waveform.analytical` gives the same power in closed form for a
  quadrupolar, inspiral-only binary, truncated at `f = alpha / ((1 + z) M)`
  for a caller-chosen dimensionless `alpha`.

For example:

```python
from astrogwb.detector import load_detector, load_sensitivity

hanford = load_detector("H1")
sensitivity = load_sensitivity("H1")
```

A prepared population can be reduced through the common generation interface:

```python
from astrogwb.constants import ISCO_ALPHA
from astrogwb.catalog import (
    AnalyticInspiralGenerator,
    Catalog,
    FrequencyDomainWaveformMetadata,
    PopulationMetadata,
)

waveform_metadata = FrequencyDomainWaveformMetadata.from_bounds(
    approximant="AnalyticInspiral",
    minimum_frequency=2.0,
    maximum_frequency=2048.0,
    reference_frequency=2.0,
    sampling_frequency=4096.0,
    df=1.0,
)
population_metadata = PopulationMetadata(
    name="my-caller-owned-graph",
    seed=42,
    num_samples=len(source_parameters["redshift"]),
)

catalog = Catalog.from_generator(
    source_parameters,
    generator=AnalyticInspiralGenerator(alpha=ISCO_ALPHA),
    waveform_metadata=waveform_metadata,
    population_metadata=population_metadata,
)
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
