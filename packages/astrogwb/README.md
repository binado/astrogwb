# astrogwb

`astrogwb` provides reusable scientific modules for modelling the stochastic
gravitational-wave background from compact-binary populations and performing
Bayesian inference with detector networks.

## Installation

```bash
pip install astrogwb
```

Optional accelerator builds are available as `astrogwb[cuda]` and
`astrogwb[tpu]`.

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
- `astrogwb.waveform` owns the `waveform_catalog` HDF5 format (IO via
  `astrogwb.waveform.catalog`), which stores per-sample polarization power,
  and reduces raw plus/cross polarizations to that power at generation time.
  `astrogwb.waveform.analytical` gives the same power in closed form for a
  quadrupolar, inspiral-only binary, truncated at `f = alpha / ((1 + z) M)`
  for a caller-chosen dimensionless `alpha`.

For example:

```python
from astrogwb.detector import load_detector, load_sensitivity

hanford = load_detector("H1")
sensitivity = load_sensitivity("H1")
```

## Examples

`examples/` holds runnable end-to-end scripts that depend only on `astrogwb`.
`examples/h0_mcmc.py` takes a waveform catalog, builds the observed spectral
density from it, and infers `H0` with a NumPyro NUTS chain:

```bash
python examples/h0_mcmc.py CATALOG.h5 -o chains.nc
```

`examples/amplitude_marginalized_model.py` is the same run with `H0` marginalized out of the
likelihood analytically and reconstructed afterwards, giving a joint
`(H0, Omega_m)` posterior -- the setup every production analysis uses.

See `examples/README.md` for the options and for how the catalog serves as its
own importance-sampling proposal.

The manuscript workflows, configurations, and notebooks live in the separate
`astrogwb-paper` workspace project in the source repository.
