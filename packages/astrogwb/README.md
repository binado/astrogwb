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

- `astrogwb.detector` loads bundled detector geometry and sensitivity data and
  evaluates overlap-reduction functions and effective PSDs.
- `astrogwb.gwb` provides spectral-density, Omega-GW conversion, and SNR
  calculations.
- `astrogwb.importance` defines the reusable importance-weighting protocol and
  compact-binary population model.
- `astrogwb.sampling` exposes the caller-prepared NumPyro model.
- `astrogwb.waveform` owns the `waveform_catalog` HDF5 format (IO via
  `astrogwb.waveform.catalog`) and reduces catalogs to polarization power.

For example:

```python
from astrogwb.detector import load_detector, load_sensitivity

hanford = load_detector("H1")
sensitivity = load_sensitivity("H1")
```

The manuscript workflows, configurations, and notebooks live in the separate
`astrogwb-paper` workspace project in the source repository.
