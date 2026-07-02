# astrogwb

Small utilities for astrophysical stochastic gravitational-wave background
array contractions and inference.

`gwmock-signal` owns preset detector networks and waveform backends.
`gwmock-noise` owns bundled PSD presets and interpolation. `astrogwb` provides:

- SGWB spectral-density contractions and Omega_GW conversions.
- Frequency-dependent ORF and effective PSD utilities.
- Minimal waveform polarization-power catalog persistence.
- A thin NumPyro model for caller-prepared arrays, driven by a single `merger_rate_and_log_weights_fn(params, samples) -> (total_merger_rate, log_weights)` callback.
- Reference importance-weighting models under `astrogwb.importance.models.*` (import explicitly from the model module, not package `__init__` barrels).

## Detector analysis

For preset networks (HLVK, ET layouts, …), use gwmock's `Network` presets and
astrogwb's ORF/effective-PSD utilities. astrogwb also ships supplemental
geometry (`geometry.toml`, `load_detector`) and local noise curves for
detectors without a suitable gwmock preset or LAL code (for example Cosmic
Explorer sites). On top of gwmock's geometry and PSD loading, astrogwb adds
the frequency-dependent overlap reduction function and the out-of-band PSD
policy:

```python
import numpy as np
from gwmock_signal.network import Network

from astrogwb.detector import (
    effective_psd,
    load_sensitivities_for_network,
    overlap_reduction_function,
)

network = Network.from_name("HLVK")  # see Network.list_names() for presets
sensitivities = load_sensitivities_for_network(network)
f = np.geomspace(20, 2048, 128)
gamma = overlap_reduction_function(f, "H1", "L1")  # frequency-dependent ORF
psd = effective_psd(f, network.detector_names, sensitivities)

network_et = Network.from_name("ET-Triangle-Sardinia")  # CustomDetector presets
sensitivities_et = load_sensitivities_for_network(network_et)
```

### Building a detector network

Use gwmock preset names (see `Network.list_names()`) or assemble a
`Network` yourself. For an Einstein Telescope network, a preset is enough:

```python
from gwmock_signal.network import Network

from astrogwb.detector import load_sensitivities_for_network

network_et = Network.from_name("ET-triangle")  # str codes E1, E2, E3
# or the L-shaped Sardinia layout (gwmock CustomDetector presets):
network_et = Network.from_name("ET-Triangle-Sardinia")
sensitivities_et = load_sensitivities_for_network(network_et)
```

For a combined ET + Cosmic Explorer network there is no gwmock preset, so
build the `Network` from detectors. Use `load_detector`, a factory that
returns a gwmock `CustomDetector` from astrogwb's `geometry.toml`, for the
CE detectors:

```python
from gwmock_signal.network import Network

from astrogwb.detector import effective_psd, load_detector, load_sensitivities_for_network

# C2 (CE at Livingston) has no LAL code, and LAL's "C1" is the Caltech 40m
# prototype (CIT_40) — not Cosmic Explorer. load_detector sources astrogwb's
# CE geometry instead, so the network is built from the intended sites.
detectors = [load_detector(name) for name in ("E1", "E2", "E3", "C1", "C2")]
network = Network.from_detectors(detectors, name="ET+CE")
sensitivities = load_sensitivities_for_network(network)

effective_psd(f, network.detector_names, sensitivities)
overlap_reduction_function(f, "E1", "C1")  # ORF between any two members
```

Sensitivity curves are matched to detectors by their public name
(`E1`, `C1`, `ET1_SARD`, …), so every member must have an entry in
`sensitivity.toml`.

## Population generation

The importance-sampling waveform catalog starts from a population of intrinsic
parameters. We delegate the sampling to [`gwmock-pop`][gwmock-pop]: the BNS
population is defined declaratively in [`examples/bns_population.yaml`](examples/bns_population.yaml)
and drawn with the `gwmock-pop simulate` CLI.

```bash
uv run gwmock-pop simulate \
  --config examples/bns_population.yaml \
  --n 1000 \
  --output out/bns_population.h5 \
  --seed 42
```

`--output` accepts `.csv`, `.h5`, or `.hdf5`; use `.h5` for the structured
output the downstream catalog step consumes. The result is a table of `n`
intrinsic samples — one column per parameter, using gwmock-pop *canonical*
names (`source_frame_mass_1/2`, `luminosity_distance`, `spin_1z/2z`,
`lambda_1/2`, `inclination`, `coa_phase`, `coa_time`).

The committed config encodes a BNS population with a Madau–Dickinson redshift
distribution (converted to luminosity distance), uniform source-frame component
masses ordered so `mass_1 >= mass_2`, aligned spins (in-plane components zero),
uniform tidal deformabilities, and inclination/coalescence phase/time fixed at
zero. Edit the `arguments` blocks to retune ranges. The aligned-spin + tidal
parameters suit a non-precessing NRTidal approximant downstream.

Masses are emitted in the **source frame**. The waveform backend
(`generate_catalog_polarization_power`, `src/astrogwb/waveform/__init__.py`)
consumes *detector-frame* masses, so the catalog-build step applies the
redshift conversion `detector_frame_mass = source_frame_mass * (1 + z)` when it
loads this population — gwmock-pop cannot express the `(1 + z)` factor in the
YAML graph itself.

[gwmock-pop]: https://leuven-gravity-institute.github.io/gwmock-pop/

## MCMC inference notebook

`notebooks/mcmc.py` (a [py:percent](https://jupytext.readthedocs.io/en/latest/formats-percent.html)
script, runnable as a notebook) is the NumPyro port of
`ASGWB.jl/notebooks/mcmc.jl`. It performs Bayesian inference on the
cosmological and astrophysical parameters that drive the CBC stochastic
background via **importance-weighted NUTS**: a fixed proposal catalog supplies
the per-source polarization powers, and NUTS reweights them analytically — no
waveform is regenerated during sampling.

**Prerequisites**

- A `.npz` polarization-power catalog written by
  `astrogwb.waveform.save_polarization_power_catalog`. Set `CATALOG_PATH` at the
  top of the notebook to point at it. Each source must carry `redshift` and
  `luminosity_distance`, and the polarization-power columns must already include
  the `1/d_{L,\mathrm{fid}}^2` scaling so the importance-weight math is exact.
- `gwmock-pop` provides the JAX-traceable Madau–Dickinson rate and flat-ΛCDM
  cosmology. The reference callback factory lives in
  `astrogwb.importance.models.bns_madau_dickinson_modified_propagation`;
  shared cosmology helpers are in `astrogwb.cosmology`.

**Conventions**

`observation_time` is in **years**. It cancels in the relative spectrum but sets
the likelihood noise scale (see `gaussian_bin_scale` in `astrogwb.gwb`).

**Running**

```bash
uv sync --group dev                                   # bundles Jupyter, arviz, corner, matplotlib
uv run jupyter nbconvert --to notebook --execute notebooks/mcmc.py
# or open in JupyterLab:
uv run jupyter lab notebooks/mcmc.py
```

End-to-end execution is deferred until a real catalog exists: the committed
`CATALOG_PATH = "catalog.npz"` is a placeholder, so the data-dependent cells are
correct by construction but only run once a catalog is produced.

**Outputs**

Each run writes an ArviZ `InferenceData` to
`chains/chains-<params>-det=<det>-seed<n>-<ts>.nc` alongside a sibling `.json`
run config (catalog path, detectors, seed, fiducials, sampler settings).
Diagnostics surface the model's `importance_relative_ess` (the key proposal
health check — should stay close to 1) and `total_merger_rate`.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Development

Install dependencies (the dev group also bundles Jupyter, arviz, corner, and
matplotlib so the MCMC notebook in `notebooks/` runs out of the box):

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
uv run pytest -m "not integration"   # fast unit tests only
```

Regression fixtures under `tests/fixtures/` are committed; integration tests
cross-check gwfast and skip if optional fixtures are missing.

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Type check:

```bash
uvx ty check
```
