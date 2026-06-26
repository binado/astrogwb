# astrogwb

Small utilities for astrophysical stochastic gravitational-wave background
array contractions and inference.

`gwmock-signal` owns preset detector networks and waveform backends.
`gwmock-noise` owns bundled PSD presets and interpolation. `astrogwb` provides:

- SGWB spectral-density contractions and Omega_GW conversions.
- Frequency-dependent ORF and effective PSD utilities.
- Minimal waveform polarization-power catalog persistence.
- A thin NumPyro model for caller-prepared arrays.

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

### Migration from the pre-refactor API

| Removed (old)                                   | Replacement (new)                                            |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `Detector` dataclass                            | `DetectorSpec` (`str` site code or gwmock `CustomDetector`) |
| `Detector.from_file("H1")`                      | `"H1"` passed directly, resolved via `geometry.toml`         |
| `PowerSpectralDensity` / `.evaluate(f)`         | `Sensitivity` + `evaluate_psd(reference, f)`                 |
| `PowerSpectralDensity.from_noise_curve_dir(...)`| `evaluate_psd(reference, f)` (reference resolves preset/file/URL) |
| `effective_psd(f, detectors)`                   | `effective_psd(f, detectors, load_sensitivities_for_network(network))` |
| `detectors.toml`                                | `geometry.toml` (geometry) + `sensitivity.toml` (PSD/band/duty) |

Out-of-band behavior: `evaluate_psd(..., out_of_band="inf")` (the default)
maps frequencies outside the curve grid to `inf`, matching the old
`PowerSpectralDensity` semantics so out-of-band bins drop out of an
inverse-variance contraction; `out_of_band="zero"` returns gwmock-noise's
raw clipped-to-zero interpolation.

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
names (`detector_frame_mass_1/2`, `luminosity_distance`, `spin_1z/2z`,
`lambda_1/2`, `inclination`, `coa_phase`, `coa_time`) — which feeds directly
into `generate_catalog_polarization_power` (`src/astrogwb/waveform/__init__.py`)
as its `samples` mapping.

The committed config encodes a BNS population with a Madau–Dickinson redshift
distribution (converted to luminosity distance), uniform detector-frame masses,
aligned spins (in-plane components zero), uniform tidal deformabilities, and
inclination/coalescence phase/time fixed at zero. Edit the `arguments` blocks to
retune ranges. The aligned-spin + tidal parameters suit a non-precessing NRTidal
approximant downstream.

[gwmock-pop]: https://leuven-gravity-institute.github.io/gwmock-pop/

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Development

Install dependencies:

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
