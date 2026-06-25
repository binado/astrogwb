# astrogwb

Small utilities for astrophysical stochastic gravitational-wave background
array contractions and inference.

`gwmock-signal` owns detector and waveform objects. `astrogwb` provides:

- SGWB spectral-density contractions and Omega_GW conversions.
- Frequency-dependent ORF and effective PSD utilities.
- Minimal waveform polarization-power catalog persistence.
- A thin NumPyro model for caller-prepared arrays.

## Detector analysis

gwmock is the single source of truth for detector geometry (`gwmock-signal`)
and noise curves (`gwmock-noise`). astrogwb adds the SGWB-specific
frequency-dependent overlap reduction function and the analysis policy on
top. `analysis_setup` is the entry point:

```python
import numpy as np
from astrogwb.detector import analysis_setup

ctx = analysis_setup("HLVK")           # see Network.list_names() for presets
f = np.geomspace(20, 2048, 128)
gamma = ctx.overlap(f, "H1", "L1")     # frequency-dependent ORF
psd = ctx.effective_psd(f)             # network effective PSD

ctx_et = analysis_setup("ET-Triangle-Sardinia")  # CustomDetector presets
```

### Migration from the pre-refactor API

| Removed (old)                                   | Replacement (new)                                            |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `Detector` dataclass                            | `analysis_setup(...)` / `AnalysisContext`, or a `DetectorSpec` (`str` site code or gwmock `CustomDetector`) |
| `Detector.from_file("H1")`                      | `"H1"` passed directly, resolved via `geometry.toml`         |
| `PowerSpectralDensity` / `.evaluate(f)`         | `Sensitivity` + `evaluate_psd(reference, f)`                 |
| `PowerSpectralDensity.from_noise_curve_dir(...)`| `evaluate_psd(reference, f)` (reference resolves preset/file/URL) |
| `effective_psd(f, detectors)`                   | `effective_psd(f, detectors, sensitivities)` or `ctx.effective_psd(f)` |
| `detectors.toml`                                | `geometry.toml` (geometry) + `sensitivity.toml` (PSD/band/duty) |

Out-of-band behavior: `evaluate_psd(..., out_of_band="inf")` (the default)
maps frequencies outside the curve grid to `inf`, matching the old
`PowerSpectralDensity` semantics so out-of-band bins drop out of an
inverse-variance contraction; `out_of_band="zero"` returns gwmock-noise's
raw clipped-to-zero interpolation.

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
```

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Type check:

```bash
uvx ty check
```
