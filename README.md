# astrogwb

Small utilities for astrophysical stochastic gravitational-wave background
array contractions and inference.

`gwmock-signal` owns detector and waveform objects. `astrogwb` provides:

- SGWB spectral-density contractions and Omega_GW conversions.
- Frequency-dependent ORF and effective PSD utilities.
- Minimal waveform polarization-power catalog persistence.
- A thin NumPyro model for caller-prepared arrays.

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
