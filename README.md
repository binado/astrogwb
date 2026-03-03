# asgwb

Constraining cosmology with the stochastic gravitational-wave background (SGWB)
from astrophysical sources.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

## Local installation

Install project and development dependencies:

```bash
uv sync --group dev
```


Run project scripts inside the managed environment:

```bash
uv run python scripts/generate_injection_waveforms.py --help
```

## Development workflow

Run tests:

```bash
uv run pytest
uv run pytest -m "not integration"
```

Integration tests for the overlap reduction function (ORF) compare against a
pre-generated reference fixture stored in `tests/fixtures/`.  The fixture is
not committed to git; generate it once before running integration tests locally:

```bash
uv run --script scripts/generate_orf_fixtures.py
uv run pytest -m integration
```

In CI the fixture is generated automatically and cached across runs (cache key
is a hash of `overlap.py` and `generate_orf_fixtures.py`).

Lint and format:

```bash
uv run ruff check . --fix
uv run ruff format .
pre-commit run --all-files
```

## Project layout

- `src/asgwb/`: package source code.
- `src/asgwb/detector/`: detector models, TOML metadata, and noise curves.
- `scripts/`: waveform generation and SLURM submission scripts.
- `config/`: default runtime config (for example, `generate_injection_waveforms.toml`).
- `tests/`: unit and integration tests.
- `data/`: input catalogs and reference data.

## scripts/ directory

Most operational workflows are in `scripts/`:

- `generate_injection_waveforms.py`: generates waveform batches from an injection catalog.
- `merge_injection_waveforms.py`: merges batch outputs into a consolidated HDF5 file.
- `convert_parameter_names.py`: normalizes parameter column names for downstream tools.
- `submit.sh`: submits array jobs on SLURM for parallel waveform generation.

Each script supports `--help`:

```bash
uv run python scripts/generate_injection_waveforms.py --help
```

## Example script usage

Generate waveforms from an injection catalog:

```bash
uv run python scripts/generate_injection_waveforms.py \
  --injection-file data/injections_COBA_BNS.csv \
  --output-file out.hdf5 \
  --batch 100 \
  --nworkers 4
```

Submit a SLURM array job:

```bash
bash scripts/submit.sh -i data/injections_COBA_BNS.csv -n 50 -o out
```

## License

This project is licensed under the MIT License. See `LICENSE`.
