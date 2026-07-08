# astrogwb

This is a package for Bayesian inference of cosmological and astrophysical parameters
with the stochastic gravitational-wave background (SGWB) from stellar-mass compact binary coalescences (CBCs) detectable by ground-based interferometer networks such as LIGO, Virgo, KAGRA, Einstein Telescope and Cosmic Explorer.

## Installation

The easiest way to install the package is using the `uv` package manager:

```bash
git clone git@github.com:binado/astrogwb.git
uv sync --all-groups --all-extras
```

## Generating a population of CBCs

Our inference framework uses an importance sampling scheme to calculate the spectral density of the astrophysical SGWB with a fixed population of CBCs. To generate the population, we suggest using the excellent [`gwmock-pop`](https://leuven-gravity-institute.github.io/gwmock-pop/) package.

An example BNS population is defined declaratively in [`examples/bns_population.yaml`](examples/bns_population.yaml)
and drawn with the `gwmock-pop simulate` CLI. Here's how to
simulate a BNS population of 1000 sources:

```bash
uv run gwmock-pop simulate \
  --config examples/bns_population.yaml \
  --n 1000 \
  --output out/bns_population.h5 \
  --seed 42
```

`--output` accepts `.csv`, `.h5`, or `.hdf5`; use `.h5` for the structured
output the downstream catalog step consumes. The result is a table of `n`
intrinsic samples — one column per parameter, using gwmock-pop canonical
names (`source_frame_mass_1/2`, `luminosity_distance`, `spin_1z/2z`,
`lambda_1/2`, `inclination`, `coa_phase`, `coa_time`).

The committed config encodes a BNS population with a Madau–Dickinson-like redshift
distribution (converted to luminosity distance), uniform source-frame component
masses ordered so `mass_1 >= mass_2`, aligned spins (in-plane components zero),
uniform tidal deformabilities, and inclination/coalescence phase/time fixed at
zero. Edit the `arguments` blocks to retune ranges. The aligned-spin + tidal
parameters suit a non-precessing NRTidal approximant downstream.
The mass priors are defined in the source frame.

## Generating waveforms for the population catalog

We provide a [helper script](./scripts/generate_waveform_catalog.py) which wraps the [`gwmock-signal`](https://github.com/Leuven-Gravity-Institute/gwmock-signal) package for generating the frequency-domain polarizations for a given population of CBCs which enter the spectral density calculation. The output is a `waveform_catalog` HDF5 file of complex polarizations (see the waveform-catalog repo's `SPEC.md`), which the inference consumers reduce to polarization power at load time.

Here is an example:

```bash
uv run python scripts/generate_waveform_catalog.py \
--population out/bns_population.csv \
--output out/bns_waveform_catalog.h5 \
--approximant IMRPhenomXAS_NRTidalv3 \
--sampling-frequency 8192 \
--minimum-frequency 2 \
--maximum-frequency 4096 \
--reference-frequency 20 \
--frequency-resolution 1 \
--chunk-size 2048
```

## Running inference

There are two working examples provided in the repo:

- The [`run_mcmc.py` script](./scripts/run_mcmc.py) accepts a TOML or JSON configuration file and is suitable for running MCMC on a cluster.
- The [`mcmc.py` notebook](./notebooks/mcmc.py) has the same functionality as the script but can be run interactively in a Jupyter notebook. You can use `uvx jupytext --to ipynb notebooks/mcmc.py` to convert it to a Jupyter notebook in your local machine.

Both examples assume a population of binary neutron star (BNS) mergers following the example described above. See the notebook for more details on how the inference is set up.

### Outputs

Each run writes an ArviZ `InferenceData` to
`chains/chains-<params>-det=<det>-seed<n>-<ts>.nc` alongside a sibling `.json`
run config (catalog path, detectors, seed, fiducials, sampler settings).
Diagnostics surface the model's `importance_relative_ess` (the key proposal
health check — should stay close to 1) and `total_merger_rate`.

See the [plotting notebook](./notebooks/mcmc_plotting.py) for examples of how to visualize the results.

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
