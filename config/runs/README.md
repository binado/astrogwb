# Committed runs

One file per run, and the filename is the mapping:

```text
config/runs/<experiment>/<run>.json  ->  outputs/chains/<experiment>/<run>.nc
```

A run file carries only what distinguishes it. Everything else is inherited
from the shared layers, one file per top-level block:

| Layer | Owns |
| --- | --- |
| `config/analysis.json` | observing time, frequency band, the target population and its redshift grid, and the two catalogs every run uses |
| `config/fiducials.json` | the fiducial value of every parameter |
| `config/networks.json` | each detector network, by name |
| `config/priors.json` | the prior on every parameter |
| `config/sampler.json` | the sampling RNG seed and the NUTS defaults |
| `config/waveform.json` | the waveform settings every catalog of a run inherits |
| `config/population.json` | the population a run's catalogs are drawn from, unless a role overrides it |

`_base.json` is the experiment override and is required in every experiment
directory — a conditional Snakemake input would complicate the DAG for no gain.
It is not a run, so it never becomes a chain.

`RunConfig` is `extra="forbid"`, and JSON has no comments, so what each run is
*for* is documented here rather than as a `description` field that would
duplicate this page and rot separately from it.

See [`docs/running-inference.md`](../../docs/running-inference.md) for the layer
model, the merge rules, and how to run one.

## The catalogs

Each run declares what its two catalogs draw in `[analysis.catalog]`: a
partial spec per role, over the run's own `[waveform]`, `[population]` and
`[fiducials]`. The file is `outputs/catalogs/<key>.h5`, named by the hash of the
resolved request, so runs that ask for the same draw share one file.
`just catalogs` lists every key, what it draws, and which runs use it. See
[`docs/catalog-generation.md`](../../docs/catalog-generation.md).

`config/analysis.json` sets the default for both roles: a seed-41,
32768-source draw from `config/population.json` with the shared IMRPhenom
waveform. That one catalog is the injection -- the "observed" data -- of every
run except `time-delay`, and the proposal of `cosmological-parameters`,
`modified-propagation` and `waveform-approximant/IMRPhenom`.

The overrides, and what each draw is for:

| Draw | Role | Used by |
| --- | --- | --- |
| seed 42, n = 8192 / 16384 / 32768 | proposal | `variable-catalog-size` |
| `bns_md_uniform_mixture`, ε = 0.1, seed 61, n = 16384 | proposal | `astrophysical-parameters`, `variable-proposal-guard/eps1e-1`, `time-delay` |
| `bns_md_uniform_mixture`, ε = 0.01 / 0.001, seeds 62 / 63, n = 16384 | proposal | `variable-proposal-guard` |
| `waveform.approximant = "TaylorF2"`, seed 41 | proposal | `waveform-approximant/TaylorF2` |
| `bns_md_time_delayed_cosmological`, seed 71 | injection | `time-delay` |

The seed-42 draws are exact prefixes of one another: `Predictive` allocates
per-draw keys with `jax.random.split`, which is prefix-stable, so the three
sizes are nested draws rather than unrelated ones.
`test_a_smaller_catalog_is_a_prefix_of_a_larger_one` in
`tests/core/test_populations.py` checks it rather than assuming it.

The TaylorF2 proposal is the seed-41 injection population with only the
approximant changed, so the comparison isolates the waveform.

The guarded proposals mix a uniform-in-redshift component into the
Madau-Dickinson law, so the importance weights do not degenerate when NUTS
moves the posterior away from the proposal. The mixture is one density: the
same `MixtureGeneral` that draws the redshifts evaluates their log density. A
guard mixture declares no merger rate -- it is a sampling density, not a
physical population -- and `snakemake validate` rejects one named as an
injection.

`time-delay` draws its catalogs at its own fiducials, `delay_slope = -1`
included, so its proposal is a separate copy of the ε = 0.1 guard: the samples
are identical to the shared one's, but the recorded fiducials differ.

Neither a network nor a detector list is ever a shared default. A run that
names no `analysis.network` must fail rather than silently inherit someone
else's, which is why `config/analysis.json` declares none.

## The experiments

### `cosmological-parameters` (8 runs)

H0 — or one parameter with H0 amplitude-marginalized out — from the
astrophysical background, across detector networks.

Every run samples a single scalar, so `_base.json` gives NUTS a diagonal mass
matrix and a shorter warmup. Six runs are one network each
(`ET-triangular`, `ET-2L-aligned`, `ET-2L-misaligned` and their CE-Hanford
variants), sampling `H0`. The other two marginalize H0 analytically:
`H0-Omega_m` samples `Omega_m`, and `H0-merger-rate` samples
`local_merger_rate` — the reference the fixed-R0 H0 posterior is compared
against.

### `modified-propagation` (8 runs)

Modified GW propagation (Ξ₀, n) across the same six networks, sampling
`xi_0` and `xi_n` together. Two runs narrow the problem: `Xi_0` pins n and
samples `xi_0` alone (1D), and `Xi_0-H0` samples `xi_0` with H0
amplitude-marginalized under a tight (1%) H0 prior, isolating the
propagation–cosmology degeneracy. That tight prior is the only `priors`
override in the tree.

### `astrophysical-parameters` (2 runs)

The Madau-Dickinson rate shape with H0 amplitude-marginalized out.
`madau-dickinson` samples `gamma`, `kappa` and `z_peak`; `redshift-peak`
samples `z_peak` alone.

Both need the guarded (ε = 0.1) proposal: sampling the rate shape moves the
posterior away from the proposal, and the uniform component keeps the
importance weights from degenerating.

### `variable-catalog-size` (3 runs)

How the H0 posterior tightens with proposal catalog size. The runs differ only
in their proposal's `num_samples`; every other setting is pinned in
`_base.json` so the comparison is clean. The three catalogs are nested prefixes
of one seed-42 stream, so the sizes are a clean series.

### `variable-proposal-guard` (3 runs)

How the uniform-mixing guard fraction ε affects the same H0–MD problem as
`astrophysical-parameters/madau-dickinson`: H0 is the marginalized amplitude
and NUTS samples the redshift-rate shape. The runs differ only in their
proposal's guard fraction, and each ε is drawn at its own mixture seed, so the
three draws are independent.

### `waveform-approximant` (2 runs)

Waveform systematics: the same H0 measurement against an IMRPhenom proposal and
a TaylorF2 one, both drawn from the seed-41 population.

### `time-delay` (1 run)

The delay-time slope α of `bns_md_time_delayed_cosmological`, whose mergers
follow a Madau-Dickinson *formation* rate after a delay
p(τ) ∝ τ^α of at least 20 Myr, capped only by the lookback time to the z = 20
formation cut-off. `delay-slope` samples `delay_slope` alone.

`_base.json` declares the target population, the one new fiducial
(α = −1) and its prior (Uniform[−3, 1]) itself rather than adding them to the
shared layers: every run's catalogs are drawn at its fiducials, so an edit to
`config/fiducials.json` would re-key every catalog. The injection is drawn from
the time-delayed population at the run's own α; the proposal is the ε = 0.1
guard.

The amplitude marginalized out is `local_merger_rate`, not H0: the delay is
fixed in Gyr while lookback time scales as 1/H0, so H0 reshapes the redshift
law instead of only rescaling it. The population declares this at
registration, and `snakemake validate` rejects `amplitude_parameter = "H0"` for
it.

## Adding one

Add a JSON file under an experiment directory; `discover_runs` picks it up by
globbing, and its stem becomes the chain path. Then add it to its experiment's
section above. Run `snakemake validate` first: it merges and catalog-checks all
runs, and builds every named population, before any GPU job is queued.
