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

`_base.json` is the experiment override and is required in every experiment
directory — a conditional Snakemake input would complicate the DAG for no gain.
It is not a run, so it never becomes a chain.

`RunConfig` is `extra="forbid"`, and JSON has no comments, so what each run is
*for* is documented here rather than as a `description` field that would
duplicate this page and rot separately from it.

See [`docs/running-inference.md`](../../docs/running-inference.md) for the layer
model, the merge rules, and how to run one.

## The catalogs

`config/analysis.json` names the two catalogs every run uses. Each name
resolves to `config/catalogs/<name>.json` and to the
`outputs/catalogs/<name>.h5` it produces; how a catalog was drawn — its
population, seeds and mixing fractions — lives in those files, not here, and
the redshift density that follows is read back off the generated file at run
time.

`config/analysis.json` sets the default injection, the "observed" data. Only
`time-delay` overrides it, with a catalog drawn from its own time-delayed
population. `variable-catalog-size`, `variable-proposal-guard`,
`astrophysical-parameters`, `waveform-approximant` and `time-delay` override the
proposal.

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
in which proposal catalog they name; every other setting is pinned in
`_base.json` so the comparison is clean. The three catalogs are nested prefixes
of one seed-42 stream, so the sizes are a clean series.

### `variable-proposal-guard` (3 runs)

How the uniform-mixing guard fraction ε affects the same H0–MD problem as
`astrophysical-parameters/madau-dickinson`: H0 is the marginalized amplitude
and NUTS samples the redshift-rate shape. The runs differ only in which guarded
proposal catalog they name, and each ε has its own catalog drawn at its own
mixture seed, so the three draws are independent.

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
shared layers: `config/fiducials.json` is a catalog layer too, so an edit there
would invalidate every catalog. The injection is
`md-delayed-imrphenom-s71-n32768`, drawn at the same α; the proposal is the
ε = 0.1 guard catalog.

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
