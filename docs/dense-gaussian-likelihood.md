# Design: the dense Gaussian likelihood

Status: proposal. Supersedes the likelihood parts of
[`stacked-pair-likelihood.md`](stacked-pair-likelihood.md), whose per-pair data
layout (D1, D2) and simulator output (D5) are already on `main` and are kept.

## Purpose

Make `astrogwb.inference` implement exactly the likelihood of the paper's
Sec. 2.7 (`eq:shot-noise-likelihood`), and remove the covariance models that no
longer correspond to it. Reshape the API around how the paper application
actually evaluates it: on parameter grids, for a handful of networks and many
data realizations, never with gradients.

## The model

The data are the segment-averaged cross powers of every distinct detector pair
and frequency bin, $d_{ab,k}$, stacked into one vector of length
$D = N_{\rm pairs} \mathcal N_f$. The likelihood is one Gaussian with a dense,
parameter-dependent covariance,

$$
\ln \mathcal L(\mathbf d|\Lambda) = -\tfrac12 (\mathbf d - \boldsymbol\mu)^T C^{-1}
    (\mathbf d - \boldsymbol\mu) - \tfrac12 \ln\det C - \tfrac{D}{2}\ln 2\pi,
\qquad C(\Lambda) = \Sigma + \mathcal C^{\rm Pois}(\Lambda).
$$

$\Sigma = {\rm diag}(\sigma^2_{ab,k})$ is the weak-signal detector-noise
covariance. The connected term factorizes (`eq:catalog-covariance-factorized`)
into a moment of the source power and a moment of the network response,

$$
\mathcal C^{\rm Pois}_{ab,k;\,cd,l} = M_{kl}(\Lambda)\, R_{ab,k;\,cd,l},
\qquad
M_{kl} = \frac{\mathcal R}{T} \langle \bar P_k \bar P_l \rangle_{\theta|\Lambda},
\qquad
R_{ab,k;\,cd,l} = \langle g \rangle^2 \gamma_{ab,k}\gamma_{cd,l}
    \langle r_{ab,k}\, r_{cd,l} \rangle .
$$

$M$ is `(F, F)` and depends on $\Lambda$ but not on the network. $R$ is
`(D, D)` and depends on the network but not on $\Lambda$. Every design decision
below follows from that split.

Units are those of `stacked-pair-likelihood.md` D1: unnormalized cross powers
with mean $\mu_{ab,k} = \gamma_{ab}(f_k) S_h(f_k|\Lambda)$, which differ from
the paper's normalized $\bar C_{ab}$ by a fixed per-entry rescaling and give
the same posterior. The $\langle g \rangle^2 \gamma\gamma$ in $R$ is that
rescaling applied to the paper's normalized response moment.

With $\mathcal C^{\rm Pois} = 0$ this is the detector-noise-only likelihood of
Sec. 2.6.

## How the likelihood is used

- Every posterior notebook (`cosmology_grid_posteriors`, `madau_dickinson_grid`,
  `shot_noise_coverage`) evaluates on a 1-2 dimensional grid. None uses NUTS or
  gradients.
- `shot_noise_coverage` scores 1000 injections at every grid point.
- Every caller passes `density_sites=()`, so the importance weights are one and
  the catalog Gram matrix is a constant.
- Fiducial grid: $F \simeq 400$ bins (`config/ensembles/fiducial/waveform.toml`).
  Pairs per network: ET-triangle 3, ET-triangle + CE 6, ET-2L-mis 1,
  ET-2L-mis + CE 3. So $D \le 2400$.

## Where the code stands

`inference/likelihood.py` (1500 lines) carries three likelihood classes and five
covariance models:

| Model | Where | Fate |
|---|---|---|
| Diagonal $\Sigma$ | `GaussianLikelihood`; `shot_noise=None` | Kept as the $\mathcal C^{\rm Pois} = 0$ limit of the one model |
| Rank one along the template, `"amplitude"` / `"fixed"` | `ImportanceGaussianLikelihood`, `shot_noise.py` | Deleted |
| Rank one per bin, `"per_frequency"` | same | Deleted |
| Top-$k$ eigenmodes, `"modes"` | same, `Network.modes/mode_blocks/mode_covariance` | Deleted |
| Exact compound-Poisson amplitude density | `ImportanceCompoundPoissonLikelihood`, `compound_poisson.py` | Deleted; its cumulants survive as a diagnostic (D7) |
| Combined `(F,)` spectrum ("`P = 1`, unit gain") | `_collapsed_response`, `_orientation_variance`, `network_response` | Deleted |

The response enters as Monte Carlo draws `(L, P, F)` held on the `Network` and
contracted inside the likelihood.

## Decisions

### D1. One covariance model: the dense $C$

The likelihood computes $C = \Sigma + M \circ R$ and its Cholesky factor at
every grid point and network. There is no rank-one, low-rank or non-Gaussian
alternative. A `shot_noise=False` switch on the model drops $M$ and gives the
diagonal Sec. 2.6 likelihood with the same code path (the factor is then the
diagonal).

If the dense solve later proves too slow, a low-rank solve may come back as an
option of the factorization (D2), never as a separate covariance model.

### D2. Three stages: predict, factorize, score

The cost of one evaluation splits by what it depends on:

| Stage | Depends on | Output | Cost |
|---|---|---|---|
| predict | $\Lambda$ | mean $S$ `(F,)`, power moment $M$ `(F, F)` | catalog contraction per redshift node |
| factorize | $\Lambda$, network | Cholesky of $C$, $\ln\det C$ | $O(D^3)$ |
| score | $\Lambda$, network, realization | $\chi^2$ | $O(D^2)$ per realization |

The API exposes the stages instead of hiding them behind
`params -> (log L, extras)`:

```python
class Prediction(NamedTuple):
    mean: jax.Array                  # (F,)
    power_moment: jax.Array | None   # (F, F); None without shot noise
    extras: Mapping[str, jax.Array]  # total_merger_rate, importance_relative_ess


class GaussianFactor(NamedTuple):
    mean: jax.Array                  # (D,)
    cholesky: jax.Array              # (D, D) lower, or (D,) for a diagonal C
    log_det: jax.Array               # ()


@dataclass(frozen=True)
class CatalogModel:                  # was ImportanceGaussianLikelihood
    @classmethod
    def from_catalog(cls, data, metadata, *, population, frequencies,
                     num_redshift_nodes, observation_time, density_sites=(),
                     minimum_redshift=None, shot_noise=True) -> Self: ...
    def spectrum(self, params) -> jax.Array: ...          # (F,)
    def power_moment(self, params) -> jax.Array: ...      # (F, F)
    def predict(self, params) -> Prediction: ...


def factorize(prediction: Prediction, network: Network) -> GaussianFactor: ...
def log_likelihood(factor: GaussianFactor, observed: ArrayLike) -> jax.Array:
    """observed (..., D) on the network's kept entries; returns (...)."""
```

`power_moment` is today's `_fold_node_moments` summed with the variance kernel
and divided by $T$: $M = T^{-1} \sum_z \kappa_z s_z^4 W_z G W_z^T$. `extras`
keeps what the notebooks already plot.

The model no longer holds data or a network. Data enter at the score stage, so
one factor scores every realization: `log_likelihood` takes `(R, D)` and does
one triangular solve with $R$ right-hand sides. For `shot_noise_coverage` that
turns 1000 Cholesky factorizations per grid point and network into one.

### D3. Response moment: Monte Carlo, once per network

$R$ is estimated from isotropic draws of $(\hat n, \psi, \cos\iota)$ with the
existing `sample_pair_response`, reduced once:

```python
def pair_response_moment(key, num_draws, frequencies, detectors, *,
                         chunk_size=4096) -> jax.Array:
    """(2/5)^2 <r r^T> over the draws, (P, F, P, F)."""
```

It accumulates `flat.T @ flat` over chunks of draws with `lax.map`, so the
`(L, P, F)` draws never sit in memory together, and $L$ is limited only by
time: $10^6$ draws at $D = 2400$ is about $6 \times 10^{12}$ flops, once.

A sample second moment is positive semidefinite, so $C$ stays positive definite
and the Cholesky cannot fail because of $R$.

The Monte Carlo error is the same at every $\Lambda$, a fixed bias in $C$
rather than noise that averages out. #380 saw the moment still converging at
$2.6 \times 10^5$ draws for ET-2L-mis + CE. The convergence check is cheap:
compare the log-likelihood at the fiducials under two independent halves of the
draws, per network, and report it with the forecasts. The analytic evaluation
of App. B can replace the estimator later without touching the likelihood.

The likelihood never sees draws: `Network.response` and
`Network.response_moments` go.

### D4. Build structured, solve flat on the kept entries

A reshape between `(P, F)` and `(P F,)` is free, and the index order only
permutes $C$, which changes neither the quadratic form, the determinant nor the
dense Cholesky cost. Two things do matter:

- **Which entries.** The data are the distinct pairs $a < b$. An
  `(F, N_det, N_det)` array would duplicate $(a, b)$ and $(b, a)$, which makes
  $C$ exactly singular, and would add the autocorrelations, which the paper
  excludes.
- **Masked entries.** Today a masked entry is an identity row of a `(D, D)`
  matrix and still pays the $O(D^3)$. Instead, each network compresses to its
  kept entries once, outside JAX transformations. The kept set is ragged (each
  pair has its own usable band near ORF zeros and PSD cutoffs), which only a
  flat vector represents.

```python
@dataclass(frozen=True)
class Network:
    keep: np.ndarray                 # (D,) pair-major indices into (P, F)
    bins: np.ndarray                 # (D,) frequency index of each kept entry
    scale: jax.Array                 # (D,) sigma_ab,k
    pair_overlap: jax.Array          # (D,) gamma_ab(f_k)
    response_moment: jax.Array | None  # (D, D)

    @classmethod
    def from_pairs(cls, scale, pair_overlap, mask, response_moment=None) -> Self:
        """From (P, F) leaves and a (P, F, P, F) moment."""

    def compress(self, values: ArrayLike) -> jax.Array:
        """(..., P, F) -> (..., D)."""
```

Then factorize is a gather, a product and a Cholesky, with no tiling:

```python
mean = network.pair_overlap * prediction.mean[network.bins]
cov = prediction.power_moment[network.bins][:, network.bins] * network.response_moment
cov = cov + jnp.diag(network.scale**2)
```

### D5. No stacking, no padding

`vmap` over networks padded to $P_{\max} = 6$ made sense when the per-network
cost was $O(F)$. With an $O(D^3)$ solve, ET-2L-mis padded to six pairs does
$6^3 = 216$ times the work it needs. Networks are a Python loop inside the
traced grid function: unrolled at trace time, each at its own $D$, one
compilation for the whole set. `pad_pairs`, `MAX_PAIRS` and the stacked-network
contract go.

### D6. A grid driver in place of the NumPyro-shaped contract

```python
def grid_log_likelihood(
    model: CatalogModel,
    networks: Mapping[str, Network],
    observed: Mapping[str, ArrayLike],   # (..., D) per network
    grids: Mapping[str, ArrayLike],
    *,
    fixed: Mapping[str, ArrayLike] | None = None,
    chunk_size: int = 8,
) -> dict[str, jax.Array]:               # (*grid sizes, ...) per network


def grid_log_prior(priors, grids, *, fixed=None) -> jax.Array:  # (*grid sizes)
```

The body maps chunks of grid points with `lax.map(batch_size=chunk_size)`:
predict once, then loop over networks, factorize, score every realization. Peak
memory per point is one $M$ (1.3 MB at $F = 400$) plus one $C$ per network
(46 MB at $D = 2400$).

Without gradients or NumPyro, the `LogLikelihood` protocol, the pytree
registration of the likelihood classes, `log_likelihood_from_prediction`,
`grid_log_posterior` and `LogDensityFn` (unused outside `inference/`) go. The
model stays a jit argument with its arrays as traced inputs, so one compilation
serves every catalog of one shape.

Rough cost at the largest network: 4.6 Gflop per Cholesky, so a $10^4$-point
grid over four networks takes on the order of half an hour on CPU and minutes
on GPU.

### D7. Diagnostics, not likelihoods

The paper reports quantities derived from the dense covariance. They move to
`inference/diagnostics.py`, each a function of a `Prediction` and a `Network`:

- $\sigma_{\rm shot}^2 = \mathbf c^T \mathcal C^{\rm Pois} \mathbf c / \rho^4$
  with $\mathbf c = \Sigma^{-1}\boldsymbol\mu$ (Sec. 4.3; today
  `relative_variance`).
- The amplitude-shape correction $\delta$ of App. `fisher-amplitude`, and the
  eigenvalues of the whitened covariance, projected orthogonal to the mean
  (Sec. 4.4 and its figure).
- The skewness of the amplitude estimator $\hat a$ (Sec. 4.4,
  App. `catalog-statistics`), from `compound_poisson.cumulants` and the point
  set. The saddlepoint and Gauss-Hermite densities go.

`fisher.py` gains the dense Gaussian Fisher matrix with the
$\tfrac12 {\rm Tr}[(C^{-1}\partial C)^2]$ term, which App. `fisher-amplitude`
asks to check against the mean term.

### D8. Deleted

- `ImportanceCompoundPoissonLikelihood`, `amplitude_log_density`,
  `AmplitudeDensityMethod`, `saddle`, `saddlepoint_log_density`,
  `hermite_log_density`, `gaussian_log_density`, `inclination_quadrature`
  (unless the skewness diagnostic needs it).
- `ShotNoiseMode`, `shot_noise_log_likelihood`, `rank_one_gaussian_log_likelihood`,
  `low_rank_gaussian_log_likelihood`, `shot_noise_modes`, `pad_modes`,
  `amplitude_direction`, `per_frequency_direction`,
  `amplitude_shot_noise_variance`; `shot_noise.py` as a whole.
- `GaussianLikelihood`, `ImportanceGaussianLikelihood` and their
  `with_shot_noise_modes`, `mode_blocks`, `response_moments`,
  `relative_variance`, `squared_intensity`, `mean_squared_intensity`.
- `network_response`, `sample_network_response`.
- `paper.posteriors.shot_noise_variant`, `pad_pairs`, `network_data` (the
  combined-spectrum one).
- `scripts/shot_noise_modes_grids.py`; `scripts/common_shape_response.py` if its
  sky-averaged check is no longer cited.

## Module layout after the change

```
inference/
    gaussian.py      # GaussianFactor, factorize, log_likelihood, Network
    model.py         # CatalogModel, Prediction
    grid.py          # grid_log_likelihood, grid_log_prior
    diagnostics.py   # sigma_shot, delta, whitened spectrum, amplitude cumulants
    fisher.py        # + dense Fisher
detector/response.py # + pair_response_moment
```

New modules let phase 1 land beside `likelihood.py` without name clashes; phase
4 deletes `likelihood.py`, `shot_noise.py`, `compound_poisson.py`,
`density.py` and `protocol.py`.

## Tests

- **T1 dense Gaussian.** `log_likelihood` of a small random factor equals
  `scipy.stats.multivariate_normal.logpdf`; a diagonal factor equals the sum of
  univariate log densities.
- **T2 compression.** The log-likelihood does not depend on the data, scale or
  response at masked entries, including infinite scales.
- **T3 realizations.** Scoring `(R, D)` at once equals scoring each realization.
- **T4 regression.** At the fiducials, `factorize` gives the same covariance as
  today's `catalog_covariance` under the same response draws (two code paths,
  kept until phase 4).
- **T5 chunking.** `pair_response_moment` is independent of `chunk_size` for a
  fixed key.
- **T6 grid driver.** `grid_log_likelihood` equals predict-factorize-score at
  each point; one compilation per model structure and network set.
- **T7 limit.** `shot_noise=False` gives the diagonal Gaussian, and
  `power_moment` scaled to zero reproduces it.
- **Integration.** `shot_noise_coverage` on the dense likelihood: 68% and 95%
  coverage of $H_0$ per network (`@pytest.mark.integration`).

## Phases (one PR each)

1. **Dense likelihood beside the old one.** `pair_response_moment`,
   `inference/gaussian.py`, `model.py`, `grid.py`; T1-T7.
2. **Diagnostics.** `diagnostics.py`, the dense Fisher matrix; port
   `scripts/shot_noise_budget.py` and the covariance-spectrum figure.
3. **Paper application.** `paper/posteriors.py` builds compressed `Network`s;
   the grid, coverage, Madau-Dickinson, ensemble and importance-convergence
   notebooks switch over; re-run and regenerate the figures and tables.
4. **Delete.** Everything in D8, the old tests
   (`test_likelihood.py`, `test_shot_noise.py`, `test_compound_poisson.py`,
   `test_inference_density.py` in their current form) and the stale
   `stacked-pair-likelihood.md` sections.

## Paper impact

- Sec. 3.3 says the forecasts use NUTS; they use grids. It should describe the
  grid evaluation.
- Sec. 3.3 says the response moment is "computed once following Appendix"; it
  becomes a Monte Carlo estimate over $L$ isotropic draws, with the split-half
  convergence check of D3.

## Open questions

- **Importance weights.** Keep `density_sites` in `CatalogModel`? The paper
  fixes the intrinsic distribution ($w_i = 1$), and dropping it would make the
  Gram matrix always constant. Keeping it costs one weighted Gram product per
  point, paid only when sites are given.
- **Draw count.** Fix $L$ per network from the split-half check, or one $L$ for
  all networks?
- **Dense cost on CPU.** If the half-hour estimate holds, decide between GPU
  runs and a low-rank solve option on `factorize` (D1).
