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
alternative. A `shot_noise=False` switch on the likelihood drops $M$ and gives the
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

One likelihood is one network: different networks are different models, and
the notebooks loop over them. The public entry point stays
`log_likelihood(params) -> (log L, extras)`; the stages are private methods
whose results are returned, never stored on the instance (a field holding a
`params`-dependent value goes stale silently under `jit`).

```python
class Prediction(NamedTuple):
    mean: jax.Array                  # (F,)
    power_moment: jax.Array | None   # (F, F); None without shot noise
    extras: Mapping[str, jax.Array]  # total_merger_rate, importance_relative_ess


class GaussianFactor(NamedTuple):
    mean: jax.Array                  # (D,)
    cholesky: jax.Array              # (D, D) lower, or (D,) for a diagonal C
    log_det: jax.Array               # ()


class Evaluation(NamedTuple):        # what the diagnostics read (D7)
    prediction: Prediction
    factor: GaussianFactor


@dataclass(frozen=True)
class CatalogModel:                  # was ImportanceGaussianLikelihood, minus data and network
    @classmethod
    def from_catalog(cls, data, metadata, *, population, frequencies,
                     num_redshift_nodes, observation_time, density_sites=(),
                     minimum_redshift=None) -> Self: ...
    def spectrum(self, params) -> jax.Array: ...          # (F,)
    def power_moment(self, params) -> jax.Array: ...      # (F, F)
    def predict(self, params, *, shot_noise: bool = True) -> Prediction: ...


@dataclass(frozen=True)
class DenseGaussianLikelihood:
    model: CatalogModel              # data field
    network: Network                 # data field
    observed: jax.Array              # data field, (..., D) on the kept entries
    shot_noise: bool = True          # meta field

    def log_likelihood(self, params) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """log L of shape (...), one per realization, and the prediction's extras."""
    def evaluate(self, params) -> Evaluation: ...
    def _factorize(self, prediction: Prediction) -> GaussianFactor: ...
    def _score(self, factor: GaussianFactor) -> jax.Array: ...
```

Every class is a registered pytree, so a likelihood is a `jit` argument with
its arrays traced: one compilation per network shape serves every catalog and
data set of that shape.

`power_moment` is $M = T^{-1} \sum_z \kappa_z s_z^4 W_z G W_z^T$, today's
`_fold_node_moments` summed with the variance kernel, computed sparsely (D2a).
`extras` keeps what the notebooks already plot.

Realizations share the factor without any `vmap`: `_score` is one triangular
solve against `observed` reshaped to `(R, D)`, with $R$ right-hand sides. For
`shot_noise_coverage` that turns 1000 Cholesky factorizations per grid point
into one. (A `vmap` over `observed` would give the same: values that do not
depend on the batched argument stay unbatched.)

### D2a. The power moment from sparse interpolation

With one likelihood per network the prediction is recomputed per network, so it
must be cheap. Today it is not: `_fold_node_moments` forms each node's
interpolation as a dense `(F, F_ref)` matrix (`jnp.interp` over the identity)
and multiplies through, $Z \times 2(F F_{\rm ref}^2 + F^2 F_{\rm ref})
\simeq 8$ Gflop per point at $Z = 32$ and $F \simeq F_{\rm ref} \simeq 400$,
more than the 4.6 Gflop Cholesky of the largest network.

Linear interpolation has at most two nonzeros per row: the reference index
$i_k$ below query $k$ and weights $(1 - t_k, t_k)$. So

$$
(W_z G W_z^T)_{kl} = \sum_{a, b \in \{0, 1\}} w_{k,a}\, w_{l,b}\,
    G_{i_k + a,\, i_l + b},
$$

four gathered `(F, F)` slices of $G$ per node, about $Z \times 4F^2 \simeq 20$
Mflop:

```python
def node_moment(index, weights, gram):    # index (F,), weights (F, 2)
    rows = jnp.stack([index, index + 1], axis=1)
    block = gram[rows[:, :, None, None], rows[None, None, :, :]]  # (F, 2, F, 2)
    return jnp.einsum("ka,kalb,lb->kl", weights, block, weights)
```

$(i_k, t_k)$ come from one `searchsorted` per node on the redshifted query
frequencies, shared with the mean (`_node_intensity`). Queries above the
reference grid get zero weights, matching `jnp.interp(..., right=0.0)`; queries
below it clamp to the first reference value, matching `jnp.interp`'s default.
The mean can use the same weights.

What is independent of both $\Lambda$ and the network is built once and shared
by reference: the catalog arrays and the Gram matrix $G$ on `CatalogModel`
(every network's likelihood holds the same model, so the device buffers are not
copied), and the response draws when every network is sampled from one key.

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
$6^3 = 216$ times the work it needs. Networks are a loop in the notebook, one
likelihood each, compiled at its own $D$: four compilations for the paper's four
networks. `pad_pairs`, `MAX_PAIRS` and the stacked-network contract go.

### D6. A grid driver in place of the NumPyro-shaped contract

```python
def grid_log_likelihood(
    likelihood: DenseGaussianLikelihood,
    grids: Mapping[str, ArrayLike],
    *,
    fixed: Mapping[str, ArrayLike] | None = None,
    chunk_size: int = 8,
) -> jax.Array:                          # (*grid sizes, ...realizations)


def grid_log_prior(priors, grids, *, fixed=None) -> jax.Array:  # (*grid sizes)
```

The body is `lax.map(likelihood.log_likelihood, points, batch_size=chunk_size)`
under one module-level `jit`, with the likelihood a pytree argument. A notebook
reads:

```python
model = CatalogModel.from_catalog(...)
log_prior = grid_log_prior(priors, grids)
posteriors = {
    name: grid_log_likelihood(DenseGaussianLikelihood(model, network, data[name]), grids)
    + log_prior
    for name, network in networks.items()
}
```

Peak memory per point is one $M$ (1.3 MB at $F = 400$) plus one $C$ (46 MB at
$D = 2400$).

Without gradients or NumPyro, the `LogLikelihood` protocol,
`log_likelihood_from_prediction`, `grid_log_posterior` and `LogDensityFn`
(unused outside `inference/`) go.

Rough cost: with D2a the prediction is negligible, and the four networks'
Cholesky factors total about 6 Gflop per point, so a $10^4$-point grid takes on
the order of half an hour on CPU and minutes on GPU.

### D7. Diagnostics, not likelihoods

The paper reports quantities derived from the dense covariance. They move to
`inference/diagnostics.py`, each a function of an `Evaluation` and its
`Network` (`likelihood.evaluate(params)`, typically at the fiducials):

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
    gaussian.py      # Network, GaussianFactor, Evaluation, DenseGaussianLikelihood
    model.py         # CatalogModel, Prediction, sparse node moments
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
- **T6 grid driver.** `grid_log_likelihood` equals `log_likelihood` at each
  point; one compilation per likelihood structure, shared by two catalogs or
  data sets of one shape.
- **T8 sparse moment.** The gathered `power_moment` equals the dense
  interpolation path, including queries above and below the reference grid.
- **T7 limit.** `shot_noise=False` gives the diagonal Gaussian.
- **Integration.** `shot_noise_coverage` on the dense likelihood: 68% and 95%
  coverage of $H_0$ per network (`@pytest.mark.integration`).

## Phases (one PR each)

1. **Dense likelihood beside the old one.** `pair_response_moment`,
   `inference/gaussian.py`, `model.py` with the sparse power moment, `grid.py`;
   T1-T8.
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
