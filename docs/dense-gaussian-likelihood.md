# Design: the dense Gaussian likelihood

Status: implemented; the model interface was later reorganized behind the
`SpectralDensityModel` protocol and the paper's connected overlap. Supersedes
the likelihood parts of the stacked-pair design (#382), whose per-pair data
layout and simulator output are kept.
Paper-side paths named below (`paper/posteriors.py`,
`scripts/shot_noise_budget.py`) now live in `experiments-astrogwb`.

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
covariance. For the dominant quadrupole the connected term factorizes
(`eq:cross-correlation-statistic-covariance-connected-quadrupole`) into a
moment of the source intensity and an average over the network geometry,

$$
\mathcal C^{\rm Pois}_{ab,k;\,cd,l} = \frac{1}{T}\,X_{kl}(\Lambda)\,
    \Gamma^c_{ab,k;\,cd,l},
\qquad
X_{kl} = 4\mathcal R \langle I_k I_l \rangle_{\theta,\iota|\Lambda},
\qquad
\Gamma^c = \Gamma^{II} + \frac{\Gamma^{QQ} + \Gamma^{UU}}{71}
    + \frac{69}{71}\Gamma^{VV}.
$$

$X$ is the raw intensity second moment, `(F, F)`, per unit time; it depends on
$\Lambda$ but not on the network, and carries the inclination average. The
connected overlap $\Gamma^c$ is `(D, D)` and depends on the network but not on
$\Lambda$: the sky and polarization average of the pair responses to each
Stokes parameter. Every design decision below follows from that split.

Units are those of the stacked-pair design: unnormalized cross powers
with mean $\mu_{ab,k} = \gamma_{ab}(f_k) S_h(f_k|\Lambda)$, which differ from
the paper's normalized $\bar C_{ab}$ by a fixed per-entry rescaling and give
the same posterior. The connected overlap is in the units of the normalized
isotropic ORF $\gamma_{ab}$; `normalize=False` gives the paper's
$\Gamma^{\mu\nu}$.

With $\mathcal C^{\rm Pois} = 0$ this is the detector-noise-only likelihood of
Sec. 2.6.

## How the likelihood is used

- Every posterior notebook (`cosmology_grid_posteriors`, `madau_dickinson_grid`,
  `shot_noise_coverage`) evaluates on a 1-2 dimensional grid. None uses NUTS or
  gradients.
- `shot_noise_coverage` scores 1000 injections at every grid point.
- The paper notebooks set `target_matches_proposal=True`, so the importance
  weights are one and the catalog Gram matrix is constant.
- Fiducial grid: $F = 199$ bins, 1 Hz from 2 to 200 Hz (`[ensembles.fiducial]`,
  i.e. `[waveforms.default]`); the estimates below were made at the earlier
  $F \simeq 400$ to 2048 Hz. Pairs per network: ET-triangle 3, ET-triangle +
  CE 6, ET-2L-mis 1, ET-2L-mis + CE 3. So $D \le 1200$.

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
class SpectralDensityPrediction(NamedTuple):
    spectral_density: jax.Array                  # (F,), S_h = 2 R <I>
    intensity_second_moment: jax.Array | None    # (F, F), 4 R <I I'>; None without shot noise
    extras: Mapping[str, jax.Array]              # fixed keys of scalar diagnostics


class SpectralDensityModel(Protocol):            # a pytree
    def __call__(self, params, *, shot_noise: bool) -> SpectralDensityPrediction: ...


class DenseGaussianLikelihood[M: SpectralDensityModel]:   # registered pytree
    def __init__(self, model: M, network: str | Sequence[Detector], *,
                 frequencies, observation_time,           # years
                 observed: Mapping[tuple[str, str], ArrayLike] | None = None,
                 mask=None, shot_noise=True, key=None, num_draws=2**15): ...
    def log_likelihood(self, params) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """log L of shape (...), one per realization, and the prediction's extras."""
    def mean(self, params) -> jax.Array: ...                # (D,)
    def covariance(self, params) -> jax.Array: ...          # (D, D)
    def solve(self, x, params) -> jax.Array:
        """C^{-1} x for x of shape (..., D), from the Cholesky factor."""
```

Internally each call factorizes the covariance once (its Cholesky factor, or
the noise scale when `C` is diagonal) and evaluates the density, the
covariance or the solve from that factor; the factor itself is private.

The likelihood's children are the model, its private network and the data;
`shot_noise` is static. JAX rebuilds it without calling `__init__`, which
resolves the detectors, builds the noise scale and ORFs and draws the
connected overlap once, outside JAX transformations: one compilation per
structure and shape serves every model and data set.

The model is any pytree implementing the protocol: `CatalogModel`
(`astrogwb.background.importance.model`), or a `jax.tree_util.Partial` of a module-level
function. `extras` keeps what the notebooks already plot.

Realizations share the factor without any `vmap`: the score is one triangular
solve against `observed` reshaped to `(R, D)`, with $R$ right-hand sides. For
`shot_noise_coverage` that turns 1000 Cholesky factorizations per grid point
into one.

### D2a. The intensity second moment from sparse interpolation

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

The redshift nodes and the reference redshift are fixed, so the query
frequencies $f_k s_z$, and with them $(i_k, t_k)$, do not depend on
$\Lambda$: they are formed once (`interpolation_weights`). Queries above the
reference grid get zero weights, matching `jnp.interp(..., right=0.0)`; queries
below it clamp to the first reference value, matching `jnp.interp`'s default.

With `target_matches_proposal=True`, $G$ is constant too, so the node
moments $s_z^8 W_z G W_z^T$, `(Z, F, F)` (41 MB at $Z = 32$, $F = 400$), are
formed once in `from_catalog`, and the second moment at each point is one
contraction with the $\Lambda$-dependent kernel, $Z F^2 \simeq 5$ Mflop. The gather
above runs per point only when the importance weights vary.

What is independent of both $\Lambda$ and the network is built once and shared
by reference: the catalog arrays and the Gram matrix $G$ on `CatalogModel`
(every network's likelihood holds the same model, so the device buffers are not
copied), and the response draws when every network is sampled from one key.

### D3. Generalized ORFs: sky Monte Carlo, once per network

The paper's generalized ORFs
$\Gamma^{\mu\nu}_{ab,cd} = 4\int d^2\hat n/4\pi\, F^\mu_{(ab)} F^\nu_{(cd)}$
are estimated from isotropic sky draws at zero polarization angle; the
polarization angle and the inclination belong to the source moments. The
likelihood combines them with the quadrupole's
$\langle \mu\nu'\rangle / \langle II'\rangle$
(`waveform.intensity.QUADRUPOLE_STOKES_WEIGHTS`),
$\Gamma^c = \Gamma^{II} + (\Gamma^{QQ} + \Gamma^{UU})/71 + \tfrac{69}{71}\Gamma^{VV}$,
in the units of the normalized isotropic ORF ($\gamma_{ab} = \tfrac52\Gamma_{ab}$,
`detector.NORMALIZED_ORF_SCALE`):

```python
def generalized_overlap_reduction_function(
    key, num_draws, frequencies, detectors, *,
    components=("II", "QQ", "UU", "VV"), chunk_size=4096,
) -> dict[StokesPair, jax.Array]:  # each (P, F, P, F)
```

It accumulates `flat.T @ flat` over chunks of draws with `lax.scan`, so the
draws never sit in memory together. Each $\Gamma^{\mu\mu}$ is a sum of outer
products, positive semidefinite, and so is their positive combination; with a
positive semidefinite $X$, the Schur product $X \circ \Gamma^c$ is too, so the Cholesky cannot fail because of either.

The Monte Carlo error is the same at every $\Lambda$, a fixed bias in $C$
rather than noise that averages out. The convergence check is cheap: compare
the log-likelihood at the fiducials under two keys, per network. The analytic
evaluation can replace the estimator later without touching the likelihood.

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

The network is private to the likelihood: its kept indices, the frequency
bin of every entry, the noise scale, the ORF, the observation time and the
connected overlap on the kept entries. The constructor counts the entries with
finite noise, optionally masked per pair, and maps the pair-keyed data onto
them. Then factorize is a gather, a product and a Cholesky, with no tiling:

```python
mean = network.pair_overlap * prediction.spectral_density[network.bins]
cov = prediction.intensity_second_moment[network.bins][:, network.bins]
cov = cov * network.connected_overlap / seconds + jnp.diag(network.scale**2)
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
    name: grid_log_likelihood(
        DenseGaussianLikelihood(model, detectors, frequencies=frequencies,
                                observation_time=1.0, observed=data[name]),
        grids,
    )
    + log_prior
    for name, detectors in networks.items()
}
```

Peak memory per point is one $X$ (1.3 MB at $F = 400$) plus one $C$ (46 MB at
$D = 2400$).

Without gradients or NumPyro, the `LogLikelihood` protocol,
`log_likelihood_from_prediction`, `grid_log_posterior` and `LogDensityFn`
(unused outside `inference/`) go.

Rough cost: with D2a the prediction is negligible, and the four networks'
Cholesky factors total about 6 Gflop per point, so a $10^4$-point grid takes on
the order of half an hour on CPU and minutes on GPU.

### D7. The amplitude statistic

The paper's amplitude statistic lives in `astrogwb.statistics.amplitude`. Only
the weights take the likelihood; the estimate and the variance take arrays:

- The weights $\mathbf w = \Sigma^{-1}\boldsymbol\mu_0 /
  \boldsymbol\mu_0^T\Sigma^{-1}\boldsymbol\mu_0$
  (`background_amplitude_weights(likelihood, params)`, needs no data), with
  `params` the template's parameters, typically the fiducials.
- The estimate $\hat x = \mathbf w^T \bar{\mathbf C}$ of each realization
  (`background_amplitude_estimate(weights, observed)`).
- The relative variance $\mathbf w^T\Sigma'\mathbf w$ of the estimator under a
  covariance $\Sigma'$ (`background_amplitude_variance(weights, covariance)`).
  With the likelihood's `covariance(params)` it is
  $1 / \boldsymbol\mu_0^T\Sigma^{-1}\boldsymbol\mu_0$, $\rho^{-2}$ without
  shot noise.
- A variance budget is the same call mapped over `covariance_components`,
  each term contributing $\mathbf w^T\Sigma^X\mathbf w$; the terms sum to the
  variance. It needs a likelihood built with `keep_generalized_orfs=True`,
  whose `covariance_components` splits $\Sigma$ into detector noise, count
  ($\Gamma^{II}$, needs the model's `merger_rate`), intensity ($\Gamma^{II}$
  less the count), linear ($\Gamma^{QQ}$, $\Gamma^{UU}$) and circular
  polarization ($\Gamma^{VV}$, and $\Gamma^{IV}$, $\Gamma^{VI}$ when kept).
  Keeping the ORFs changes memory, not the covariance.

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

## Module layout

```
inference/
    model.py         # SpectralDensityModel, SpectralDensityPrediction
    gaussian.py      # DenseGaussianLikelihood, private factor and network
    grid.py          # grid_log_likelihood, grid_log_prior
    diagnostics.py   # sigma_shot, delta, whitened spectrum, response templates
background/importance/
    model.py         # CatalogModel, sparse node moments
    diagnostics.py   # relative ESS, amplitude cumulants
detector/response.py # generalized_overlap_reduction_function, pair_names
```

## Paper impact

- Sec. 3.3 says the forecasts use NUTS; they use grids. It should describe the
  grid evaluation.
- Sec. 3.3 says the angular part is computed "by averaging the products of the
  per-source pair responses" over sky, polarization and inclination draws; it
  is the connected overlap, a Monte Carlo over sky and polarization with the
  inclination averaged exactly (D3).

## Open questions

- **Draw count.** Fix $L$ per network from the split-half check, or one $L$ for
  all networks?
- **Dense cost on CPU.** If the half-hour estimate holds, decide between GPU
  runs and a low-rank solve option on `factorize` (D1).
