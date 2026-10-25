# Design: the stacked-pair likelihood

Status: proposal. Branch base: `feat/shot-noise-modes` at `c6259a6` (= `main`
at `3a2a0c4`, #380).

## Purpose

The paper (outline "Computation checklist", main.tex Sec. 3.3) describes the
likelihood of the stacked per-pair data vector $d_{ab,k}$, one entry per
detector pair and frequency bin, with the dense catalog covariance

$$
\mathrm{Cov}[d_{ab,k}, d_{cd,k'}] = \delta_{ab,cd}\delta_{kk'} \sigma_{ab,k}^2
    + \frac{\mathcal{R}}{T} \langle P_k P_{k'} \rangle
      \langle r_{ab,k} r_{cd,k'} \rangle .
$$

The code scores one combined spectrum per network instead. The combination is
a sufficient statistic only for diagonal detector noise. The shot noise
reaches each pair through its own $\gamma_{ab}(\hat n, \psi, \iota; f)$,
and the cross-pair moments $\langle r_{ab} r_{cd} \rangle$ are where
the geometrical shot noise and the polarization leakage live, so collapsing the
pairs first discards exactly the information the paper is about.

## Where the code stands

| Piece | Today | Per-pair information |
|---|---|---|
| `simulators/spectra/network.py` | `pair_sums` `(P, F)` per draw, then `_combine_pairs` to `(F,)`; only the combined spectrum is cached | computed, dropped |
| `detector/response.py` | `network_response` sums `w_ab * gamma_ab(n, psi, iota)` inside the pair loop, returns `(..., F)` | computed, dropped |
| `inference/likelihood.py` | `Network.scale`, `mask`, `observed`, `response` `(L, F)`, `modes` `(F, k)`, `response_moments` `(Z, F, F)` share one `F` axis with the prediction | absent |
| `paper/posteriors.py::network_data` | `gaussian_bin_scale(effective_psd(...))`, one `(F,)` scale | absent |
| Exact response moments (App. B) | none; Monte Carlo draws only | n/a |

Detector pairs per network: ET-triangle 3, ET-triangle + CE 6, ET-2L 1,
ET-2L + CE 3. Stacking networks along `vmap` needs a common `P_max = 6`.

## Decisions

### D1. Units of the per-pair data: unnormalized cross-power

Store and score $Y_{ab,k}$ with mean $\gamma_{ab}(f_k) S_h(f_k)$,
that is $5 k D_{ab}$ in the simulator's notation, and noise

$$
\sigma_{ab,k} = \frac{\sqrt{S_a S_b}}{\tfrac{2}{5}\sqrt{2 T \Delta f_k}},
$$

so that $\sum_{ab} \gamma_{ab,k}^2 / \sigma_{ab,k}^2 = 1/\sigma_k^2$
reproduces `gaussian_bin_scale(effective_psd(...))` bin by bin.

Not $Y_{ab}/\gamma_{ab}$ (mean $S_h$). That form diverges at the
zeros of $\gamma_{ab}$, and the per-source response
$\gamma_{ab}(\hat n\ldots)/\gamma_{ab}$ becomes heavy-tailed for pairs
with small overlap (ET-2L S2-R2 has $\gamma \simeq 0.09$ at 10 Hz).
The two forms differ by a fixed per-entry rescaling, so they give the same
posterior. The paper keeps the normalized notation
$\mu_{ab,k} = S_h(f_k)$ and $\Delta S_{h,ab}$; the code uses the
unnormalized form for numerical stability. Negative overlaps (ET-triangle,
$-0.37$) need no special case.

### D2. Layout: flatten (pair, bin) onto one data axis; `predict` stays `(F,)`

The expensive prediction does not depend on the network, and that is what lets
`vmap` over networks share it. Keep it: `predict(params)` still returns
`(F,)`, and the network expands it.

New `Network` leaf:

- `pair_overlap`: $\gamma_{ab}(f_k)$, `(P, F)`. Padded pairs are zero
  and masked.

These leaves move to the stacked axis, `(P, F)` (flattened pair-major to
`D = P F` where a matrix needs it): `scale`, `mask`, `observed`, `response`
`(L, P, F)` and `modes` `(D, k)`.

`pair_overlap = None` means `P = 1` with unit gain, the current
combined-spectrum behaviour. That keeps the existing notebooks and tests running,
and gives a regression test for free (T2).

The expanded mean is `pair_overlap * mean[None, :]`. It is formed only in
`log_likelihood_from_prediction` and the projectors.

### D3. Keep every per-point contraction at length `F`

The parameter dependence of the projectors factors out of the pair axis. The
template projector is

$$
c_{ab,k} = \frac{\gamma_{ab,k} S_k}{\sigma_{ab,k}^2}
         = S_k\, g_{ab,k}, \qquad g_{ab,k} = \gamma_{ab,k}/\sigma_{ab,k}^2,
$$

with $g$ fixed per network. So the amplitude mode's projected second
moment at node $z$ is

$$
\mathbf{c}^T (M_z \circ R) \mathbf{c}
  = \mathbf{S}^T (M_z \circ \bar R) \mathbf{S}, \qquad
\bar R_{kk'} = \sum_{ab,cd} g_{ab,k} \langle r_{ab,k} r_{cd,k'} \rangle g_{cd,k'},
$$

an `(F, F)` matrix per network, formed once. The held `response_moments` become
$K_z = M_z \circ \bar R$, still `(Z, F, F)`. The amplitude mode then costs
the same as today per grid point, whatever `P` is.

With response draws, the same collapse is one effective projector per draw:
$p_{l,k} = \sum_{ab} c_{ab,k}\, r_{l,ab,k}$. That feeds the existing
`_projected_second_moment` unchanged.

Modes: $M_z^{\rm modes} = \tilde U^T (M_z \circ R) \tilde U$ with
$\tilde U$ `(D, k)`. With draws this is
$\frac1L \sum_l Q_l^T M_z Q_l$ with
$Q_{l,k,i} = \sum_{ab} \tilde U_{ab,k,i} r_{l,ab,k}$, `(F, k)`, so the
`(D, D)` matrix is never formed per node. It is formed once, at build time, for
the eigen-decomposition that picks the basis (`shot_noise_modes`). With
`D = 6F` that is a single dense `eigh`, affordable outside JAX
transformations.

Do not hold `(Z, D, D)` response moments: at `P = 6` they are 36 times
today's.

### D4. Response moments: Monte Carlo now, exact later behind the same seam

Phase 1 samples per-pair draws `(L, P, F)` from `overlap_reduction_function`,
the per-pair loop body of `network_response` with the sum over pairs removed.
The likelihood consumes moments only through two quantities: $\bar R$
(D3) and the per-node mode blocks. The exact Clebsch-Gordan evaluation (App. B)
later supplies $\langle r_{ab,k} r_{cd,k'} \rangle$ directly, `(D, D)`,
and replaces the draws without changing the likelihood. #380 found the Monte
Carlo second moment still converging at $2.6\times10^5$ draws for
ET-2L + CE, which is the argument for doing the exact evaluation and not
just increasing `L`.

### D5. Simulator: emit per-pair spectra, derive the combined one

`NetworkSpectralDensitySimulator` and `JointNetworkSpectralDensitySimulator`
add `pair_spectral_density`, `(1, P, F)`, normalized like the combined spectrum,
plus the pair list. The combined spectrum stays as a derived field. The cache
key changes, so the #380 injection caches are re-drawn once (about 45 s per
paper-size catalog on an M4, per the #380 notes).

### D6. Shot-noise modes per point

| `ShotNoiseMode` | Stacked form |
|---|---|
| `None` | diagonal over `D` |
| `"amplitude"` / `"fixed"` | direction `pair_overlap * mean`, $s^2$ from D3 |
| `"per_frequency"` | per-entry variance $V_k \langle r_{ab,k}^2 \rangle$, rank one along $\sqrt{V}$ |
| `"modes"` | basis on `D`, blocks via D3 |

`ImportanceCompoundPoissonLikelihood` (exact amplitude density, App. A and
`shot_noise_exact_vs_gaussian_H0`) keeps the combined spectrum in phase 2 and
moves in phase 5.

## Tests

- **T1 sufficiency.** No shot noise: the stacked log-likelihood minus the
  combined one is the same constant at every parameter point (checked over a
  parameter grid; differences of log-likelihoods, no tolerance tuning).
- **T2 compatibility.** `pair_overlap=None` gives exactly today's values for
  every `ShotNoiseMode`.
- **T3 collapse.** The held $\bar R$ path and the per-draw projection
  path give the same $s^2$ and the same mode blocks (two code paths
  against each other, not a re-implementation).
- **T4 simulator.** Combining `pair_spectral_density` with the optimal
  weights returns the combined spectrum of the same draw.
- **T5 stacking.** A `vmap` over networks padded to `P_max` matches the
  networks scored one at a time.
- **Integration.** `shot_noise_coverage.py` with stacked pairs: 68% and 95%
  coverage of $H_0$ per network (`@pytest.mark.integration` for any test
  form).

## Phases (one PR each)

1. **Per-pair response and simulator output.** `sample_pair_response`
   `(L, P, F)`, `pair_spectral_density` in both network simulators, new cache
   keys, T4.
2. **Stacked `Network` and likelihood.** `pair_overlap`, stacked leaves, D3
   collapse, all five shot-noise modes, T1-T3, T5.
3. **Paper application.** `network_data` returns per-pair `scale`, `mask` and
   `pair_overlap`, padded to `P_max`; the grid, coverage, Madau-Dickinson and
   ensemble notebooks switch to stacked data; re-run.
4. **Exact response moments (App. B).** Clebsch-Gordan evaluation of
   $\langle r_{ab,k} r_{cd,k'} \rangle$ with spherical Bessel functions
   evaluated directly. Validate against the sky quadrature and Monte Carlo, and
   make it the default source of $\bar R$.
5. **Compound-Poisson likelihood on stacked pairs.** Then regenerate
   `shot_noise_exact_vs_gaussian_H0` and the skewness numbers.

After phase 3, the paper's figures and tables are regenerated from these runs:
`minimum_redshift_scan`, `shot_noise_density_spectrum` (both panels),
`mcmc_cosmological_parameters_H0_by_detector`, `corner_H0_Omega_m`,
`corner_xi_0_xi_n`, `grid_astrophysical_parameters_MD_corner`, the H0 and
Xi0 tables, and the new shot-noise budget table.

## Open questions

- Fix the basis for the `"modes"` likelihood at the fiducials, or recompute it
  per $\Lambda$? Fixed is what `with_shot_noise_modes` does today, and
  the paper's Sec. 3.3 has to say which.
- Report the sum of dropped eigenvalues per network, as the outline asks:
  computed from the build-time `eigh` (D3) at no extra cost.
- Earth rotation: the static per-pair response stays an upper bound at low
  $f$. Out of scope here.
