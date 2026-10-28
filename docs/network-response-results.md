# Paper numbers with network-response injections

The grid, coverage and minimum-redshift analyses now inject the background as
each detector network records it, and every likelihood carries that network's
per-source response. This note collects the numbers that change, old against
new, so the paper text can be updated from it.

- **Old** is the paper as it stands: a sky-averaged Poisson injection
  (`BackgroundSpectralDensitySimulator`), one spectrum shared by the six
  networks, and shot noise with the sky-averaged inclination factor
  $\langle g^2 \rangle / \langle g \rangle^2 = 355/252$. Rerun from
  `origin/main` at `ffe174a`, it reproduces the paper's tables to the quoted
  digits.
- **New** injects the same Poisson catalog through
  `NetworkSpectralDensitySimulator`: every source reaches each detector pair
  through its antenna patterns at an isotropic sky position and polarization
  angle, and the pairs are combined with the optimal cross-correlation weights.
  The six networks see one catalog (the same sources, positions and angles)
  but record six spectra. Each network's likelihood uses its own
  `Network.response` (32768 isotropic draws, seed 0), so the catalog
  covariance is the Hadamard product
  $\langle P_f P_{f'} \rangle \circ \langle r_f r_{f'} \rangle$.

The response replaces the inclination factor, so it changes $\sigma_B$ but not
$\rho$. Everything else (population, waveform, reference catalog, seeds,
grids, $z_{\min} = 0.35$, $T = 1$ yr) is unchanged.

## Why $\sigma_B$ grows, and by how much

The estimator divides each detector pair by its isotropic overlap
$\gamma_{ab}$, the sky average of a source's cross-power. One source
contributes its own antenna-pattern product, of order one whatever
$\gamma_{ab}$ is, so its response $r$ scatters about one by roughly
$1/\gamma_{ab}$. Networks whose weight sits on pairs with small $\gamma_{ab}$
behave like catalogs of fewer, unequal sources. The SNR-weighted response
second moment reproduces the inflation of $\sigma_B$ exactly, with $\gamma$
and the pair weights at the median-SNR frequency (10-12 Hz):

| Network | pairs: $\gamma$ (weight) | $\sigma_B$ new/old |
|---|---|---|
| ET-$\Delta$ | three co-located pairs, $-0.37$ | 1.18 |
| ET-2L-par | S1-R1, $+0.97$ | 1.21 |
| ET-2L-par + CE | S1-R1 $+0.97$ (77%), ET-CE $0.19$-$0.28$ | 1.26 |
| ET-$\Delta$ + CE | ET pairs (57%), ET-CE $-0.12$ to $+0.17$ (43%) | 1.35 |
| ET-2L-mis + CE | S2-C1 $0.18$ (68%), R2-C1 $0.12$ (28%), S2-R2 $0.09$ | 2.22 |
| ET-2L-mis | S2-R2 $0.09$ | 3.52 |

## Shot noise and $H_0$

| Network | $10^3\sigma_B$ old | new | $\zeta=\rho\sigma_B$ old | new | $\sigma_{H_0}$ old | new | $\sigma_{H_0}/H_0$ (%) old | new | $(\rho^{-2}+\sigma_B^2)^{1/2}$ (%) new | $\sigma_{H_0}$ new/old |
|---|---|---|---|---|---|---|---|---|---|---|
| ET-$\Delta$ | 2.16 | 2.55 | 0.41 | 0.48 | 0.388 | 0.401 | 0.574 | 0.592 | 0.588 | 1.033 |
| ET-$\Delta$ + CE | 2.16 | 2.91 | 0.51 | 0.69 | 0.321 | 0.350 | 0.475 | 0.518 | 0.512 | 1.091 |
| ET-2L-par | 2.16 | 2.62 | 1.10 | 1.33 | 0.199 | 0.223 | 0.294 | 0.330 | 0.328 | 1.122 |
| ET-2L-par + CE | 2.16 | 2.72 | 1.18 | 1.49 | 0.192 | 0.223 | 0.284 | 0.329 | 0.327 | 1.159 |
| ET-2L-mis | 2.16 | 7.60 | 0.10 | 0.34 | 1.533 | 1.643 | 2.265 | 2.428 | 2.371 | 1.072 |
| ET-2L-mis + CE | 2.16 | 4.78 | 0.35 | 0.78 | 0.443 | 0.537 | 0.655 | 0.794 | 0.780 | 1.212 |

$\rho$ is unchanged (188.6, 237.4, 507.5, 548.7, 44.5, 162.2). $\sigma_{H_0}$ is
the half-width of the 68% HDI of the grid posterior, in
km s$^{-1}$ Mpc$^{-1}$; $H_0 = 67.66$. Old is the paper's `amplitude`
likelihood; new is the `"modes"` likelihood (the leading eigenmodes of the
whitened fiducial catalog covariance, dropped eigenvalue sum $10^{-3}$, with
their block evaluated at each grid point), which the H0, $(H_0, \Omega_m)$ and
$(\Xi_0, n)$ grids now report, as the Madau-Dickinson grid already did. Against
`amplitude` on the same injection it changes the half-widths by at most 0.9%
(ET-2L-mis + CE, 0.542 to 0.537) and $\sigma_{\Xi_0}$ by at most 0.7%.

- $\sigma_B$ now spans $2.6$-$7.6\times10^{-3}$ instead of $2.2\times10^{-3}$
  for every network.
- $\sigma_{H_0}$ grows by 3% to 21%; the relative precision spans 0.33% to 2.4%
  (was 0.28% to 2.3%).
- $\sigma_{H_0}/H_0$ follows $(\rho^{-2}+\sigma_B^2)^{1/2}$ to within 3% (was
  1%); each network now fits its own realization.
- The inflation $\sqrt{1+\zeta^2}$ is 1.66 and 1.80 for ET-2L-par and ET-2L-par
  + CE (was about 1.5), 1.27 for ET-2L-mis + CE and 1.06 for ET-2L-mis (was 1.005).
  The crossover $\zeta = 1$ is at $\rho = 1/\sigma_B \simeq 370$-$380$ for the
  ET-2L-par networks (was 450).
- **The posteriors no longer peak at one value.** Each network records its own
  amplitude fluctuation $\hat\epsilon$ (the injection projected on the
  template). For the common-shape likelihood the MAP is analytic,
  $H_0^{\rm MAP} = H_{0,t} / (\hat x - \tfrac12 v'(\hat x))$ with
  $\hat x = 1 + \hat\epsilon$ and $v = \rho^{-2} + x^2\sigma_B^2$, that is
  $H_{0,t}/(1+\hat\epsilon)$ up to $O(\sigma_B^2) < 0.004$; the grid MAPs match
  it to a few $10^{-4}$, and the `"modes"` grid MAPs agree with them to
  one grid step (0.02; ET-2L-mis peaks at 68.40 instead of 68.42). Here $\hat\epsilon = -1.05$, $-2.01$, $-0.99$, $-1.28$,
  $-1.47$, $-2.40\,\sigma_B$, so the `"modes"` medians are 67.85, 68.06, 67.84,
  67.89, 68.48, 68.45, that is $+0.47$, $+1.14$, $+0.81$, $+1.05$, $+0.50$,
  $+1.46$ posterior standard deviations. The peaks sit $0.18$ to
  $0.78$ km s$^{-1}$ Mpc$^{-1}$ above the fiducial. The networks share sources and
  angles, so their fluctuations are correlated; this realization is low for
  all of them. The sky-averaged spectrum of the same catalog has
  $\hat\epsilon = -0.49\,\sigma_B$.

## Coverage of the $H_0$ intervals

1000 Poisson catalogs at the fiducials, each recorded by every network
(`notebooks/shot_noise_coverage.py`), 20 detector-noise draws each, errors
bootstrapped over catalogs. The `amplitude` likelihood with the response:

| Network | cover 68% | cover 95% | var $z$ | mean $z$ | bias (km/s/Mpc) | empirical / model $\rho\sigma_B$ |
|---|---|---|---|---|---|---|
| ET-$\Delta$ | 0.678(4) | 0.952(2) | 0.999(13) | $-0.01(2)$ | $+0.002(6)$ | 1.016(23) |
| ET-$\Delta$ + CE | 0.674(5) | 0.951(2) | 1.020(18) | $-0.02(2)$ | $-0.001(7)$ | 1.034(23) |
| ET-2L-par | 0.681(8) | 0.952(4) | 1.003(29) | $-0.02(3)$ | $-0.004(6)$ | 1.010(23) |
| ET-2L-par + CE | 0.676(9) | 0.947(4) | 1.036(33) | $-0.02(3)$ | $-0.004(6)$ | 1.025(23) |
| ET-2L-mis | 0.683(3) | 0.952(2) | 0.987(11) | $+0.05(1)$ | $+0.158(19)$ | 0.998(22) |
| ET-2L-mis + CE | 0.667(5) | 0.948(2) | 1.033(19) | $-0.00(2)$ | $+0.007(11)$ | 1.043(24) |

- The intervals cover at their nominal rates, to within 1.6 percentage points
  at 68% and 0.5 at 95%.
- Without the response the same likelihood under-covers: 68% coverage
  0.666, 0.634, 0.627, 0.607, 0.660, 0.582 with the sky-averaged $\sigma_B$
  (`amplitude_sky`), and 0.42 to 0.66 with detector noise only.
- No bias in the likelihood. The ET-2L-mis posterior mean sits $+0.16$ high (and
  appears identically with detector noise only): a likelihood Gaussian in
  $1/H_0$ under a uniform $H_0$ prior has a posterior mean shifted by
  $\simeq 4vH_0 = 0.15$, which a toy with no shot noise reproduces
  ($+0.141\pm0.011$). It is below 0.01 for the other networks, and the PIT,
  hence the coverage, is unaffected.
- The injections scatter 1-4% more than the model $\sigma_B$ for the CE
  networks, each within $2\sigma$ and all of one sign; ET-2L-mis + CE's
  68% coverage is 0.667. The response second moment is heavy-tailed for
  these networks: $\sigma_B$ moves by $\pm1.4\%$ between seeds at 8192 draws and
  is still converging at $2.6\times10^5$ (4.82, 4.86, 4.88$\times10^{-3}$ for
  ET-2L-mis + CE at 32768, 65536, 262144 draws, against 5.03 measured). It is the
  1-3% of the Hadamard covariance check.

## Shot-noise figure

`outputs/figures/shot_noise_coverage/shot_noise_density_spectrum.pdf`, now
from 1000 realizations of the network simulator, for ET-2L-par + CE.

- **Left.** $\epsilon_k$ no longer has one width: sd($\epsilon_k$) = 0.0026,
  0.0029, 0.0033, 0.0028 and 0.0044 at 5, 10, 20, 50 and 100 Hz (was
  $\simeq\sigma_B = 2.2\times10^{-3}$ in every inspiral bin). The response
  decorrelates the bins: 0.19 at 500 Hz, where the bins carry $2\times10^{-10}$
  of $\rho^2$.
- **Right.** Leading eigenvalues of the whitened catalog covariance,
  analytic (Hadamard) against the 1000 realizations: 2.28 (2.41), 0.186
  (0.189), 0.045 (0.043), 0.012 (0.013), 0.0023 (0.0023). The common amplitude
  is still the leading mode, $\lambda_1 \simeq \zeta^2$, but the perpendicular
  modes are no longer negligible: $\lambda_2 \simeq 0.19$ of the detector noise
  (was $1.4\times10^{-5}$).
- **Common-shape accuracy** (`shot_noise_common_shape.csv`). The perpendicular
  shot noise the rank-one likelihood drops, $\lambda_{\max}(\Delta)$, is
  0.0012, 0.16, 0.065, 0.30, 0.069 and 0.51 of the detector noise, and the KL
  divergence of the common-shape Gaussian from the full one is
  $5\times10^{-7}$, 0.008, 0.0006, 0.013, 0.0012 and 0.067 nats (was below
  $2.3\times10^{-6}$ for every network). The posterior variance of the shape
  parameters along the perpendicular modes grows by up to 4% (17% for ET-2L-mis
  + CE). For $H_0$ the coverage above shows the rank-one likelihood stays
  calibrated, but the paper's "at least five orders of magnitude below the
  detector noise" and "exact to the precision of this work" no longer hold:
  the `"modes"` likelihood keeps these modes.

## $\Xi_0$ and $n$

| Network | $\sigma_{\Xi_0}$ old | new | new/old | $\sigma_{\Xi_0}/(\rho^{-2}+\sigma_B^2)^{1/2}$ new | $\sigma_n$ new |
|---|---|---|---|---|---|
| ET-$\Delta$ | 0.00591 | 0.00616 | 1.043 | 1.048 | 0.737 |
| ET-$\Delta$ + CE | 0.00490 | 0.00572 | 1.167 | 1.118 | 0.737 |
| ET-2L-par | 0.00308 | 0.00357 | 1.158 | 1.088 | 0.737 |
| ET-2L-par + CE | 0.00298 | 0.00365 | 1.223 | 1.114 | 0.737 |
| ET-2L-mis | 0.02312 | 0.02481 | 1.073 | 1.047 | 0.738 |
| ET-2L-mis + CE | 0.00674 | 0.00910 | 1.349 | 1.166 | 0.737 |

- New is the `"modes"` likelihood, as for $H_0$. $\sigma_{\Xi_0}$ grows by 4% to
  35%.
- It exceeds $(\rho^{-2}+\sigma_B^2)^{1/2}$ by 5% to 17% (was within 5.5%). The
  $\Xi_0$ marginal integrates over $n$, whose degeneracy gives it elongated
  tails whose extent depends on the realization each network recorded.
- $n$ is still unconstrained: $\sigma_n = 0.737$ for every network, set by the
  prior.

## Minimum redshift

$\sqrt{v(x)}/x = (\rho^{-2}+\sigma_B^2)^{1/2}$ against $z_{\min}$ at the
fiducials, with $\sigma_B$ now carrying the response
(`outputs/figures/ensemble_spectrum/minimum_redshift_scan.{pdf,csv}`; the old
curves are `minimum_redshift_scan_sky.csv`).

| Network | best $z_{\min}$ old | new | gain over $z_{\min}=0.35$ old | new | $\sqrt{v}/x$ at 0.35 (%) new |
|---|---|---|---|---|---|
| ET-$\Delta$ | 0.18 | 0.21 | 7.0% | 5.0% | 0.585 |
| ET-$\Delta$ + CE | 0.23 | 0.28 | 4.2% | 1.3% | 0.509 |
| ET-2L-par | 0.38 | 0.42 | 0.1% | 0.9% | 0.327 |
| ET-2L-par + CE | 0.40 | 0.44 | 0.4% | 1.7% | 0.327 |
| ET-2L-mis | 0.05 | 0.16 | 21% | 9.3% | 2.36 |
| ET-2L-mis + CE | 0.16 | 0.30 | 8.9% | 0.6% | 0.777 |

The larger $\sigma_B$ moves every minimum to higher $z_{\min}$. For the
ET-2L-par networks the minimum is at $z_{\min} \simeq 0.42$-$0.44$, and 0.35 is
within 2% of it. For ET-$\Delta$, ET-$\Delta$ + CE and ET-2L-mis + CE the minimum is
at $0.21$-$0.30$, and choosing it would gain at most 5%. For ET-2L-mis the minimum
moves from 0.05 to 0.16 and would gain 9% (was 21%).

## Madau-Dickinson parameters

ET-2L-par + CE, median and 68% HDI. The `"modes"` likelihood (now adopted for
the shape parameters) and the paper's `"fixed"` one on the network injection,
against the paper's sky-averaged run:

| | $z_p$ | $\gamma$ | $\kappa$ | corr($z_p,\gamma$) | corr($z_p,\kappa$) | corr($\gamma,\kappa$) |
|---|---|---|---|---|---|---|
| old (sky, fixed) | 1.34 (0.90-2.50) | 1.83 (1.10-2.52) | 4.72 (1.07-7.06) | $-0.82$ | 0.48 | $-0.37$ |
| new (network, fixed) | 1.24 (0.24-1.76) | 1.96 (1.10-2.77) | 5.18 (1.85-7.88) | $-0.82$ | 0.45 | $-0.34$ |
| new (network, modes) | 1.25 (0.23-1.76) | 1.94 (1.10-2.72) | 5.15 (1.37-7.38) | $-0.82$ | 0.44 | $-0.33$ |

- The change comes from the data, not from the extra modes: `fixed` and
  `modes` on the same network injection differ by 0.01-0.03 in the medians.
- The $z_p$ marginal is broad and its 68% HDI depends on the realization: it
  no longer reaches the upper edge of the prior. "The interval of $z_p$ reaches
  the upper edge of its prior" no longer holds for this injection.
- $\gamma$ is still bounded from below at $\simeq 1.1$, $\kappa$ still flat over
  most of its prior, and the correlations barely move.
- The intervals cover 61% ($z_p$), 16% ($\gamma$) and 60% ($\kappa$) of the
  prior ranges (were 64%, 14%, 60%).

## Not regenerated

These paper numbers still assume the sky-averaged $\sigma_B$: the skewness
$g_B$ and $g$ of the Gaussian-truncation paragraph, the exact-density
appendix and its figure (`shot_noise_exact_vs_gaussian_H0`), and the
statement $\zeta \simeq 1.2$ quoted there.

## How to regenerate

```bash
uv run --extra notebook python notebooks/cosmology_grid_posteriors.py
uv run --extra notebook python notebooks/madau_dickinson_grid.py
uv run --extra notebook python notebooks/shot_noise_coverage.py
```

The minimum-redshift scan is section 9 of `notebooks/ensemble_spectrum.py`.
The network injections are the one ensemble `outputs/ensembles/fiducial/`
(seed 41: 1000 draws for the coverage, draw `41-0` for the grids), written by
`workflow/coverage_draws.smk`; each draw's background holds every network's
spectra from one waveform pass, about 45 s per paper-size catalog on an M4. Every likelihood holds
`Network.response_moments`, so the response costs nothing per grid point.
