# Alternative likelihoods for physical shot noise

This note collects the likelihood constructions discussed for modelling the
*physical* shot noise of the catalog-based forward model, and how each relates
to the offset statistic $\mathrm{sd}(r)$ of `notebooks/spectrum_snrs.py`.
Nothing here is implemented; it is a design record and a validation plan.

## 1. Setting and notation

The spectrum is sampled on $F$ frequency bins $f_i$ ($F = 403$ in the notebook).
A single merger $x$ contributes a power vector $P(f_i; x) \in \mathbb{R}^F$,
$P \ge 0$.

**Data.** Over an observing time $T$ the real universe produces
$N \sim \mathrm{Poisson}(\lambda)$ events, $\lambda = R(\theta)\,T$, with
$R$ the total merger rate and $x_k \overset{\text{iid}}{\sim} p(x \mid \theta)$:

$$
S^{\mathrm{obs}}_i \;=\; \frac{1}{T}\sum_{k=1}^{N} P(f_i; x_k).
$$

**Template (what the code does today).** A bank $\{x_j\}_{j=1}^{M}$ drawn from a
proposal $q$ is reweighted to the target. With density ratio
$\rho_j = p(x_j \mid \theta)/q(x_j)$ and the power rescaled to the target
distance, $P_j(\theta) = \kappa_j(\theta)\,P_j^{\mathrm{ref}}$,
$\kappa_j = (d_j^{\mathrm{ref}}/d_L(z_j \mid \theta))^2$:

$$
\hat S_i(\theta) \;=\; R(\theta)\,\frac{1}{M}\sum_{j=1}^{M} \rho_j\,P_j(f_i;\theta).
$$

This is `spectral_density` in `gwb/spectral.py`, with the code's weight
$w_j = \rho_j\kappa_j$ acting on $P^{\mathrm{ref}}$. It is an unbiased estimate
of the **expectation**

$$
\mu_i(\theta) \;=\; R(\theta)\,m_i(\theta),\qquad
m_i(\theta) = \mathbb{E}_p\!\left[P(f_i)\right].
$$

Any expectation under the target is estimated by the same recipe,
$\mathbb{E}_p[g] \approx M^{-1}\sum_j \rho_j\, g(P_j(\theta))$.

**Two different noises.**

| | Source | Reducible by | Where it lives |
|---|---|---|---|
| (a) Monte Carlo error | finite bank $M$ estimating $\mu$ | larger $M$, better $q$ | template |
| (b) Physical shot noise | finite number of real mergers | nothing | data |

The current model has no term for (b): it compares $S^{\mathrm{obs}}$ to $\hat S$
with detector noise only. For $N \approx RT$ events drawn from $p$ itself, a
fixed-count draw is already a physical realization; the Poisson count
contributes only a relative $1/\sqrt N \sim 10^{-3}$. So the Poisson point of the
notebook ($\mathrm{sd}(r)\approx 0.5$) is, to leading order, irreducible physical
noise and not template error.

## 2. Why not sample $N$ and use a bank subset

* $N$ depends on $\theta$ through $R(\theta)$ and is an integer. A sum over the
  first $N(\theta)$ bank events has zero gradient through the cutoff and jumps by
  one event's contribution whenever $N$ crosses an integer.
* Resampling the subset every sampler step makes the target stochastic: the
  Hamiltonian changes between leapfrog steps, energy is not conserved and the
  acceptance rate drops (pseudo-marginal HMC).
* A subset *fixed* for the whole chain is smooth, but it is one particular noisy
  template, not a model of the data's noise. It also discards bank information
  ($\approx 30\%$ for $N = 7\times 10^5$ from $M = 10^6$), which raises (a).
* Marginalizing $N$ exactly needs the $N$-fold convolution of an
  $F$-dimensional density. It is intractable, and anyway immaterial: the count
  contributes $\sim 0.1\%$ of the noise. What matters is *which* events occurred.

The constructions below marginalize $N$ and the event draw analytically (or
nearly so) instead.

## 3. Moments of the compound Poisson sum

By the law of total variance, with $\lambda = RT$,

$$
\mathbb{E}\!\left[S_i\right] = \mu_i,
\qquad
\mathrm{Var}\!\left[\sum_{k\le N} P_{k,i}\right]
= \lambda\,\mathrm{Var}(P_i) + \lambda\,\mathbb{E}[P_i]^2
= \lambda\,\mathbb{E}\!\left[P_i^2\right].
$$

Hence

$$
\boxed{\;
\Sigma^{\mathrm{shot}}_{ij}(\theta)
\;=\; \mathrm{Cov}\!\left[S_i, S_j\right]
\;=\; \frac{R(\theta)}{T}\,\mathbb{E}_p\!\left[P_i P_j\right]
\;}
$$

The second moment, not the covariance, appears: the Poisson count supplies the
$m_i m_j$ term. Estimate it from the bank:

$$
\hat\Sigma^{\mathrm{shot}}_{ij}
= \frac{R(\theta)}{T}\,\frac{1}{M}\sum_{l=1}^{M}\rho_{l}\,P_{l}(f_i;\theta)\,P_{l}(f_j;\theta).
$$

All higher **cumulants** are raw moments of one event (a defining property of
compound Poisson): for a scalar per-event contribution $u$,

$$
\kappa_n \;=\; \lambda\,\mathbb{E}_p\!\left[(u/T)^n\right].
$$

**Tail behaviour.** Near sources dominate: $P \propto d^{-2}$ and counts scale as
$d^{2}\,\mathrm{d}d$, so $\mathbb{E}[P^n] \propto \int d^{2-2n}\,\mathrm{d}d
\propto d_{\min}^{\,3-2n}$ for $n\ge 2$. The variance grows as $1/z_{\min}$, the
third cumulant as $z_{\min}^{-3}$, the fourth as $z_{\min}^{-5}$. The single-event
distribution has tail index $3/2$ (mean finite, variance infinite without a
cutoff); the cutoff makes everything finite but large.

**Cost.** $\hat\Sigma^{\mathrm{shot}}$ costs $O(MF^2)$ per evaluation if the
weights change with $\theta$. The correlation matrices in the notebook show the
shot noise is close to rank one, so project onto $K\!\sim\!2$–$5$ modes,
$P_l \approx \Phi\,a_l$ with $\Phi\in\mathbb{R}^{F\times K}$ (PCA of the bank):

$$
\mathbb{E}_p[PP^\top] \approx \Phi\,\mathbb{E}_p[aa^\top]\,\Phi^\top,
$$

which costs $O(MK^2)$ and keeps $\hat\Sigma$ low rank plus the detector diagonal
(Woodbury inversion).

## 4. Route A: Gaussian likelihood with $\theta$-dependent covariance

Treat $S^{\mathrm{obs}}$ as Gaussian with mean $\mu(\theta)$ and covariance

$$
\Sigma(\theta) \;=\; \Sigma^{\mathrm{det}} + \Sigma^{\mathrm{shot}}(\theta) + \Sigma^{\mathrm{MC}},
\qquad
\Sigma^{\mathrm{det}} = \mathrm{diag}(\sigma_i^2),
$$

with $\sigma_i = S_{\mathrm{eff},i}/\sqrt{2T\Delta f_i}$ as in the notebook. Then

$$
\ln \mathcal L(\theta)
= -\tfrac12\,\big(S^{\mathrm{obs}}-\mu(\theta)\big)^{\!\top}\Sigma(\theta)^{-1}\big(S^{\mathrm{obs}}-\mu(\theta)\big)
-\tfrac12\ln\det\Sigma(\theta).
$$

The log-determinant is essential: $\Sigma^{\mathrm{shot}}\propto R(\theta)$ depends
on the parameters, so dropping it lets the fit inflate the variance for free and biases the posterior towards large $R$.

$\Sigma^{\mathrm{MC}}$ is the uncertainty of $\hat\mu$ itself (error type (a)):

$$
\Sigma^{\mathrm{MC}}_{ij} \;\approx\; \frac{R^2}{M}\,\mathrm{Cov}_q\!\big(\rho P_i,\;\rho P_j\big).
$$

Estimate $\Sigma^{\mathrm{shot}}$ from a bank independent of the one that gives
$\hat\mu$; sharing events makes the noise partly cancel, which is the shared-draw
caveat already noted in the notebook.

**Link to the notebook's statistic.** Take the amplitude-only template
$\varphi = \mu$ and write $D = \Sigma^{\mathrm{det}}$. The matched-filter amplitude
$\hat A = \varphi^\top D^{-1}S/\varphi^\top D^{-1}\varphi$ has

$$
\mathrm{Var}_{\mathrm{det}}(\hat A) = \frac{1}{\varphi^\top D^{-1}\varphi},
\qquad
\mathrm{Var}_{\mathrm{shot}}(\hat A)
= \frac{\varphi^\top D^{-1}\Sigma^{\mathrm{shot}}D^{-1}\varphi}{(\varphi^\top D^{-1}\varphi)^2}.
$$

Since $r$ is the $H_0$ offset in units of the fixed detector-limited width,

$$
\mathrm{sd}(r)^2 \;\approx\;
\frac{\mathrm{Var}_{\mathrm{shot}}(\hat A)}{\mathrm{Var}_{\mathrm{det}}(\hat A)}
\;=\;
\frac{\varphi^\top D^{-1}\Sigma^{\mathrm{shot}}D^{-1}\varphi}{\varphi^\top D^{-1}\varphi}.
$$

This is a prediction of $\mathrm{sd}(r)$ for the Poisson ensemble from one bank, to be
compared with the measured value ($\approx 0.48$ at 100 draws).

**Strengths.** Smooth, cheap, NUTS-friendly, handles shape through $\Sigma$.
**Weaknesses.** Gaussian tails could understate rare large offsets at small
$z_{\min}$. At $N\sim7\times10^5$ the measured skewness of the amplitude shift is
consistent with zero (Section 4A), so this may be a minor concern at the baseline cutoff.

## 4A. Route A′: amplitude-only shot-noise likelihood (recommended)

**Observation.** For the network `ET-2L-aligned-CE-Hanford` the SNR is
concentrated at low frequency, where every event is still in its inspiral
power law, so the shot noise is a *common multiplicative factor* on the spectrum.
Using the 100-draw cached ensembles of `spectrum_snrs.py`, the weights
$w_i = \bar S_i^2/\sigma_i^2$ (the per-bin share of $\rho^2$), the relative residuals
$r_{i}=S_i/\bar S_i-1$ and $\varepsilon = \sum_i w_i r_i/\sum_i w_i$ (the
matched-filter amplitude shift):

| quantity | result |
|---|---|
| $\rho^2$ cumulative share below 10 Hz / 31 Hz | 50% / 90% |
| leading noise-weighted mode, relative-residual profile $g(f)$ | $1.00$ up to 50 Hz, $1.03$ at 100 Hz, deviates only above 200 Hz |
| share of weighted shot-noise variance in the amplitude mode $\varepsilon$ | $\approx 100\%$ (leftover shape $<0.1\%$) |
| $\rho\cdot\mathrm{sd}(\varepsilon)$ for $N = 2^{14},2^{16},2^{18}$, Poisson | $2.45,\ 1.22,\ 0.70,\ 0.48$ |
| skewness of $\varepsilon$ ($N=2^{18}$, Poisson), 100 draws | $-0.01$, $0.2$ (standard error $\approx 0.24$) |

The fourth row reproduces the $\mathrm{sd}(r)$ values of Figure A2, as it must:
$\mathrm{sd}(r)=\rho\,\mathrm{sd}(\varepsilon)$.

**Model.** The amplitude-marginalized model already splits the likelihood as

$$
-\tfrac12\sum_i\Big(\frac{d_i-A\,m_i}{\sigma_i}\Big)^2
= -\mathcal R - \tfrac12\,\rho^2\big(A-\hat A\big)^2,
$$

with $\mathcal R$ the best-fit residual (independent of $A$), $\hat A$ the
amplitude MLE and $\rho$ the template SNR. Only the second term involves
shot noise, because the shape directions orthogonal to the template $m$ carry
essentially none. Replace it by

$$
\hat A \mid A \;\sim\; \mathcal N\!\big(A,\;V(A)\big),
\qquad
V(A) \;=\; \underbrace{\rho^{-2}}_{\text{detector}}
\;+\; \underbrace{A^2\,s^2(A)}_{\text{shot noise}},
$$

where $s^2$ is the relative variance of the matched-filter amplitude:

$$
s^2 \;=\; \frac{\mathrm{Var}(\textstyle\sum_k u_k)}{\big(\mathbb{E}\sum_k u_k\big)^2}
\;=\; \frac{\mathbb{E}_p[u^2]}{\lambda\,\mathbb{E}_p[u]^2},
\qquad
u(x) = m^\top D^{-1}P(x),\quad \lambda = R\,T .
$$

Both moments are weighted bank sums, $\hat{\mathbb{E}}_p[u^n]=M^{-1}\sum_j\rho_j u_j^n$
($\rho_j$ is the density ratio of Section 1). If $u_j$ is evaluated once at the
fiducial template it is a single array of $M$ scalars, so the extra cost per
sampler step is $O(M)$.

**Dependence on the marginalized parameter.** The relative variance is inversely
proportional to the event count, $s^2\propto 1/\lambda$, and $\lambda$ scales with the
merger-rate part of the amplitude. With $g_R(\varphi)$ the existing
`MergerRateAmplitudeFn`,

$$
s^2(\varphi) = s^2_{\mathrm{fid}}\;\frac{g_R(\varphi_{\mathrm{fid}})}{g_R(\varphi)},
\qquad
s^2_{\mathrm{fid}} = \frac{\mathrm{sd}(r)^2}{\rho^2}.
$$

For $H_0$ with $g_R\propto H_0^{-3}$ and $A\propto H_0^{-1}$ this gives
$V_{\mathrm{shot}} = s_{\mathrm{fid}}^2\,H_0/H_{0,\mathrm{fid}}$ (to be confirmed against the
conventions of `amplitude_H0_fn`). Sampled rate parameters enter through $\lambda$ in the same way.

**Where it plugs in.** `AmplitudeConditional` already integrates the 1D log-integrand
$\ln\pi(\varphi) - \tfrac12\rho^2(A(\varphi)-\hat A)^2$ on a trapezoid grid, so
a variance that depends on $\varphi$ is a local change:

$$
\ell(\varphi)
= \ln\pi(\varphi)
- \frac{\big(\hat A - A(\varphi)\big)^2}{2\,V(\varphi)}
- \tfrac12\ln V(\varphi).
$$

The $\ln V$ term is new and necessary because $V$ now varies with $\varphi$.
Nothing else changes: the shape parameters see the unchanged residual term $\mathcal R$.

**Skewness and tails.** A Gaussian is adequate at $N\sim7\times10^5$ (measured
skewness consistent with zero). If it is not at a smaller $z_{\min}$, the Edgeworth term of
Section 6 or the exact Fourier form of Section 5 can replace the Gaussian
in $\ell(\varphi)$ without altering the rest of the model, because only $\hat A\mid A$ is affected.

**Limits of the argument.**

* The flat-mode result is for the mean spectrum as template; the real analysis varies
  the shape parameters, so recheck $g(f)$ at the posterior's extremes.
* It depends on the network's SNR being concentrated below about 30 Hz. Another
  network or band changes $g(f)$ and may break the amplitude-only picture. Evaluate the
  table above for each network before relying on it.
* $s_{\mathrm{fid}}$ comes from the bank, so it carries the heavy-tailed noise of
  second moments: bootstrap it and use a bank independent of the one that
  gives $\hat\mu$.
* The table uses the 100-draw ensembles; the 200-draw sweep had not been run when
  it was computed.

## 5. Route B: exact scalar likelihood by Fourier inversion

The shot noise is nearly rank one, so project the per-event power onto the
matched filter, $u(x) = \varphi^\top D^{-1}P(x)$, and let
$Y = T^{-1}\sum_{k\le N} u(x_k)$. A compound Poisson variable has the
characteristic function

$$
\Psi_Y(t) \;=\; \mathbb{E}\!\left[e^{itY}\right]
\;=\; \exp\!\Big\{\lambda\big(\psi(t)-1\big)\Big\},
\qquad
\psi(t) = \mathbb{E}_p\!\left[e^{\,i t\,u/T}\right],
$$

which marginalizes $N$ **exactly**. Estimate $\psi$ from the bank:

$$
\hat\psi(t) = \frac{1}{M}\sum_{j=1}^{M}\rho_j\,e^{\,i t\,u_j(\theta)/T}.
$$

(bin the weighted $u_j$ on a grid and use an FFT). Detector noise is Gaussian and
independent, so it multiplies by its own characteristic function:

$$
p\big(y^{\mathrm{obs}}\mid\theta\big)
= \frac{1}{2\pi}\int_{-\infty}^{\infty}
e^{-ity^{\mathrm{obs}}}\;
\exp\!\Big\{R(\theta)T\big(\hat\psi(t)-1\big)\Big\}\;
e^{-\tfrac12\sigma_{\mathrm{det}}^2 t^2}\,\mathrm{d}t .
$$

The rate $R(\theta)$ enters only in the exponent, so the likelihood is smooth in
$\theta$ and differentiable through the FFT.

**Strengths.** Exact in $N$ and in the single-event distribution (for the scalar
amplitude), capturing skewness and heavy tails.
**Weaknesses.**

* Only the scalar projection: shape variation, which decorrelates above roughly
  500 Hz, is lost. Extend to $K = 2$–$3$ modes with a $K$-dimensional FFT.
* $\psi$ is non-analytic at $t=0$ when the cutoff is small
  ($\sim|t|^{3/2}$ behaviour), so the $t$-grid must resolve $t\to0$ and extend far
  enough to cover the largest events.
* $\hat\psi$ at small $|t|$ is dominated by the largest bank events, so it is as
  noisy as the tail.
* The projection vector $\varphi$ depends on $\theta$; fix it at the fiducial or
  update it each step.

## 6. Route C: Edgeworth correction (saddlepoint is awkward)

An Edgeworth expansion of the scalar law about its Gaussian limit, with
$z = (y-\kappa_1)/\sqrt{\kappa_2}$ and $\gamma_n=\kappa_n/\kappa_2^{n/2}$,

$$
p(y) \approx \frac{\phi(z)}{\sqrt{\kappa_2}}
\left[1 + \frac{\gamma_3}{6}\,\mathrm{He}_3(z)
+ \frac{\gamma_4}{24}\,\mathrm{He}_4(z)
+ \frac{\gamma_3^2}{72}\,\mathrm{He}_6(z)\right],
\qquad
\gamma_3 = \frac{\mathbb{E}[u^3]}{\sqrt{\lambda}\,\mathbb{E}[u^2]^{3/2}}\ \propto\ \lambda^{-1/2}.
$$

The skewness shrinks as the rate grows, but the moments $\mathbb{E}[u^n]$ scale as
$z_{\min}^{3-2n}$, so the series is asymptotic and can go negative in the tails.

A saddlepoint approximation needs the cumulant generating function
$K(s)=\lambda(M_u(s)-1)$ with a real argument. The right tail of $u\ge0$ is a
power law (before the cutoff), so $M_u(s)$ does not exist for $s>0$, and with a cutoff
it exists but is enormous and ill-conditioned. A Laplace-transform inversion
($s\le0$) or Route B's Fourier inversion is the better choice. A
saddlepoint approximation is therefore not a good fit here.

## 7. Route D: latent noise vector (non-centred)

Introduce explicit noise variables and let NUTS sample them:

$$
S^{\mathrm{obs}} = \mu(\theta) + L(\theta)\,\epsilon + \eta_{\mathrm{det}},
\qquad
\Sigma^{\mathrm{shot}} \approx L L^\top,\quad L\in\mathbb{R}^{F\times K},
\quad \epsilon\in\mathbb{R}^K .
$$

With $\epsilon\sim\mathcal N(0,I_K)$ and $\epsilon$ integrated out this reproduces
Route A. The advantage is that the prior on $\epsilon$ can be changed to something
heavy-tailed or skewed, for instance the empirical scalar-amplitude law from
Route B or a fitted skew/stable family:

$$
S^{\mathrm{obs}} = \mu(\theta) + \eta\,\varphi + \eta_{\mathrm{det}},
\qquad \eta \sim p_{\mathrm{shot}}(\,\cdot\mid\theta).
$$

The cost is extra latent dimensions in the sampler and a funnel when
$\Sigma^{\mathrm{shot}}\to0$. Treat it as a fallback if Routes A and B disagree.

## 8. Reducing the template error (a)

None of the above lowers (a). That needs a larger $M$ or a proposal tailored to
the integrand. The variance of the weighted term $\rho P$ is dominated by near
sources, and the zero-variance proposal at $\theta_{\mathrm{fid}}$ is

$$
q^\star(x) \;\propto\; p(x\mid\theta_{\mathrm{fid}})\,P(x),
$$

for which $\rho P$ is constant. A tilt towards low redshift does this
approximately. It is optimal only at $\theta_{\mathrm{fid}}$; watch
`importance_relative_ess` as $\theta$ moves, and keep a mixture component for
robustness. Estimating the second moment $\mathbb{E}_p[PP^\top]$ needs $\rho P^2$,
which is still heavy-tailed under this $q$, so check its ESS separately.

## 9. Validation plan

Use the spectrum cache of `spectrum_snrs.py` (200 draws per ensemble):

1. **Variance check.** Predict $\mathrm{sd}(r)$ from the formula in section 4 using
   one bank and compare it with the Poisson ensemble's measured value and its
   bootstrap error.
2. **Shape check.** Compare the empirical distribution of $r$ (skew, quantiles
   such as $q_{95}(|r|)$) with the Route B density at the same $\theta$. A
   mismatch means the Gaussian is inadequate.
3. **Calibration.** Simulate observed spectra from Poisson subsets of an
   *independent* bank, run inference with each likelihood, and compare the $H_0$
   coverage and rank statistics (simulation-based calibration) against the
   present detector-only likelihood.
4. **Cost.** Time a full gradient evaluation for $M = 2^{17}$ and $10^6$ with the
   low-rank $\Sigma^{\mathrm{shot}}$.

## 10. Summary

| Route | Marginalizes $N$ | Tails | Shape | Cost per step | Notes |
|---|---|---|---|---|---|
| A′: amplitude-only, $V(A)$ | at 2nd order | Gaussian (skew $\approx0$ at $N\sim10^6$) | amplitude only; shape unaffected | $O(M)$ | recommended; plugs into `AmplitudeConditional` |
| A: Gaussian, $\Sigma(\theta)$ | at 2nd order | Gaussian (too light) | full via $\Sigma$ | $O(MK^2)+F^3$ or low rank | start here; predicts $\mathrm{sd}(r)$ |
| B: scalar Fourier | exactly | exact | scalar only | $O(M + G\log G)$ | needs $t$-grid care |
| C: Edgeworth | via cumulants | asymptotic, can fail | scalar only | cheap | series unreliable at small $z_{\min}$ |
| D: latent $\epsilon$ | depends on prior | chosen by prior | rank $K$ | sampler dimension $+K$ | fallback |

**Recommended order:** implement A′ (Section 4A), check it against the Poisson
ensemble and by calibration (items 1 and 3 of Section 9), and move to A or B only
if the shape parameters or the tails turn out to matter for $H_0$ coverage.

## 11. Caveats

* The cross-bin covariance and the per-bin detector noise are estimated from the
  same population model; a mismatch between data and template population then enters
  twice.
* $\Sigma^{\mathrm{shot}}$ estimates from the bank are themselves heavy-tailed
  (they are second moments). Quote bootstrap errors on them.
* The scalar formulas assume the amplitude-only picture of the notebook. In the
  full multi-parameter inference a coherent amplitude error can be absorbed by a
  rate or normalization parameter, so the real effect on $H_0$ may be smaller than
  $\mathrm{sd}(r)$ suggests; shape errors project through the parameter degeneracies
  instead.
