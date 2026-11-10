# The shot-noise budget of the amplitude estimator

This note derives the variance budget in Table `tab:shot-noise-budget` of the
paper, which `scripts/shot_noise_budget.py` writes. The function that computes it
is `astrogwb.inference.amplitude_variance_budget`.

## Setup

The data $\mathbf d$ are the stacked per-pair cross powers of one network on the
entries it counts (`inference/gaussian.py`). Their mean is
$\boldsymbol\mu_{ab,k} = \gamma_{ab}(f_k) S(f_k)$, and the amplitude $x$ rescales
it. With the total covariance $\Sigma$ held at the fiducials, the paper's estimator
(§4.2, eq. `amplitude-relative-uncertainty`) is the generalized least-squares one:

$$
\hat x = \mathbf w^T \mathbf d, \qquad
\mathbf w = \frac{\Sigma^{-1}\boldsymbol\mu}{\boldsymbol\mu^T\Sigma^{-1}\boldsymbol\mu},
\qquad
\frac{{\rm Var}[\hat x]}{x^2} = \mathbf w^T\Sigma\mathbf w
  = \frac{1}{\boldsymbol\mu^T\Sigma^{-1}\boldsymbol\mu}.
$$

The covariance is the sum of five terms (eqs. `connected-covariance-decomposition`
and `connected-covariance-intensity-split`):

$$
\Sigma = \Sigma^g + \Sigma^{\rm count} + \Sigma^I + \Sigma^{\rm lin} + \Sigma^{\rm circ}.
$$

## The five terms as uncorrelated noise components

A catalog of $N_{\rm obs}\sim{\rm Poisson}(N)$ sources contributes
$\sum_i \mathbf y_i$ to the data. Source $i$ contributes
$\mathbf y_i = P_i\,\mathbf r_i$, its power spectrum $P_i(f_k)$ times its per-pair
response $r_{i,ab}(f_k)$. The two factors are independent. The response is linear
in the source's Stokes vector, so
$\mathbf y = \mathbf y^I + \mathbf y^{\rm lin} + \mathbf y^{\rm circ}$. Under the
isotropic average, $\psi \to \psi + \pi/4$ flips the sign of the linear response
and leaves the other two unchanged, and $\cos\iota \to -\cos\iota$ flips the sign
of the circular response and leaves the other two unchanged. So
$\langle\mathbf y^{\rm lin}\rangle = \langle\mathbf y^{\rm circ}\rangle = 0$, the
three Stokes parts are mutually uncorrelated, and
$\langle\mathbf y^I\rangle = \boldsymbol\mu/N$. Write the catalog fluctuation as
the sum of four components:

| component | definition | covariance |
|---|---|---|
| $\mathbf n^{\rm count}$ | $(N_{\rm obs} - N)\,\boldsymbol\mu/N$ | $\Sigma^{\rm count} = \boldsymbol\mu\boldsymbol\mu^T/N$ |
| $\mathbf n^I$ | $\sum_i (\mathbf y^I_i - \boldsymbol\mu/N)$ | $\Sigma^I = N\langle\mathbf y^I\mathbf y^{I\,T}\rangle - \boldsymbol\mu\boldsymbol\mu^T/N$ |
| $\mathbf n^{\rm lin}$ | $\sum_i \mathbf y^{\rm lin}_i$ | $\Sigma^{\rm lin} = N\langle\mathbf y^{\rm lin}\mathbf y^{{\rm lin}\,T}\rangle$ |
| $\mathbf n^{\rm circ}$ | $\sum_i \mathbf y^{\rm circ}_i$ | $\Sigma^{\rm circ} = N\langle\mathbf y^{\rm circ}\mathbf y^{{\rm circ}\,T}\rangle$ |

Detector noise $\mathbf n^g$, with covariance $\Sigma^g$, is the fifth component.
The five components are pairwise uncorrelated:

- **Count and $I$.** Given $N_{\rm obs}$, the sum defining $\mathbf n^I$ has zero
  mean, so ${\rm Cov}[\mathbf n^{\rm count}, \mathbf n^I] = 0$.
- **Different Stokes terms.** For two of $I$, lin and circ,
  $N\langle\mathbf y^X\mathbf y^{Y\,T}\rangle = 0$ by the two sign flips above.
- **Detector noise.** It is independent of the catalog.

Each term is a covariance, so each is positive semidefinite. In the code,
$N\langle\mathbf y^X\mathbf y^{X\,T}\rangle$ is `catalog_covariance` with the
response moment of Stokes term $X$: the power moment $M$ times the response
moment $R^X$, element by element. Using the code's conventions
(`inference/model.py`), $\boldsymbol\mu\boldsymbol\mu^T/N$ is the paper's
$\Sigma^{c,{\rm count}} = S_h S_h'/(\mathcal R\tau)$ multiplied by
$\gamma_{ab}\gamma_{cd}$, with $N = \mathcal R T$ the expected count. This holds
because the code's data are $\gamma_{ab}S$ rather than $S$.

## The budget

The estimator's error is the sum of the five components, each projected on the
same weights:

$$
\hat x - x = \sum_X \mathbf w^T\mathbf n^X
\quad\Longrightarrow\quad
{\rm Var}[\hat x] = \sum_X \mathbf w^T\Sigma^X\mathbf w ,
$$

because the components are uncorrelated. The table reports
$\mathbf w^T\Sigma^X\mathbf w / {\rm Var}[\hat x]$:

- the shares add to 100% exactly, because $\Sigma$ is linear in its terms, with no
  approximation and no ordering;
- each share is non-negative, because each $\Sigma^X$ is positive semidefinite;
- each share is the variance that one physical noise source contributes to the
  estimator that is actually used.

On the fiducial data the shares close to $10^{-15}$. The smallest eigenvalue of
$\Sigma^I$ is $-10^{-17}$ of its largest, which is zero to round-off.

### The same shares as derivatives

Scale each term, $\Sigma(\boldsymbol\epsilon) = \sum_X \epsilon_X\Sigma^X$, and let
$V(\boldsymbol\epsilon) = (\boldsymbol\mu^T\Sigma(\boldsymbol\epsilon)^{-1}\boldsymbol\mu)^{-1}$.
Then

$$
\frac{\partial V}{\partial\epsilon_X}
  = V^2\,\boldsymbol\mu^T\Sigma^{-1}\Sigma^X\Sigma^{-1}\boldsymbol\mu
  = \mathbf w^T\Sigma^X\mathbf w .
$$

$V$ is homogeneous of degree one in $\boldsymbol\epsilon$, so Euler's theorem gives
$\sum_X \epsilon_X\,\partial_X V = V$. At $\boldsymbol\epsilon = 1$ the projected
shares are therefore also the elasticities $\partial\ln V/\partial\ln\epsilon_X$:
a 10% increase in term $X$ raises the variance by (share of $X$) times 10%. The
weights $\mathbf w$ minimize $\mathbf w^T\Sigma\mathbf w$, so the envelope theorem
lets them be held fixed when differentiating. The derivative and projection
definitions agree to $10^{-16}$ (`jax.grad` against
`amplitude_variance_budget`).

### Attributions not used

**Sequential increments.** Switch the terms on in some order and give each the
increase in $V$ it causes. $V(\Sigma)$ is the minimum of the linear functions
$\mathbf w^T\Sigma\mathbf w$ over unbiased $\mathbf w$, so it is concave in $\Sigma$:
a term's increment shrinks the later it is switched on, and the shares depend on
the order. Subsets without $\Sigma^g$ can be singular. Averaging over orders with
$\Sigma^g$ always first (a Shapley value) removes the ordering but answers a
different question: how much variance a term adds to an estimator re-optimized
without it. With $\Sigma^g$ first, the detector share is $\rho^{-2}/V$, the
template's detector variance.

**Template weights.** The earlier version of the script projected every term on
the detector-noise template $\mathbf c/\rho^2$ with $\mathbf c = (\Sigma^g)^{-1}\boldsymbol\mu$.
That gives an exact budget of a different estimator, whose variance is
$\rho^{-2} + \sigma_{\rm shot}^2$, with
$\sigma^2_{\rm shot} = \mathbf c^T\Sigma^c\mathbf c/\rho^4$. The paper's estimator
does better by the shape correction $\delta$ (`diagnostics.shape_correction`):

$$
{\rm Var}[\hat x] = \rho^{-2} + \sigma_{\rm shot}^2 - \delta/\rho^2 .
$$

**The count term is the same under every attribution.** $\mathbf w^T\boldsymbol\mu = 1$
for any unbiased weights, so $\mathbf w^T\Sigma^{\rm count}\mathbf w = 1/N$. By
Sherman-Morrison, adding $\boldsymbol\mu\boldsymbol\mu^T/N$ to any covariance
raises $V$ by exactly $1/N$. Only the shares of $\Sigma^g$, $\Sigma^I$, lin and
circ depend on the method.

## Numbers

The fiducials are those of draw `fiducial/41-0`, with $N = 7.26\times10^5$
($z \ge 0.35$) and $T = 1$ yr. The response moments come from 8192 isotropic
draws, each with its three parity images. Shares are in percent.

| network | method | $\sigma_x/x$ | $\Sigma^g$ | count | $I$ | lin | circ |
|---|---|---|---|---|---|---|---|
| ET-Δ | projected (adopted) | 0.769% | 88.6 | 2.3 | 9.0 | 0.0 | 0.0 |
| | template | 0.769% | 88.6 | 2.3 | 9.0 | 0.0 | 0.0 |
| ET-Δ + CE | projected | 0.643% | 80.8 | 3.3 | 13.0 | 0.1 | 2.8 |
| | template | 0.646% | 79.4 | 3.3 | 13.4 | 0.1 | 3.8 |
| | sequential, order of the table | | 80.1 | 3.3 | 13.3 | 0.1 | 3.2 |
| | Shapley, $\Sigma^g$ first | | 80.1 | 3.3 | 13.2 | 0.1 | 3.3 |
| ET-2L-par | projected | 0.379% | 50.5 | 9.6 | 39.2 | 0.7 | 0.0 |
| | template | 0.379% | 50.4 | 9.6 | 39.3 | 0.7 | 0.0 |
| ET-2L-par + CE | projected | 0.368% | 46.8 | 10.2 | 41.2 | 0.7 | 1.1 |
| | template | 0.372% | 44.8 | 10.0 | 41.7 | 0.8 | 2.8 |
| | sequential, order of the table | | 45.7 | 10.2 | 42.0 | 0.8 | 1.4 |
| | Shapley, $\Sigma^g$ first | | 45.7 | 10.2 | 41.7 | 0.8 | 1.7 |
| ET-2L-mis | projected | 3.156% | 94.3 | 0.1 | 2.3 | 1.3 | 1.9 |
| | template | 3.156% | 94.3 | 0.1 | 2.3 | 1.3 | 2.0 |
| ET-2L-mis + CE | projected | 0.961% | 78.4 | 1.5 | 10.6 | 0.4 | 9.1 |
| | template | 0.969% | 75.7 | 1.5 | 11.4 | 0.4 | 11.0 |
| | sequential, order of the table | | 77.0 | 1.5 | 11.2 | 0.4 | 9.8 |
| | Shapley, $\Sigma^g$ first | | 77.0 | 1.5 | 11.0 | 0.4 | 10.1 |

The template shares are the template estimator's own budget, normalized to its
own variance $\rho^{-2} + \sigma_{\rm shot}^2$. For each network, the shape
correction $\delta$ and the template's fractional excess variance
$\delta/(1 + \rho^2\sigma^2_{\rm shot} - \delta)$ are:

| network | $\delta$ | excess |
|---|---|---|
| ET-Δ | $6\times10^{-7}$ | $5\times10^{-7}$ |
| ET-Δ + CE | 0.011 | 0.9% |
| ET-2L-par | 0.0011 | 0.06% |
| ET-2L-par + CE | 0.043 | 2.0% |
| ET-2L-mis | 0.0005 | 0.04% |
| ET-2L-mis + CE | 0.023 | 1.8% |

The weights matter only where $\delta$ is not small. That happens for the networks
with a CE pair, whose circular term has a spectral shape the full-covariance
weights can partly project out. For co-located ET-Δ the circular response
vanishes and $\delta \approx 0$. Under the projected budget, circular
polarization's share drops from 11.0% to 9.1% for ET-2L-mis + CE, from 3.8% to
2.8% for ET-Δ + CE, and from 2.8% to 1.1% for ET-2L-par + CE. Detector noise's
share rises accordingly.

## Monte Carlo

The response moments $R^X$ are averages over draws of direction, $\psi$ and
$\cos\iota$. Every draw enters together with its images under $\psi + \pi/4$ and
$-\cos\iota$. The Stokes cross terms $\langle\mathbf y^X\mathbf y^{Y\,T}\rangle$
then cancel draw by draw, and $R^I + R^{\rm lin} + R^{\rm circ}$ equals the moment
of the full response to round-off. The script records this as `stokes_closure`.
Without the images, the cross terms are Monte Carlo noise of order
$L^{-1/2}$. In the previous version of the script they left the Stokes shares
summing to $1 \pm 0.014$.
