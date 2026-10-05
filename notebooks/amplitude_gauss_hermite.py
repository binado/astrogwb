import marimo

__generated_with = "0.25.0"
app = marimo.App()

with app.setup(hide_code=True):
    from dataclasses import dataclass

    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    from jax.scipy.special import log_ndtr, logsumexp
    from matplotlib.colors import Normalize
    from numpy.polynomial.hermite import hermgauss
    from numpy.polynomial.legendre import leggauss
    from scipy.stats import norm

    from astrogwb.distributions.amplitude import AmplitudeConditional, amplitude_prior
    from astrogwb.inference import amplitude_H0_transform


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Can Gauss-Hermite replace the fixed amplitude grid?

    `AmplitudeConditional.log_normalizer` integrates

    $$Z = \int \pi_A(A)\, e^{-\frac12 (\rho (A-\hat A))^2}\,\mathrm{d}A$$

    with the trapezoid rule on a fixed 1024-node grid spanning the whole prior.
    The likelihood factor is a Gaussian of width $\sigma_A = 1/\rho$ centred on
    $\hat A$, so a rule *adapted to that Gaussian* is the natural alternative.
    Substituting $A = \hat A + \sqrt2\,x/\rho$ turns it into the Gauss-Hermite
    weight $e^{-x^2}$:

    $$Z_k = \frac{\sqrt2}{\rho} \sum_{i=1}^{k} w_i\, \pi_A(A_i), \qquad
    A_i = \hat A + \frac{\sqrt2\, x_i}{\rho},$$

    with $(x_i, w_i)$ from `numpy.polynomial.hermite.hermgauss(k)`. The nodes
    follow $\hat A$ and the width follows $\rho$, so $k$ nodes resolve the
    likelihood at any SNR -- the thing the fixed grid cannot do (it needs
    `effective_nodes` to stay large). The price is that the prior is evaluated
    only at $k$ points, so it must be smooth *across the node span*,
    $|A - \hat A| \lesssim 2\sqrt{k}\,\sigma_A$ (the outermost node sits at
    $x_{\max}\approx\sqrt{2k}$).

    A **bounded prior** is where that fails. Nodes outside the support are
    masked to zero (the mask is explicit: `Uniform.log_prob` is *not* $-\infty$
    off-support), and the integrand then has a step the rule cannot see. This
    notebook measures what that costs. It uses library code only and compares
    against analytic or dense-quadrature references.

    Notation: $d = \rho(\hat A - a)$ is the distance from the MLE to the lower
    bound $a$ in units of $\sigma_A$.
    """)
    return


@app.cell
def _():
    # Both bounded priors share the support [0.5, 3.5] in A, so they differ only
    # in shape: flat vs 1/A^2 (the pushforward of a uniform H0 prior).
    LOWER, UPPER = 0.5, 3.5
    H0_FIDUCIAL = 70.0
    K_VALUES = [4, 8, 16, 32, 64, 128]
    RHOS = [1.0, 3.0, 10.0, 30.0, 100.0]
    # Fine enough to catch every node crossing the bound: jumps in ln Z are narrow.
    D_GRID = np.linspace(-2.0, 12.0, 701)
    NORMAL_PRIOR = dist.Normal(2.0, 0.5)

    @dataclass(frozen=True)
    class BoundedPrior:
        """A prior on ``A`` with explicit support, since log_prob is not -inf off it."""

        prior: dist.Distribution
        lower: float
        upper: float

    BOUNDED = {
        "uniform": BoundedPrior(dist.Uniform(LOWER, UPPER), LOWER, UPPER),
        "1/A^2 (H0 pushforward)": BoundedPrior(
            amplitude_prior(
                dist.Uniform(20.0, 140.0), amplitude_H0_transform(H0_FIDUCIAL)
            ),
            LOWER,
            UPPER,
        ),
    }
    return BOUNDED, D_GRID, K_VALUES, LOWER, NORMAL_PRIOR, RHOS, UPPER, BoundedPrior


@app.cell
def _(BoundedPrior):
    def gauss_hermite_log_normalizer(bounded, mle, rho, k):
        """``ln Z_k``: the Gauss-Hermite estimate, nodes off the support masked."""
        x, w = hermgauss(k)
        mle, rho = jnp.asarray(mle), jnp.asarray(rho)
        nodes = mle[..., None] + jnp.sqrt(2.0) * x / rho[..., None]
        inside = (nodes >= bounded.lower) & (nodes <= bounded.upper)
        # Clip before the prior sees a node, so no off-support value is evaluated.
        log_prior = bounded.prior.log_prob(
            jnp.clip(nodes, bounded.lower, bounded.upper)
        )
        terms = jnp.where(inside, jnp.log(w) + log_prior, -jnp.inf)
        return jnp.log(jnp.sqrt(2.0) / rho) + logsumexp(terms, axis=-1)

    def gauss_legendre_log_normalizer(bounded, mle, rho, k, half_width=7.0):
        """``ln Z_k``: Gauss-Legendre on the support clipped to ``mle +/- 7 sigma_A``.

        The bound enters as an integration limit, not as a mask, so the estimate
        is smooth in ``mle`` and autodiff picks up the boundary term. The window
        is centred on ``mle`` clipped into the support: an ``mle`` outside it gets
        the slice of the support nearest to it.
        """
        x, w = leggauss(k)
        mle, rho = jnp.asarray(mle), jnp.asarray(rho)
        reach = half_width / rho
        centre = jnp.clip(mle, bounded.lower, bounded.upper)
        lo = jnp.maximum(bounded.lower, centre - reach)
        hi = jnp.minimum(bounded.upper, centre + reach)
        nodes = 0.5 * (hi + lo)[..., None] + 0.5 * (hi - lo)[..., None] * x
        log_integrand = (
            bounded.prior.log_prob(nodes)
            - 0.5 * (rho[..., None] * (nodes - mle[..., None])) ** 2
        )
        return jnp.log(0.5 * (hi - lo)) + logsumexp(jnp.log(w) + log_integrand, axis=-1)

    def trapezoid_log_normalizer(bounded, mle, rho, num_nodes=1024):
        """``ln Z`` from the fixed 1024-node trapezoid the library used to apply.

        The library now integrates with Gauss-Legendre (see the conclusion), so
        the old rule is reproduced here: nodes uniform in the *base*
        distribution's variable, pushed through the transforms and sorted.
        """
        prior = bounded.prior
        if isinstance(prior, dist.TransformedDistribution):
            base = prior.base_dist
            nodes = jnp.linspace(
                base.support.lower_bound, base.support.upper_bound, num_nodes
            )
            for transform in prior.transforms:
                nodes = transform(nodes)
            nodes = jnp.sort(nodes)
        else:
            nodes = jnp.linspace(bounded.lower, bounded.upper, num_nodes)
        mle, rho = jnp.asarray(mle), jnp.asarray(rho)
        log_y = (
            prior.log_prob(nodes)
            - 0.5 * (rho[..., None] * (nodes - mle[..., None])) ** 2
        )
        shift = jnp.max(log_y, axis=-1, keepdims=True)
        integral = jnp.trapezoid(jnp.exp(log_y - shift), nodes, axis=-1)
        return jnp.squeeze(shift, axis=-1) + jnp.log(integral)

    def _log_diff(upper, lower):
        return upper + jnp.log1p(-jnp.exp(lower - upper))

    def uniform_reference(mle, rho, lower, upper):
        r"""Closed form: :math:`\sqrt{2\pi}[\Phi(u) - \Phi(l)]/(\rho(b-a))`."""
        u, lo = rho * (upper - mle), rho * (lower - mle)
        # Subtract on the side where the normal CDF is small, to avoid cancelling.
        log_mass = jnp.where(
            lo > 0,
            _log_diff(log_ndtr(-lo), log_ndtr(-u)),
            _log_diff(log_ndtr(u), log_ndtr(lo)),
        )
        return (
            0.5 * jnp.log(2.0 * jnp.pi)
            - jnp.log(rho)
            - jnp.log(upper - lower)
            + log_mass
        )

    def dense_reference(bounded, mle, rho, num_nodes=50_001):
        """Dense trapezoid over the support within 12 sigma_A of the MLE."""
        out = []
        for m, r in zip(np.atleast_1d(mle), np.atleast_1d(rho), strict=True):
            a, b = max(bounded.lower, m - 12.0 / r), min(bounded.upper, m + 12.0 / r)
            nodes = jnp.linspace(a, b, num_nodes)
            log_y = bounded.prior.log_prob(nodes) - 0.5 * (r * (nodes - m)) ** 2
            shift = log_y.max()
            out.append(shift + jnp.log(jnp.trapezoid(jnp.exp(log_y - shift), nodes)))
        return jnp.stack(out)

    def normal_reference(prior, mle, rho):
        """Product of Gaussians: ``sqrt(2 pi)/rho * N(mle; mu, sqrt(s^2 + 1/rho^2))``."""
        total = jnp.sqrt(prior.scale**2 + rho**-2)
        return (
            0.5 * jnp.log(2.0 * jnp.pi)
            - jnp.log(rho)
            + dist.Normal(prior.loc, total).log_prob(mle)
        )

    return (
        dense_reference,
        gauss_hermite_log_normalizer,
        gauss_legendre_log_normalizer,
        normal_reference,
        trapezoid_log_normalizer,
        uniform_reference,
    )


@app.cell(hide_code=True)
def _(LOWER, UPPER, BOUNDED, dense_reference, uniform_reference):
    # The dense routine is the reference for the 1/A^2 prior, so first check it
    # on the prior that has a closed form.
    _mle = jnp.linspace(LOWER - 0.1, UPPER + 0.1, 9)
    _rho = jnp.full_like(_mle, 20.0)
    _gap = float(
        jnp.max(
            jnp.abs(
                dense_reference(BOUNDED["uniform"], _mle, _rho)
                - uniform_reference(_mle, _rho, LOWER, UPPER)
            )
        )
    )
    assert _gap < 1e-7, _gap
    mo.md(
        f"The dense trapezoid reproduces the closed-form uniform result to "
        f"{_gap:.1e} in $\\ln Z$, so it is a trustworthy reference for $1/A^2$."
    )
    return


@app.cell
def _(
    BOUNDED,
    D_GRID,
    K_VALUES,
    LOWER,
    RHOS,
    UPPER,
    dense_reference,
    gauss_hermite_log_normalizer,
    gauss_legendre_log_normalizer,
    trapezoid_log_normalizer,
    uniform_reference,
):
    # errors[name][rho] -> (len(K_VALUES), len(D_GRID)); the trapezoid is (len(D_GRID),).
    # Cells with the MLE more than 2 sigma_A above the upper bound are NaN: the
    # sweep is about the lower bound, and past that the upper one takes over.
    errors: dict[str, dict[float, np.ndarray]] = {}
    trapezoid_errors: dict[str, dict[float, np.ndarray]] = {}
    legendre_errors: dict[str, dict[float, np.ndarray]] = {}
    for _name, _bounded in BOUNDED.items():
        errors[_name], trapezoid_errors[_name] = {}, {}
        legendre_errors[_name] = {}
        for _rho in RHOS:
            _rho_grid = jnp.full(D_GRID.shape, _rho)
            _mle = LOWER + D_GRID / _rho
            _reference = (
                uniform_reference(_mle, _rho_grid, LOWER, UPPER)
                if _name == "uniform"
                else dense_reference(_bounded, np.asarray(_mle), np.asarray(_rho_grid))
            )
            _valid = np.asarray(_rho * (UPPER - _mle) >= -2.0)
            _per_k = np.stack(
                [
                    np.abs(
                        np.asarray(
                            gauss_hermite_log_normalizer(_bounded, _mle, _rho_grid, _k)
                            - _reference
                        )
                    )
                    for _k in K_VALUES
                ]
            )
            _trap = np.abs(
                np.asarray(
                    trapezoid_log_normalizer(_bounded, _mle, _rho_grid) - _reference
                )
            )
            _per_k_legendre = np.stack(
                [
                    np.abs(
                        np.asarray(
                            gauss_legendre_log_normalizer(_bounded, _mle, _rho_grid, _k)
                            - _reference
                        )
                    )
                    for _k in K_VALUES
                ]
            )
            legendre_errors[_name][_rho] = np.where(_valid, _per_k_legendre, np.nan)
            errors[_name][_rho] = np.where(_valid, _per_k, np.nan)
            trapezoid_errors[_name][_rho] = np.where(_valid, _trap, np.nan)
    return errors, legendre_errors, trapezoid_errors


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Bounded priors: error against bound distance

    One heatmap per SNR $\rho$ and prior, $\log_{10}|\Delta\ln Z|$ against the
    number of nodes $k$ and the bound distance $d$. Black crosses mark the
    outermost node, $2\sqrt k$: *below* the cross the bound is inside the node
    span and the rule is blind to a step in the integrand; *above* it the bound
    lies beyond every node and GH sees a smooth prior, so the error should
    approach the Gaussian mass cut off by the bound, $\Phi(-d)$. White cells
    are infinite: every node was masked.

    The supports are $A\in[0.5, 3.5]$, so $\rho=1$ has *both* bounds within
    $3\sigma_A$ of each other and no $k$ can work -- the minimum SNR for a given
    prior width is visible here as the first column that has a clean region.
    """)
    return


@app.cell
def _(BOUNDED, D_GRID, K_VALUES, RHOS):
    def error_heatmaps(errs, *, span_crosses):
        """One ``log10 |dlnZ|`` heatmap of ``d`` against ``k`` per prior and rho."""
        fig, axes = plt.subplots(
            len(BOUNDED),
            len(RHOS),
            figsize=(3.0 * len(RHOS), 2.9 * len(BOUNDED)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        norm_ = Normalize(vmin=-12, vmax=0)
        image = None
        for row, name in enumerate(BOUNDED):
            for col, rho in enumerate(RHOS):
                ax = axes[row, col]
                with np.errstate(divide="ignore"):
                    log_err = np.log10(errs[name][rho])
                log_err = np.where(np.isinf(log_err) & (log_err > 0), np.nan, log_err)
                image = ax.pcolormesh(
                    np.arange(len(K_VALUES) + 1) - 0.5,
                    np.append(D_GRID - 0.01, D_GRID[-1] + 0.01),
                    np.clip(log_err, -12, 0).T,
                    norm=norm_,
                    cmap="viridis_r",
                )
                if span_crosses:
                    ax.plot(
                        np.arange(len(K_VALUES)),
                        2 * np.sqrt(K_VALUES),
                        "kx",
                        markersize=5,
                    )
                ax.set_xticks(range(len(K_VALUES)), [str(k) for k in K_VALUES])
                ax.set_ylim(D_GRID[0], D_GRID[-1])
                if row == 0:
                    ax.set_title(rf"$\rho={rho:g}$")
                if row == len(BOUNDED) - 1:
                    ax.set_xlabel("$k$")
                if col == 0:
                    ax.set_ylabel(f"{name}\n$d = \\rho(\\hat A - a)$")
        fig.colorbar(image, ax=axes, label=r"$\log_{10}|\Delta \ln Z|$")
        return fig

    return (error_heatmaps,)


@app.cell
def _(error_heatmaps, errors):
    error_heatmaps(errors, span_crosses=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. What NUTS would see: $\ln Z$ and $\partial\ln Z/\partial\hat A$ across a bound

    Slide the MLE across the lower bound at $\rho=30$ and plot the estimator's
    error and its derivative. The gradient is taken with `jax.grad` through the
    masked rule, which is what the sampler differentiates (the red curve is the
    Gauss-Legendre window of section 3, which has no mask). Two things matter:
    the **jumps** in $\ln Z$ each time a node crosses the bound, and the
    gradient, which for a *flat* prior is identically zero (the mask is
    piecewise constant and the prior's own derivative vanishes) while the true
    derivative is the bound's pull on the posterior.
    """)
    return


@app.cell
def _(
    BOUNDED,
    LOWER,
    UPPER,
    dense_reference,
    gauss_hermite_log_normalizer,
    gauss_legendre_log_normalizer,
    trapezoid_log_normalizer,
    uniform_reference,
):
    _rho = 30.0
    _d = np.linspace(-1.0, 8.0, 1801)
    _mle = jnp.asarray(LOWER + _d / _rho)
    _rho_grid = jnp.full_like(_mle, _rho)
    _ks = (16, 64)

    def _gradient(fn, bounded):
        scalar = lambda m: fn(bounded, m, jnp.asarray(_rho))
        return jax.vmap(jax.grad(scalar))(_mle)

    _fig, _axes = plt.subplots(
        len(BOUNDED), 2, figsize=(10, 3.2 * len(BOUNDED)), constrained_layout=True
    )
    for _row, (_name, _bounded) in enumerate(BOUNDED.items()):
        if _name == "uniform":
            _ref = uniform_reference(_mle, _rho_grid, LOWER, UPPER)
            _ref_grad = jax.vmap(
                jax.grad(lambda m: uniform_reference(m, _rho, LOWER, UPPER))
            )(_mle)
        else:
            _ref = dense_reference(_bounded, np.asarray(_mle), np.asarray(_rho_grid))
            _step = 1e-5
            _ref_grad = (
                dense_reference(
                    _bounded, np.asarray(_mle + _step), np.asarray(_rho_grid)
                )
                - dense_reference(
                    _bounded, np.asarray(_mle - _step), np.asarray(_rho_grid)
                )
            ) / (2 * _step)
        _ax_err, _ax_grad = _axes[_row]
        for _k in _ks:
            _fn = lambda b, m, r, k=_k: gauss_hermite_log_normalizer(b, m, r, k)
            _ax_err.plot(
                _d, _fn(_bounded, _mle, _rho_grid) - _ref, label=f"GH $k={_k}$"
            )
            _ax_grad.plot(_d, _gradient(_fn, _bounded), label=f"GH $k={_k}$")
        _gl = lambda b, m, r: gauss_legendre_log_normalizer(b, m, r, 16)
        _ax_err.plot(
            _d, _gl(_bounded, _mle, _rho_grid) - _ref, "C3", label="GL window $k=16$"
        )
        _ax_grad.plot(_d, _gradient(_gl, _bounded), "C3", label="GL window $k=16$")
        _ax_err.plot(
            _d,
            trapezoid_log_normalizer(_bounded, _mle, _rho_grid) - _ref,
            "k--",
            label="trapezoid (1024)",
        )
        _ax_grad.plot(_d, _ref_grad, "k:", label="reference")
        _ax_err.axhline(0, color="gray", lw=0.5)
        _ax_err.set_ylim(-0.3, 0.3)
        _ax_err.set_ylabel(r"$\ln Z - \ln Z_{\mathrm{ref}}$")
        _ax_grad.set_ylabel(r"$\partial \ln Z/\partial \hat A$")
        _ax_err.set_title(f"{_name}, $\\rho={_rho:g}$")
        for _ax in (_ax_err, _ax_grad):
            _ax.set_xlabel(r"$d = \rho(\hat A - a)$")
        _ax_grad.legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Gauss-Legendre on the clipped window

    Gauss-Hermite fails at a bound because the bound is a *mask* it cannot
    differentiate. The alternative is to make the bound an integration *limit*.
    Integrate over $[\max(a,\hat A - 7\sigma_A),\ \min(b,\hat A + 7\sigma_A)]$
    with a $k$-point Gauss-Legendre rule. On that window the integrand is a
    Gaussian times a smooth prior, so the rule converges exponentially in $k$;
    the window is centred on $\hat A$ and scaled by $1/\rho$, so it follows the
    SNR; and the limits are piecewise-smooth in $\hat A$, so autodiff
    differentiates through them (the Leibniz boundary term is exactly the
    bound's pull). The $7\sigma_A$ truncation costs $\approx10^{-12}$. The
    same code covers an infinite support, where nothing is clipped.

    Same sweep as section 1, with no span markers (there is no node span to
    cross):
    """)
    return


@app.cell
def _(error_heatmaps, legendre_errors):
    error_heatmaps(legendre_errors, span_crosses=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The MLE far *outside* the support (a very bad fit, e.g. early in warmup) is
    the one case the window cannot centre on $\hat A$: it takes the slice of the
    support nearest to $\hat A$, over which the integrand is a steep exponential
    that $k$ nodes must resolve. Error in $\ln Z$ at $\rho=30$, uniform prior, as
    $\hat A$ moves below the lower bound:
    """)
    return


@app.cell
def _(
    BOUNDED,
    K_VALUES,
    LOWER,
    UPPER,
    gauss_legendre_log_normalizer,
    uniform_reference,
):
    _rho = 30.0
    _outside = np.array([2.0, 5.0, 10.0, 20.0, 40.0])  # sigma_A below the bound
    _mle = jnp.asarray(LOWER - _outside / _rho)
    _rho_grid = jnp.full_like(_mle, _rho)
    _reference = uniform_reference(_mle, _rho_grid, LOWER, UPPER)
    print("MLE sigma_A below the lower bound:", _outside)
    for _k in K_VALUES:
        _err = np.abs(
            np.asarray(
                gauss_legendre_log_normalizer(BOUNDED["uniform"], _mle, _rho_grid, _k)
                - _reference
            )
        )
        print(f"GL k={_k:<4}", " ".join(f"{e:9.1e}" for e in _err))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. A smooth prior: Normal on $A$

    With no bound, GH has nothing to mask. What decides the error is whether the
    prior is smooth *on the scale of the node spacing*: a prior much wider than
    $\sigma_A$ is nearly constant across the nodes (GH is exact for a constant,
    already at $k=1$), while one narrower than $\sigma_A$ is sampled by too few
    nodes. Here $\pi_A = \mathcal N(2, 0.5)$ and $\hat A$ sits one prior
    standard deviation from the mean; the ratio $\sigma_\pi/\sigma_A = 0.5\rho$
    is what varies along the horizontal axis.
    """)
    return


@app.cell
def _(
    K_VALUES,
    NORMAL_PRIOR,
    gauss_hermite_log_normalizer,
    gauss_legendre_log_normalizer,
    normal_reference,
    trapezoid_log_normalizer,
):
    _rhos = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    _mle = jnp.full(_rhos.shape, NORMAL_PRIOR.loc + NORMAL_PRIOR.scale)
    _reference = normal_reference(NORMAL_PRIOR, _mle, jnp.asarray(_rhos))
    _bounded = type(
        "Unbounded", (), {"prior": NORMAL_PRIOR, "lower": -np.inf, "upper": np.inf}
    )()
    normal_errors = np.stack(
        [
            np.abs(
                np.asarray(
                    gauss_hermite_log_normalizer(_bounded, _mle, jnp.asarray(_rhos), _k)
                    - _reference
                )
            )
            for _k in K_VALUES
        ]
    )
    normal_legendre = np.stack(
        [
            np.abs(
                np.asarray(
                    gauss_legendre_log_normalizer(
                        _bounded, _mle, jnp.asarray(_rhos), _k
                    )
                    - _reference
                )
            )
            for _k in K_VALUES
        ]
    )
    normal_trapezoid = np.abs(
        np.asarray(
            trapezoid_log_normalizer(_bounded, _mle, jnp.asarray(_rhos)) - _reference
        )
    )
    _fig, _ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    for _row, _k in enumerate(K_VALUES):
        _ax.loglog(
            NORMAL_PRIOR.scale * _rhos,
            normal_errors[_row] + 1e-17,
            "o-",
            label=f"GH $k={_k}$",
        )
        _ax.loglog(
            NORMAL_PRIOR.scale * _rhos,
            normal_legendre[_row] + 1e-17,
            "s--",
            color=f"C{_row}",
            alpha=0.5,
            label=f"GL $k={_k}$",
        )
    _ax.loglog(
        NORMAL_PRIOR.scale * _rhos,
        normal_trapezoid + 1e-17,
        "k--",
        label="trapezoid (1024)",
    )
    _ax.set_xlabel(r"$\sigma_\pi/\sigma_A = 0.5\rho$")
    _ax.set_ylabel(r"$|\Delta \ln Z|$")
    _ax.set_ylim(1e-17, 10)
    _ax.legend(fontsize=6, ncol=3)
    _fig
    return normal_errors, normal_legendre, normal_trapezoid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## High SNR

    The sweeps above stop at $\rho=100$. The fixed grid's resolution is
    $\Delta A\approx3/1024\approx3\times10^{-3}$ against $\sigma_A=1/\rho$, so it
    breaks down at $\rho\gtrsim300$; GL with a $7\sigma_A$ window does not care.
    Error in $\ln Z$ for $\rho=10^3$--$10^5$ with the MLE a distance $d$ from the
    lower bound (the library's `AmplitudeConditional` is Gauss-Legendre $k=32$).
    """)
    return


@app.cell
def _(
    BOUNDED,
    LOWER,
    UPPER,
    dense_reference,
    gauss_legendre_log_normalizer,
    trapezoid_log_normalizer,
    uniform_reference,
):
    _rows = []
    for _name, _bounded in BOUNDED.items():
        for _rho in [1e3, 1e4, 1e5]:
            for _d in [-2.0, 0.0, 3.0, 12.0]:
                _mle = jnp.asarray([LOWER + _d / _rho])
                _r = jnp.asarray([_rho])
                _ref = (
                    uniform_reference(_mle, _r, LOWER, UPPER)
                    if _name == "uniform"
                    else dense_reference(_bounded, np.asarray(_mle), np.asarray(_r))
                )
                _library = AmplitudeConditional(
                    _mle, _r, prior=_bounded.prior
                ).log_normalizer
                _rows.append(
                    (
                        _name,
                        _rho,
                        _d,
                        float(
                            jnp.abs(
                                trapezoid_log_normalizer(_bounded, _mle, _r) - _ref
                            )[0]
                        ),
                        float(
                            jnp.abs(
                                gauss_legendre_log_normalizer(_bounded, _mle, _r, 32)
                                - _ref
                            )[0]
                        ),
                        float(jnp.abs(_library - _ref)[0]),
                    )
                )
    _lines = [
        f"{'prior':<24}{'rho':>8}{'d':>6}{'trapezoid':>12}{'GL(32)':>12}{'library':>12}"
    ]
    _lines += [
        f"{n:<24}{r:>8g}{d:>6g}{t:>12.1e}{g:>12.1e}{lib:>12.1e}"
        for n, r, d, t, g, lib in _rows
    ]
    assert all(lib < 1e-6 for *_, lib in _rows), _rows
    mo.md("```\n" + "\n".join(_lines) + "\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Conclusion

    For each prior, SNR and tolerance, the table gives the best $k$ and the
    smallest bound distance $d_{\min}$ such that the GH error is below the
    tolerance for *every* $d \ge d_{\min}$ on the sweep. The trapezoid column
    is the same quantity for the library's fixed grid.
    """)
    return


@app.cell
def _(
    BOUNDED,
    D_GRID,
    K_VALUES,
    RHOS,
    errors,
    legendre_errors,
    normal_errors,
    normal_legendre,
    trapezoid_errors,
):
    def minimum_distance(err, tol):
        """Smallest ``d`` with ``err < tol`` for all valid ``d' >= d`` (else inf)."""
        for i in range(len(D_GRID)):
            tail = err[i:]
            tail = tail[~np.isnan(tail)]
            if tail.size and np.all(tail < tol):
                return D_GRID[i]
        return np.inf

    thresholds = {}
    lines = []
    for tol in (1e-3, 1e-6):
        lines.append(
            f"\n=== tolerance |dlnZ| < {tol:g} "
            f"(Gaussian tail Phi(-d) < tol needs d > {-norm.ppf(tol):.2f}) ==="
        )
        lines.append(
            f"{'prior':<24}{'rho':>6}{'GH k':>6}{'d_min GH':>10}"
            f"{'GL k':>6}{'d_min GL':>10}{'d_min trap':>12}"
        )
        for name in BOUNDED:
            for rho in RHOS:
                per_k = [
                    minimum_distance(errors[name][rho][i], tol)
                    for i in range(len(K_VALUES))
                ]
                best = int(np.argmin(per_k))
                trap = minimum_distance(trapezoid_errors[name][rho], tol)
                # Smallest k, since for GL more nodes never hurt: the first k to hit
                # the lowest d_min on offer.
                per_k_gl = [
                    minimum_distance(legendre_errors[name][rho][i], tol)
                    for i in range(len(K_VALUES))
                ]
                best_gl = int(np.argmin(per_k_gl))
                thresholds[(name, rho, tol)] = (
                    K_VALUES[best],
                    per_k[best],
                    K_VALUES[best_gl],
                    per_k_gl[best_gl],
                    trap,
                )
                lines.append(
                    f"{name:<24}{rho:>6g}"
                    f"{K_VALUES[best] if np.isfinite(per_k[best]) else '-':>6}"
                    f"{per_k[best]:>10.2f}"
                    f"{K_VALUES[best_gl] if np.isfinite(per_k_gl[best_gl]) else '-':>6}"
                    f"{per_k_gl[best_gl]:>10.2f}{trap:>12.2f}"
                )
        for name in BOUNDED:
            per_k = [
                minimum_distance(errors[name][30.0][i], tol)
                for i in range(len(K_VALUES))
            ]
            lines.append(
                f"d_min by k at rho=30, {name}: "
                + ", ".join(
                    f"k={k}: {d:.2f}" for k, d in zip(K_VALUES, per_k, strict=True)
                )
                + "  (outermost node 2 sqrt(k) = "
                + ", ".join(f"{2 * np.sqrt(k):.1f}" for k in K_VALUES)
                + ")"
            )
        for name in BOUNDED:
            per_k = [
                minimum_distance(legendre_errors[name][30.0][i], tol)
                for i in range(len(K_VALUES))
            ]
            lines.append(
                f"GL d_min by k at rho=30, {name}: "
                + ", ".join(
                    f"k={k}: {d:.2f}" for k, d in zip(K_VALUES, per_k, strict=True)
                )
            )
        smooth_gl = [
            min(
                (
                    k
                    for k, e in zip(K_VALUES, normal_legendre[:, j], strict=True)
                    if e < tol
                ),
                default=None,
            )
            for j in range(normal_legendre.shape[1])
        ]
        smooth = [
            min(
                (
                    k
                    for k, e in zip(K_VALUES, normal_errors[:, j], strict=True)
                    if e < tol
                ),
                default=None,
            )
            for j in range(normal_errors.shape[1])
        ]
        lines.append(
            f"normal prior, smallest k below tol at sigma_pi/sigma_A "
            f"= 0.5, 1.5, 5, 15, 50, 150: GH {smooth}, GL {smooth_gl}"
        )
    print("\n".join(lines))
    return (thresholds,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **What the numbers say** (swept over $k\in\{4,\dots,128\}$,
    $\rho\in\{1,\dots,100\}$, $d\in[-2,12]$ in steps of 0.02):

    1. *The threshold is the Gaussian tail mass, not the node span.* GH is
       within $10^{-3}$ of the truth once $d\gtrsim 3.1$ and within $10^{-6}$
       once $d\gtrsim 4.6$--$4.9$, for both bounded priors and **almost
       independent of $k$** ($k=4$: 3.1 / 4.8; $k=128$: 2.7 / 4.6). The
       expected picture -- error stops converging once the bound is inside the
       $2\sqrt k\,\sigma_A$ span, so a larger $k$ pulls the bound in -- is only
       true for $d \lesssim 3$: the outermost nodes carry weights of order
       $e^{-d^2/2}$, so masking them costs no more than the tail mass the bound
       removes anyway. Inside $d \lesssim 3$ the error is $10^{-3}$--$1$ at any
       $k$, $\ln Z$ jumps by up to $\pm0.3$ each time a node crosses the bound
       (section 2), and more nodes only soften the jumps.
    2. *The gradient is the real problem.* The sampler differentiates through a
       masked sum. For a flat prior that gradient is identically $0$ and for
       $1/A^2$ it is the prior's slope, while the true
       $\partial\ln Z/\partial\hat A$ is $\sim 45$ at $d=-1$ and still
       $\sim 1$ at $d\approx 2$ (section 2, right). A GH likelihood therefore
       does not feel a nearby bound at all, which is exactly the regime where
       the bound shapes the posterior.
    3. *Both bounds must clear $d_{\min}$*, so the prior must be at least
       $2 d_{\min}\sigma_A$ wide with the MLE inside it: a width of
       $\gtrsim 6\,\sigma_A$ for $10^{-3}$ and $\gtrsim 9.5\,\sigma_A$ for
       $10^{-6}$, i.e. $\rho \gtrsim 2$ and $\rho\gtrsim 3$ for the width-3
       supports used here. Below that (the $\rho=1$ column) no $k$ works.
    4. *A smooth prior needs few nodes.* For the normal, $k=4$ reaches
       $10^{-6}$ once $\sigma_\pi/\sigma_A\gtrsim 5$; $k=8$--$16$ at
       $\sigma_\pi/\sigma_A\approx1.5$; $k=16$--$32$ at $0.5$. The same holds
       for $1/A^2$ away from its bounds.
    5. *The trapezoid fails differently.* It is exact on the support, so a
       nearby bound costs it nothing ($d_{\min}\approx-2$ at $\rho\le10$), but
       a fixed grid under-resolves the likelihood at high SNR: at $\rho=100$
       the 1024 nodes put $\sigma_A=0.01$ across $\sim3$ nodes and $d_{\min}$
       rises to 1.9 ($10^{-3}$) and 4.3 ($10^{-6}$), no better than GH. On
       $1/A^2$ it also has a $\sim10^{-6}$ floor from the grid being uniform in
       $H_0$.

    6. *Gauss-Legendre on the clipped window has no such failure.* With the
       bound as an integration limit, $k=16$ ($10^{-3}$) or $k=32$ ($10^{-6}$)
       is accurate over the whole sweep, $d\in[-2,12]$ at every $\rho$, even
       $\rho=1$ where the support is only $3\sigma_A$ wide (the window just
       shrinks to the support). Its gradient lies on the reference in section 2.
       The error is smooth in $\hat A$: no jumps. The price against GH is node
       count on a smooth infinite prior: the window always spans
       $\pm7\sigma_A$, so a normal prior needs $k=16$--$32$ ($10^{-3}$) and
       $32$--$64$ ($10^{-6}$) where GH needs 4 once $\sigma_\pi/\sigma_A\gtrsim5$.
       Its weak spot is the MLE far outside the support: with $k=32$ the error
       is $\lesssim5\times10^{-6}$ out to $40\sigma_A$ and $k=64$ gives
       $\lesssim10^{-8}$ (the check above; the $4\times10^{-9}$ plateau at
       $20\sigma_A$ is $k$-independent and I have not traced whether it is the
       reference's precision).

    **Recommendation: replace the trapezoid with Gauss-Legendre on the clipped
    window; do not adopt Gauss-Hermite.** GH is rejected for bounded priors: it
    is inaccurate and its gradient blind to a bound within $\sim3\sigma_A$ of
    the MLE, which is where the bound matters. A hybrid would work but needs a
    runtime `lax.cond` (a select under `vmap`). GL needs no branch, handles a
    bounded or unbounded support with the same code, is correct near a bound,
    and unlike the 1024-node trapezoid its accuracy does not degrade with
    $\rho$. Suggested defaults are $k=32$ (about $10^{-6}$ in $\ln Z$) and a
    $7\sigma_A$ half-width, i.e. 32 prior evaluations instead of 1024. What it
    changes in the library, beyond `log_normalizer`: `AmplitudeConditional.icdf`
    and `sample` tabulate a CDF on the fixed grid, so sampling would still need
    that grid (or a window-based CDF), and `effective_nodes` would lose its
    meaning. Before committing to it, benchmark it inside NUTS with the
    production chain layout; the speed-up is structural (32 vs 1024
    evaluations) and I have not measured it.

    **Adopted.** `AmplitudeConditional` now integrates with Gauss-Legendre
    $k=32$ on a $7\\sigma_A$ window clipped to the support
    (`astrogwb.distributions.amplitude`), `sample`/`icdf` tabulate their CDF on
    that same window, and `quadrature_grid` and `effective_nodes` are gone. The
    `trapezoid` estimator in this notebook is a local copy of the retired rule,
    kept so the comparison stays reproducible. The high-SNR table above shows
    the library reaching $\\lesssim10^{-9}$ at $\\rho$ up to $10^5$.
    """)
    return


if __name__ == "__main__":
    app.run()
