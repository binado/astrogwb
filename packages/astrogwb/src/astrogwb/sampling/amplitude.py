r"""Numerical marginalization of a multiplicative amplitude direction.

Under the per-frequency Gaussian likelihood used by
:func:`~astrogwb.sampling.models.spectral_density_model`, one parameter can
enter the predicted spectrum as a pure multiplicative factor,

.. math:: \boldsymbol{\mu}(\varphi, \theta) = A(\varphi)\, \mathbf{m}(\theta)

with :math:`\mathbf{m}(\theta)` the *template* -- the spectrum evaluated at a
fixed reference value :math:`\varphi_{\mathrm{fid}}` of the marginalized
parameter -- and

.. math:: A(\varphi) = f(\varphi) / f(\varphi_{\mathrm{fid}})

the dimensionless amplitude relative to that template, for an arbitrary
scaling :math:`f` from the physical parameter :math:`\varphi`. Normalizing by
:math:`f(\varphi_{\mathrm{fid}})` here rather than trusting :math:`f` to
already satisfy :math:`f(\varphi_{\mathrm{fid}}) = 1` makes the anchoring
structurally impossible to get wrong; it is also the correct construction for
a non-power-law :math:`f`, where :math:`f(\varphi/\varphi_{\mathrm{fid}})`
would be something else entirely. The predicted spectrum factorizes into two
independently-scaling pieces, a total merger rate and a mean energy flux (the
importance-weighted polarization-power contraction), so
:math:`f = g_R \cdot g_F`; see
:func:`~astrogwb.importance.models.bns_madau_dickinson_modified_propagation.amplitude_H0_fn`
and
:func:`~astrogwb.importance.models.bns_madau_dickinson_modified_propagation.amplitude_local_merger_rate_fn`
for the concrete scalings for :math:`H_0` and ``local_merger_rate``. Define
the noise-weighted inner product
:math:`(x|y) = \sum_i x_i y_i / \sigma_i^2`. Then

.. math::

    \hat{A} = \frac{(d|m)}{(m|m)}, \qquad \rho = \sqrt{(m|m)},

are the maximum-likelihood amplitude and the template optimal SNR, and
completing the square in :math:`A` gives

.. math::

    -\tfrac{1}{2}\sum_i \left(\frac{d_i - A\, m_i}{\sigma_i}\right)^2
    = -R - \tfrac{1}{2}\rho^2 (A - \hat{A})^2,

with :math:`R = \tfrac{1}{2}\sum_i((d_i - \hat{A}m_i)/\sigma_i)^2` the
best-fit residual. This module marginalizes :math:`\varphi` numerically under
the caller's actual prior :math:`\pi(\varphi)`, rather than requiring the
prior to be stated on :math:`A` itself. The log-integrand is

.. math::

    \ell(\varphi) = \ln\pi(\varphi) - \tfrac{1}{2}\bigl(\rho\bigl(A(\varphi) - \hat{A}\bigr)\bigr)^2,

and the amplitude direction is integrated with the trapezoid rule on a fixed
1D grid after a stable max-shift:

.. math::

    \ln Z = \ln p(d \mid \hat{A})
        + \ell_{\max}
        + \ln\!\int \exp\bigl(\ell(\varphi) - \ell_{\max}\bigr)\, d\varphi,

where :math:`\ln p(d \mid \hat{A}) = \ln\mathcal{N}_d - R` is the Gaussian
log-likelihood at the MLE amplitude. Squaring
:math:`\rho(A(\varphi) - \hat{A})` rather than forming
:math:`\rho^2(A-\hat A)^2` avoids overflowing :math:`\rho^2` at very high SNR.

The grid is *purely a quadrature scheme*: it is where the normalizing
integral is evaluated, not what defines the distribution. The support is the
prior's, and :meth:`AmplitudeConditional.log_prob` evaluates the analytic
density at any :math:`\varphi` without touching the grid. The one place the
distinction shows is :meth:`AmplitudeConditional.sample`, which inverts a CDF
tabulated on the grid and therefore returns draws clipped to
``[grid[0], grid[-1]]`` -- slightly less than the declared support. That is
deliberate: the grid must cover essentially all the prior mass anyway (see
:func:`quadrature_grid`), or the normalizer is wrong for a reason no amount
of clipping would fix.

"Exact up to quadrature error" only holds if the grid resolves the conditional
posterior, whose width in :math:`\varphi` is :math:`\sigma_A/|A'(\varphi)|`.
No quadrature rule rescues a Gaussian bump spanning three nodes, so grid
adequacy must be checked with :attr:`AmplitudeConditional.effective_nodes`,
not assumed.

The conditional posterior of :math:`\varphi` given the two statistics is
exposed as :class:`AmplitudeConditional`, a NumPyro ``Distribution``: the
marginalization factor in the model is its :attr:`log_normalizer`, and
post-processing reconstructs :math:`\varphi` by drawing from it.

Everything broadcasts over leading batch dimensions and contracts over the
trailing grid axis, so post-processing can feed ``(chain, draw)`` shaped
statistics directly.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import constraints

from astrogwb.importance.diagnostics import relative_ess


class MergerRateAmplitudeFn(Protocol):
    """Total merger rate at :math:`\\varphi` up to a constant, :math:`g_R(\\varphi)`.

    Only ratios :math:`g_R(\\varphi)/g_R(\\varphi_{\\mathrm{fid}})` are used, so
    any overall normalization cancels.
    """

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


class AmplitudeFn(Protocol):
    """Full multiplicative scaling :math:`f(\\varphi) = g_R(\\varphi)\\, g_F(\\varphi)`.

    Implementations must be **hashable by value**:
    :class:`AmplitudeConditional` carries this callable as pytree *aux* data,
    which JAX hashes into the jit cache key. A module-level ``def`` is the
    safe choice; a freshly-minted lambda or a ``functools.partial`` over
    floats is identity-hashed and silently retraces the model on every
    construction.
    """

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


def quadrature_grid(
    prior: dist.Distribution,
    *,
    num_nodes: int = 4001,
    span_sigma: float = 10.0,
) -> jax.Array:
    r"""A quadrature grid covering essentially all of ``prior``'s mass.

    Spans :math:`\pm` ``span_sigma`` prior standard deviations about the prior
    mean, clipped to the prior's support. That reproduces the obvious grid for
    the two priors that matter in practice: a ``Uniform`` collapses onto its
    exact ``[low, high]`` bounds (its own support is tighter than ten standard
    deviations), and a ``Normal`` spans ``loc +/- span_sigma * scale``. At the
    default ``span_sigma=10.0`` the lost Normal tail mass is of order
    ``1e-23``, a constant offset identical for every posterior draw, so it does
    not perturb NUTS.

    The grid must cover the prior support because the normalizing integral in
    :attr:`AmplitudeConditional.log_normalizer` runs over exactly this grid --
    narrowing it truncates the prior.

    This is *not* generic over every NumPyro prior: it needs ``.mean`` and
    ``.variance``, which some distributions (``TruncatedNormal``, for one) do
    not implement. Those callers must pass an explicit ``grid=``.

    Raises
    ------
    TypeError
        If ``prior`` does not implement ``.variance``.
    """
    try:
        variance = prior.variance
    except NotImplementedError as exc:
        raise TypeError(
            f"cannot derive a quadrature grid for {type(prior).__name__}: it does "
            "not implement .variance. Pass an explicit grid= covering the prior "
            "support instead."
        ) from exc

    half_width = span_sigma * jnp.sqrt(variance)
    lower = jnp.maximum(
        prior.mean - half_width, getattr(prior.support, "lower_bound", -jnp.inf)
    )
    upper = jnp.minimum(
        prior.mean + half_width, getattr(prior.support, "upper_bound", jnp.inf)
    )
    return jnp.linspace(lower, upper, num_nodes)


def _log_trapezoid(log_y: jax.Array, x: jax.Array) -> jax.Array:
    r"""Stable :math:`\ln\int \exp(\log y)\, dx` via a shifted trapezoid rule.

    Broadcasts over leading dimensions of ``log_y`` and integrates along the
    trailing axis against the 1D abscissa ``x``.
    """
    log_y_max = jnp.max(log_y, axis=-1, keepdims=True)
    integral = jnp.trapezoid(jnp.exp(log_y - log_y_max), x, axis=-1)
    return jnp.squeeze(log_y_max, axis=-1) + jnp.log(integral)


def _cumulative_trapezoid(y: jax.Array, x: jax.Array) -> jax.Array:
    """Cumulative trapezoid integral of ``y`` vs ``x``, starting at 0."""
    dx = jnp.diff(x)
    segments = 0.5 * (y[..., :-1] + y[..., 1:]) * dx
    zeros = jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype)
    return jnp.concatenate([zeros, jnp.cumsum(segments, axis=-1)], axis=-1)


class AmplitudeConditional(dist.Distribution):
    r"""Conditional posterior of the marginalized parameter given the amplitude statistics.

    Given the amplitude sufficient statistics :math:`\hat A` and :math:`\rho`
    published by
    :func:`~astrogwb.sampling.models.amplitude_marginalized_model`, this is
    the density

    .. math::

        p(\varphi \mid d, \theta) \propto
        \pi(\varphi)\,
        \exp\!\left[-\tfrac12\bigl(\rho\,(A(\varphi) - \hat A)\bigr)^2\right],
        \qquad A(\varphi) = f(\varphi)/f(\varphi_{\mathrm{fid}}).

    The distribution owns the live pieces it is defined by -- the prior, the
    scaling, the fiducial -- rather than a precomputed tabulation of them, so
    nothing can go stale. The density above is evaluated analytically wherever
    it is asked for; the ``grid`` enters only as the quadrature scheme for the
    normalizing integral:

    - :attr:`log_normalizer` -- :math:`\ln Z` of the conditional under the
      trapezoid rule; this *is* the marginalization factor the model adds to
      its ``numpyro.factor`` site.
    - :meth:`sample` / :meth:`icdf` -- inverse-transform draws of
      :math:`\varphi` for post-processing reconstruction, clipped to the grid.
    - :attr:`effective_nodes` -- grid-adequacy diagnostic.

    Recomputing :math:`f` on the grid every step costs nothing in practice:
    the grid enters the jitted model as a closure constant, so XLA
    constant-folds :math:`f(\text{grid})` away entirely.

    The ``batch_shape`` is the broadcast of the two statistics' shapes, so a
    ``(chain, draw)`` posterior feeds in directly.

    .. warning::

        This distribution is meant for :class:`~numpyro.infer.Predictive`
        (generative-only use in
        :func:`~astrogwb.sampling.models.amplitude_reconstruction_model`). Do
        **not** ``numpyro.sample`` it as a latent site inside a NUTS model
        without revisiting two things. Its ``support`` is a
        :class:`~numpyro.distributions.constraints.dependent_property`, which
        routes latent use through NumPyro's dynamic-support path; and because
        the support is the *prior's* -- unbounded for a ``Normal`` prior --
        ``biject_to`` may be the identity, so a proposal outside the grid gets
        a perfectly finite :meth:`log_prob` normalized against an integral
        that never covered it. The tabulated implementation this replaced
        returned ``-inf`` there and failed loudly instead.

    Parameters
    ----------
    amplitude_mle, template_optimal_snr:
        The amplitude sufficient statistics :math:`\hat A` and :math:`\rho`,
        broadcast against each other.
    amplitude_fn:
        The *absolute* scaling :math:`f(\varphi)`; the ratio to the fiducial is
        formed here. Must be hashable by value -- see :class:`AmplitudeFn`.
    prior:
        The prior :math:`\pi(\varphi)` on the marginalized parameter. Defines
        the support and, together with ``num_nodes`` / ``span_sigma``, the
        default quadrature grid.
    fiducial:
        Reference value :math:`\varphi_{\mathrm{fid}}` that defines the
        template, i.e. the point at which :math:`A(\varphi) = 1`.
    grid:
        Explicit quadrature nodes. Defaults to
        ``quadrature_grid(prior, num_nodes=..., span_sigma=...)``. An explicit
        grid is *not* validated: it must be strictly increasing and cover the
        prior mass, and neither can be checked eagerly inside a traced
        ``__init__``. The default is monotone by construction.
    num_nodes, span_sigma:
        Forwarded to :func:`quadrature_grid` when ``grid`` is not given.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`.
    """

    # Plain dict like every NumPyro distribution: annotating ClassVar would
    # violate ty's override rules against Distribution's instance annotation.
    arg_constraints = {  # noqa: RUF012
        "amplitude_mle": constraints.real,
        "template_optimal_snr": constraints.positive,
    }
    pytree_data_fields = (
        "amplitude_mle",
        "template_optimal_snr",
        "prior",
        "grid",
        "fiducial",
    )
    # Aux, not data: `amplitude_fn` is a Python callable, and JAX hashes aux
    # data into the jit cache key. See `AmplitudeFn`.
    pytree_aux_fields = ("amplitude_fn",)

    def __init__(
        self,
        amplitude_mle: ArrayLike,
        template_optimal_snr: ArrayLike,
        *,
        amplitude_fn: AmplitudeFn,
        prior: dist.Distribution,
        fiducial: ArrayLike,
        grid: jax.Array | None = None,
        num_nodes: int = 4001,
        span_sigma: float = 10.0,
        validate_args: bool | None = None,
    ) -> None:
        self.amplitude_mle = jnp.asarray(amplitude_mle)
        self.template_optimal_snr = jnp.asarray(template_optimal_snr)
        self.amplitude_fn = amplitude_fn
        self.prior = prior
        self.fiducial = jnp.asarray(fiducial)
        self.grid = (
            quadrature_grid(prior, num_nodes=num_nodes, span_sigma=span_sigma)
            if grid is None
            else jnp.asarray(grid)
        )
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(amplitude_mle), jnp.shape(template_optimal_snr)
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """The prior's support -- mathematically what the conditional lives on.

        Note that :meth:`sample` covers slightly less than this, because
        :meth:`icdf` clips to the quadrature grid.
        """
        prior_support = self.prior.support
        if prior_support is None:
            raise TypeError(
                f"{type(self.prior).__name__} declares no support, so the "
                "conditional has none either"
            )
        return prior_support

    def _log_density(
        self,
        marginalized_parameter: jax.Array,
        amplitude_mle: jax.Array,
        template_optimal_snr: jax.Array,
    ) -> jax.Array:
        r"""Unnormalized :math:`\ell(\varphi)` at arbitrary :math:`\varphi`.

        The single implementation behind the normalizer, the density, and the
        inverse-CDF draw -- which is what keeps them from drifting apart. The
        statistics are passed in rather than read off ``self`` so the caller
        controls broadcasting against the trailing grid axis.
        """
        amplitude = self.amplitude_fn(marginalized_parameter) / self.amplitude_fn(
            self.fiducial
        )
        scaled_residual = template_optimal_snr * (amplitude - amplitude_mle)
        log_prior = jnp.asarray(self.prior.log_prob(marginalized_parameter))
        return log_prior - 0.5 * scaled_residual**2

    @property
    def _log_integrand(self) -> jax.Array:
        """``batch_shape + (K,)`` log-integrand on the quadrature grid."""
        return self._log_density(
            self.grid,
            self.amplitude_mle[..., None],
            self.template_optimal_snr[..., None],
        )

    @property
    def log_normalizer(self) -> jax.Array:
        r""":math:`\ln Z` of the conditional -- the marginalization factor itself.

        Stable :math:`\ln\int\exp(\ell)\,d\varphi` over the grid via
        :func:`_log_trapezoid`. The amplitude-marginalized model adds exactly
        this to the log-likelihood at the MLE: the factor *is* the normalizing
        constant of the conditional that post-processing later samples.
        """
        return _log_trapezoid(self._log_integrand, self.grid)

    @property
    def effective_nodes(self) -> jax.Array:
        r"""Grid-adequacy diagnostic: how many nodes carry the conditional posterior.

        Reuses :func:`~astrogwb.importance.diagnostics.relative_ess` on the
        log-integrand -- the same Kish effective-sample-size construction used
        for importance weights -- and rescales it by :math:`K` so the result
        is a node count rather than a fraction. A Gaussian conditional
        posterior spanning only a handful of grid nodes reports a small value
        here even though the assembled log evidence looks finite and
        plausible; this should be comfortably above approximately 30.
        """
        return relative_ess(self._log_integrand) * self.grid.shape[0]

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Exact log density, evaluated analytically off the grid.

        The normalizing constant is still the trapezoid integral over the
        grid, so :meth:`log_prob` integrates to 1 only up to quadrature error.
        """
        value = jnp.asarray(value)
        log_density = (
            self._log_density(value, self.amplitude_mle, self.template_optimal_snr)
            - self.log_normalizer
        )
        # NumPyro's `Uniform.log_prob` returns its constant density everywhere
        # rather than -inf off-support, so the mask -- not the prior term -- is
        # what keeps `log_prob` consistent with `support`.
        return jnp.where(self.support.check(value), log_density, -jnp.inf)

    def icdf(self, q: ArrayLike) -> jax.Array:
        r"""Inverse CDF by linear-in-CDF inversion on the trapezoid CDF.

        The CDF is the cumulative trapezoid of the same integrand that
        :attr:`log_normalizer` integrates, so draws follow precisely the
        density that was marginalized -- a piecewise-*constant* approximation
        of it, which agrees with :meth:`log_prob` at node resolution and
        differs sub-cell.

        Parameters
        ----------
        q:
            Quantiles in ``[0, 1]``, shaped ``sample_shape + batch_shape``.

        Returns
        -------
        jax.Array
            :math:`\varphi` values, clipped to ``[grid[0], grid[-1]]``.
        """
        q = jnp.asarray(q)
        log_integrand = self._log_integrand
        shifted = jnp.exp(
            log_integrand - jnp.max(log_integrand, axis=-1, keepdims=True)
        )
        cdf = _cumulative_trapezoid(shifted, self.grid)
        cdf = cdf / cdf[..., -1:]

        grid = self.grid
        num_nodes = grid.shape[0]
        cdf = jnp.broadcast_to(cdf, jnp.shape(q) + (num_nodes,))
        idx = jnp.clip(jnp.sum(cdf < q[..., None], axis=-1), 1, num_nodes - 1)

        cdf_hi = jnp.take_along_axis(cdf, idx[..., None], axis=-1)[..., 0]
        cdf_lo = jnp.take_along_axis(cdf, (idx - 1)[..., None], axis=-1)[..., 0]
        grid_hi = grid[idx]
        grid_lo = grid[idx - 1]

        # Deep in the tails `shifted` underflows to 0, so the CDF has long flat
        # plateaus; guard the division so those draws land at `grid_lo` instead of
        # NaN from 0/0.
        span = jnp.where(cdf_hi > cdf_lo, cdf_hi - cdf_lo, 1.0)
        fraction = jnp.where(cdf_hi > cdf_lo, (q - cdf_lo) / span, 0.0)
        return grid_lo + fraction * (grid_hi - grid_lo)

    def sample(
        self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Inverse-transform draw: one uniform per element, through :meth:`icdf`."""
        # `None` only exists to match the base-class signature; handlers
        # always pass a real key.
        assert key is not None
        return self.icdf(jax.random.uniform(key, sample_shape + self.batch_shape))
