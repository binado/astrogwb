r"""Numerical marginalization of a multiplicative amplitude direction.

Under the per-frequency Gaussian likelihood used by
:func:`~astrogwb.inference.models.gaussian_gwb_model.gwb_spectral_density_model`,
one parameter can enter the predicted spectrum as a pure multiplicative
factor,

.. math:: \boldsymbol{\mu}(A, \theta) = A\, \mathbf{m}(\theta)

with :math:`\mathbf{m}(\theta)` the *template* -- the spectrum evaluated at the
point where the dimensionless amplitude :math:`A` equals one -- and
:math:`A = T(\varphi)` a monotone map from the physical parameter
:math:`\varphi` anchored so that :math:`T(\varphi_{\mathrm{fid}}) = 1`. This
module knows nothing about :math:`\varphi`: it integrates over :math:`A` itself,
under the pushforward prior :math:`\pi_A = T_{\#}\pi_\varphi` that
:func:`amplitude_prior` builds. The marginal likelihood is the same integral
either way, and a caller recovers :math:`\varphi` from amplitude draws with
:math:`T^{-1}`. The predicted spectrum factorizes into two independently-scaling
pieces, a total merger rate and a mean energy flux (the importance-weighted
polarization-power contraction); see
:func:`~astrogwb.inference.models.gaussian_gwb_marginalized_amplitude.amplitude_H0_transform`
and
:func:`~astrogwb.inference.models.gaussian_gwb_marginalized_amplitude.amplitude_local_merger_rate_transform`
for the concrete maps for :math:`H_0` and ``local_merger_rate``. Define
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
best-fit residual. The log-integrand is

.. math::

    \ell(A) = \ln\pi_A(A) - \tfrac{1}{2}\bigl(\rho\,(A - \hat{A})\bigr)^2,

and the amplitude direction is integrated with a fixed Gauss-Legendre rule
on a window clipped to the prior support,

.. math::

    \ln Z = \ln p(d \mid \hat{A}) + \ln\!\int_{l}^{u}
        \exp\bigl(\ell(A)\bigr)\, dA,

where :math:`\ln p(d \mid \hat{A}) = \ln\mathcal{N}_d - R` is the Gaussian
log-likelihood at the MLE amplitude. Squaring
:math:`\rho(A - \hat{A})` rather than forming
:math:`\rho^2(A-\hat A)^2` avoids overflowing :math:`\rho^2` at very high SNR.

The window follows the likelihood, not the prior. The conditional posterior has
width :math:`\sigma_A = 1/\rho`, so the integral runs over
:math:`[\max(a, c - 7\sigma_A),\ \min(b, c + 7\sigma_A)]`, with :math:`[a, b]`
the prior support (see :func:`support_bounds`) and
:math:`c = \mathrm{clip}(\hat A, a, b)` the support point nearest the
likelihood peak. The truncated tails carry :math:`\lesssim 10^{-11}` of the
mass, and the rule is accurate to about :math:`10^{-12}` at every SNR because
the nodes always sample the Gaussian rather than the prior range.

A prior bound is a *limit of integration*, not a mask. Masking the integrand
at a bound makes the rule's error depend on where the bound falls between
nodes and gives the bound a zero derivative, so a gradient-based sampler
cannot see it; as a limit it enters the quadrature nodes and weights, and
``jax.grad`` differentiates through it.

The window is only a quadrature scheme: the support is the prior's, and
:meth:`AmplitudeConditional.log_prob` evaluates the analytic density at any
:math:`A`. :meth:`AmplitudeConditional.sample` tabulates the CDF on the same
window, so draws lie in the support and miss only the :math:`\lesssim 10^{-11}`
beyond :math:`7\sigma_A`.

The conditional posterior of :math:`A` given the two statistics is
exposed as :class:`AmplitudeConditional`, a NumPyro ``Distribution``: the
marginalization factor in the model is its :attr:`log_normalizer`, and
post-processing reconstructs :math:`A` -- and :math:`\varphi = T^{-1}(A)` -- by
drawing from it.

Everything broadcasts over leading batch dimensions and contracts over the
trailing node axis, so post-processing can feed ``(chain, draw)`` shaped
statistics directly.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform

from astrogwb.utils import cumulative_trapezoid, gauss_legendre_rule

#: Half-width of the integration window in units of :math:`\sigma_A = 1/\rho`.
HALF_WIDTH = 7.0
#: Gauss-Legendre order of :attr:`AmplitudeConditional.log_normalizer`.
_ORDER = 32
#: Nodes of the per-element grid that :meth:`AmplitudeConditional.icdf` inverts.
_SAMPLING_NODES = 1024
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = gauss_legendre_rule(_ORDER)


def amplitude_prior(
    prior: dist.Distribution, transform: Transform
) -> dist.TransformedDistribution:
    r"""The amplitude-space pushforward :math:`\pi_A = T_{\#}\pi_\varphi` of a prior.

    ``transform`` maps the physical parameter :math:`\varphi` to the
    dimensionless amplitude :math:`A`. It must be monotone and anchored at the
    template, :math:`T(\varphi_{\mathrm{fid}}) = 1`: the model evaluates the
    spectrum at the fiducial, and that spectrum is the :math:`A = 1` template.
    An unanchored ``transform`` silently rescales every inferred amplitude.

    The marginal likelihood is the same whether the integral runs over
    :math:`\varphi` under :math:`\pi_\varphi` or over :math:`A` under
    :math:`\pi_A`; the Jacobian is carried by the pushforward density. Amplitude
    draws map back with ``transform.inv``.

    Example
    -------
    For :math:`H_0` the amplitude is :math:`A = H_{0,\mathrm{fid}}/H_0`, a
    decreasing map::

        from numpyro.distributions.transforms import (
            AffineTransform, ComposeTransform, PowerTransform,
        )

        h0_fid = 70.0
        transform = ComposeTransform(
            [PowerTransform(-1.0), AffineTransform(0.0, h0_fid)]
        )  # transform(h0_fid) == 1
        prior = amplitude_prior(dist.Uniform(20.0, 140.0), transform)
        # ... run the model, draw A from `AmplitudeConditional(..., prior=prior)`
        h0 = transform.inv(amplitude)

    See :func:`~astrogwb.inference.models.gaussian_gwb_marginalized_amplitude.amplitude_H0_transform`
    for the packaged map.
    """
    return dist.TransformedDistribution(prior, transform)


def support_bounds(prior: dist.Distribution) -> tuple[jax.Array, jax.Array]:
    r"""The ``(lower, upper)`` bounds of ``prior``'s support, ``-inf``/``inf`` if open.

    A :class:`~numpyro.distributions.TransformedDistribution` (what
    :func:`amplitude_prior` returns) reports an unbounded support, so the bounds
    are read from the base distribution and pushed through the transforms; the
    pair is re-ordered because a decreasing map such as ``fid / H0`` swaps them.
    Each transform must send :math:`\pm\infty` to something sensible if the base
    is unbounded on that side, as an affine map does.
    """
    if isinstance(prior, dist.TransformedDistribution):
        lower, upper = support_bounds(prior.base_dist)
        for transform in prior.transforms:
            lower, upper = transform(lower), transform(upper)
        return jnp.minimum(lower, upper), jnp.maximum(lower, upper)
    return (
        jnp.asarray(getattr(prior.support, "lower_bound", -jnp.inf), dtype=float),
        jnp.asarray(getattr(prior.support, "upper_bound", jnp.inf), dtype=float),
    )


class AmplitudeConditional(dist.Distribution):
    r"""Conditional posterior of the amplitude given its sufficient statistics.

    Given the amplitude sufficient statistics :math:`\hat A` and :math:`\rho`
    published by
    :func:`~astrogwb.inference.models.gaussian_gwb_marginalized_amplitude.gwb_amplitude_marginalized_model`,
    this is the density

    .. math::

        p(A \mid d, \theta) \propto
        \pi_A(A)\,
        \exp\!\left[-\tfrac12\bigl(\rho\,(A - \hat A)\bigr)^2\right].

    The distribution owns the live prior rather than a precomputed tabulation
    of it, so nothing can go stale. The density above is evaluated analytically
    wherever it is asked for; the integration window (:meth:`_window`) enters
    only as the quadrature scheme:

    - :attr:`log_normalizer` -- :math:`\ln Z` of the conditional under a
      32-point Gauss-Legendre rule on the window; this *is* the
      marginalization factor the model adds to its ``numpyro.factor`` site.
    - :meth:`sample` / :meth:`icdf` -- inverse-transform draws of
      :math:`A` for post-processing reconstruction, from a CDF tabulated on
      the same window.

    The prior bounds are read once from :func:`support_bounds` and enter the
    window as limits of integration, so the normalizer is differentiable in
    them and in :math:`\hat A`; see the module docstring.

    The ``batch_shape`` is the broadcast of the two statistics' shapes, so a
    ``(chain, draw)`` posterior feeds in directly. Map draws back to the
    physical parameter with the inverse of the transform the prior was built
    from (see :func:`amplitude_prior`).

    .. warning::

        This distribution is meant for generative post-processing
        (``AmplitudeConditional(...).sample(key)``). Do **not**
        ``numpyro.sample`` it as a latent site inside a NUTS model without
        revisiting two things. Its ``support`` is a
        :class:`~numpyro.distributions.constraints.dependent_property`, which
        routes latent use through NumPyro's dynamic-support path; and because
        the support is the *prior's* -- unbounded for a ``Normal`` prior, and
        reported as ``Real()`` for a transformed one -- ``biject_to`` may be the
        identity and a proposal outside the window is normalized against an
        integral that never covered it.

    Parameters
    ----------
    amplitude_mle, template_optimal_snr:
        The amplitude sufficient statistics :math:`\hat A` and :math:`\rho`,
        broadcast against each other.
    prior:
        The amplitude-space prior :math:`\pi_A`, with :math:`A = 1` at the
        template; see :func:`amplitude_prior`. Defines the support, hence the
        window's clipping bounds.
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
        "lower",
        "upper",
    )

    def __init__(
        self,
        amplitude_mle: ArrayLike,
        template_optimal_snr: ArrayLike,
        *,
        prior: dist.Distribution,
        validate_args: bool | None = None,
    ) -> None:
        self.amplitude_mle = jnp.asarray(amplitude_mle)
        self.template_optimal_snr = jnp.asarray(template_optimal_snr)
        self.prior = prior
        self.lower, self.upper = support_bounds(prior)
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(amplitude_mle), jnp.shape(template_optimal_snr)
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """The prior's support -- mathematically what the conditional lives on.

        A transformed prior reports ``Real()`` here, wider than its true
        support; :meth:`log_prob` stays ``-inf`` off-support regardless, via the
        base prior. :meth:`sample` covers slightly less than the support,
        because the window stops at :math:`7\\sigma_A` from the peak.
        """
        prior_support = self.prior.support
        if prior_support is None:
            raise TypeError(
                f"{type(self.prior).__name__} declares no support, so the "
                "conditional has none either"
            )
        return prior_support

    def _window(self) -> tuple[jax.Array, jax.Array]:
        r"""The integration window ``(lo, hi)``, shaped like ``batch_shape``.

        :math:`\pm` :data:`HALF_WIDTH` :math:`\sigma_A` about the support point
        nearest :math:`\hat A`, clipped to the support. The one definition
        shared by the normalizer and the sampler, so they cannot drift apart.
        """
        center = jnp.clip(self.amplitude_mle, self.lower, self.upper)
        reach = HALF_WIDTH / self.template_optimal_snr
        lo = jnp.maximum(self.lower, center - reach)
        hi = jnp.minimum(self.upper, center + reach)
        return jnp.broadcast_to(lo, self.batch_shape), jnp.broadcast_to(
            hi, self.batch_shape
        )

    def _log_density(
        self,
        amplitude: jax.Array,
        amplitude_mle: jax.Array,
        template_optimal_snr: jax.Array,
    ) -> jax.Array:
        r"""Unnormalized :math:`\ell(A)` at arbitrary :math:`A`.

        The single implementation behind the normalizer, the density, and the
        inverse-CDF draw -- which is what keeps them from drifting apart. The
        statistics are passed in rather than read off ``self`` so the caller
        controls broadcasting against the trailing node axis.
        """
        scaled_residual = template_optimal_snr * (amplitude - amplitude_mle)
        log_prior = jnp.asarray(self.prior.log_prob(amplitude))
        return log_prior - 0.5 * scaled_residual**2

    @property
    def log_normalizer(self) -> jax.Array:
        r""":math:`\ln Z` of the conditional -- the marginalization factor itself.

        :math:`\ln\int\exp(\ell)\,dA` by Gauss-Legendre on :meth:`_window`. The
        amplitude-marginalized model adds exactly this to the log-likelihood at
        the MLE: the factor *is* the normalizing constant of the conditional
        that post-processing later samples. The nodes lie inside the support,
        so the prior is never evaluated off it.
        """
        lo, hi = self._window()
        half = 0.5 * (hi - lo)
        nodes = lo[..., None] + half[..., None] * (1.0 + _LEGENDRE_NODES)
        log_integrand = self._log_density(
            nodes,
            self.amplitude_mle[..., None],
            self.template_optimal_snr[..., None],
        )
        return jnp.log(half) + logsumexp(
            log_integrand + jnp.log(_LEGENDRE_WEIGHTS), axis=-1
        )

    def log_prob(
        self, value: ArrayLike, intermediates: list[Any] | None = None
    ) -> jax.Array:
        """Exact log density, evaluated analytically off the window.

        The normalizing constant is the Gauss-Legendre integral over the
        window, so :meth:`log_prob` integrates to 1 up to the :math:`7\\sigma_A`
        truncation and quadrature error. ``intermediates`` is accepted for
        signature compatibility with :class:`numpyro.distributions.Distribution`;
        nothing here produces or replays sampling intermediates.
        """
        value = jnp.asarray(value)
        log_density = (
            self._log_density(value, self.amplitude_mle, self.template_optimal_snr)
            - self.log_normalizer
        )
        # NumPyro's `Uniform.log_prob` returns its constant density everywhere
        # rather than -inf off-support, so the mask -- not the prior term -- is
        # what keeps `log_prob` consistent with `support`. A transformed prior
        # reports `Real()` support, so there the mask is vacuous and the
        # `-inf` comes from the base prior's log-prob through the pushforward.
        return jnp.where(self.support.check(value), log_density, -jnp.inf)

    def icdf(self, q: ArrayLike) -> jax.Array:
        r"""Inverse CDF by linear-in-CDF inversion on a trapezoid CDF.

        The CDF is the cumulative trapezoid of the integrand on a uniform
        grid of :data:`_SAMPLING_NODES` points spanning :meth:`_window`, one
        grid per batch element. It agrees with :meth:`log_prob` at node
        resolution and differs sub-cell. Draws lose the
        :math:`\lesssim 10^{-11}` of the mass beyond the window.

        Parameters
        ----------
        q:
            Quantiles in ``[0, 1]``, shaped ``sample_shape + batch_shape``.

        Returns
        -------
        jax.Array
            :math:`A` values, within the window.
        """
        q = jnp.asarray(q)
        lo, hi = self._window()
        grid = lo[..., None] + (hi - lo)[..., None] * jnp.linspace(
            0.0, 1.0, _SAMPLING_NODES
        )
        log_integrand = self._log_density(
            grid,
            self.amplitude_mle[..., None],
            self.template_optimal_snr[..., None],
        )
        shifted = jnp.exp(
            log_integrand - jnp.max(log_integrand, axis=-1, keepdims=True)
        )
        cdf = cumulative_trapezoid(shifted, grid)
        cdf = cdf / cdf[..., -1:]

        shape = jnp.shape(q) + (_SAMPLING_NODES,)
        cdf = jnp.broadcast_to(cdf, shape)
        grid = jnp.broadcast_to(grid, shape)
        idx = jnp.clip(jnp.sum(cdf < q[..., None], axis=-1), 1, _SAMPLING_NODES - 1)[
            ..., None
        ]

        cdf_hi = jnp.take_along_axis(cdf, idx, axis=-1)[..., 0]
        cdf_lo = jnp.take_along_axis(cdf, idx - 1, axis=-1)[..., 0]
        grid_hi = jnp.take_along_axis(grid, idx, axis=-1)[..., 0]
        grid_lo = jnp.take_along_axis(grid, idx - 1, axis=-1)[..., 0]

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
