r"""Numerical marginalization of a multiplicative amplitude direction.

Under the per-frequency Gaussian likelihood used by
:func:`~astrogwb.sampling.models.spectral_density_model`, one parameter can
enter the predicted spectrum as a pure multiplicative factor,

.. math:: \boldsymbol{\mu}(\varphi, \theta) = f(\varphi)\, \mathbf{m}(\theta)

with :math:`\mathbf{m}(\theta)` the *template* -- the spectrum evaluated at a
fixed reference value of the marginalized parameter -- and :math:`f` an
arbitrary scaling from the physical parameter :math:`\varphi` to the
dimensionless multiplicative amplitude :math:`A = f(\varphi)`. The predicted
spectrum factorizes into two independently-scaling pieces, a total merger
rate and a mean energy flux (the importance-weighted polarization-power
contraction), so :math:`f = g_R \cdot g_F`; see
:func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.amplitude_scalings`
for the concrete pair of exponents for :math:`H_0` and
``local_merger_rate``. Define the noise-weighted inner product
:math:`(x|y) = \sum_i x_i y_i / \sigma_i^2`. Then

.. math::

    \hat{A} = \frac{(d|m)}{(m|m)}, \qquad \rho = \sqrt{(m|m)},

are the maximum-likelihood amplitude and the template optimal SNR, and
completing the square in :math:`A` gives

.. math::

    -\tfrac{1}{2}\sum_i \left(\frac{d_i - A\, m_i}{\sigma_i}\right)^2
    = -R - \tfrac{1}{2}\rho^2 (A - \hat{A})^2,

with :math:`R = \tfrac{1}{2}\sum_i((d_i - \hat{A}m_i)/\sigma_i)^2` the
best-fit residual. This module marginalizes :math:`\varphi` numerically on a
fixed 1D grid under the caller's actual prior :math:`\pi(\varphi)`, rather
than requiring the prior to be stated on :math:`A` itself. The log-integrand
on the grid is

.. math::

    \ell(\varphi) = \ln\pi(\varphi) - \tfrac{1}{2}\bigl(\rho\bigl(f(\varphi) - \hat{A}\bigr)\bigr)^2,

and the amplitude direction is integrated with the trapezoid rule after a
stable max-shift:

.. math::

    \ln Z = \ln p(d \mid \hat{A})
        + \ell_{\max}
        + \ln\!\int \exp\bigl(\ell(\varphi) - \ell_{\max}\bigr)\, d\varphi,

where :math:`\ln p(d \mid \hat{A}) = \ln\mathcal{N}_d - R` is the Gaussian
log-likelihood at the MLE amplitude. The caller must supply a prior density
that is already normalized on the grid
(:math:`\int \pi(\varphi)\, d\varphi \approx 1` under the same trapezoid
rule); this module does not renormalize. Squaring
:math:`\rho(f(\varphi) - \hat{A})` rather than forming
:math:`\rho^2(f-\hat A)^2` avoids overflowing :math:`\rho^2` at very high
SNR.

"Exact up to quadrature error" only holds if the grid resolves the conditional
posterior, whose width in :math:`\varphi` is :math:`\sigma_A/|f'(\varphi)|`.
No quadrature rule rescues a Gaussian bump spanning three nodes, so grid
adequacy must be checked with :attr:`AmplitudeConditional.effective_nodes`,
not assumed.

The conditional posterior of :math:`\varphi` given the two statistics is
exposed as :class:`AmplitudeConditional`, a NumPyro ``Distribution``: the
marginalization factor in the model is its :attr:`log_normalizer`, and
post-processing reconstructs :math:`\varphi` by drawing from it. Two
approximations of the same grid density live in one object on purpose:
:meth:`~AmplitudeConditional.log_prob` is the piecewise-linear interpolation
of the density the normalizer integrates (the trapezoid rule is exact for
it), while :meth:`~AmplitudeConditional.icdf` inverts the trapezoid CDF, a
piecewise-constant density. They agree at node resolution and differ
sub-cell; the ``icdf`` path is what the reconstruction draws use.

Everything broadcasts over leading batch dimensions and contracts over the
trailing grid axis, so post-processing can feed ``(chain, draw)`` shaped
statistics directly.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import constraints

from astrogwb.importance.diagnostics import relative_ess


class MergerRateAmplitudeFn(Protocol):
    """Ratio of the total merger rate at :math:`\\varphi` to its value at the
    fiducial template, :math:`g_R(\\varphi)`."""

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


class MeanEnergyFluxAmplitudeFn(Protocol):
    """Ratio of the importance-weighted polarization-power contraction at
    :math:`\\varphi` to its value at the fiducial template,
    :math:`g_F(\\varphi)`."""

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


class AmplitudeQuadrature(NamedTuple):
    """Precomputed grid, scalings, and prior for numerical marginalization.

    Built once by :func:`make_amplitude_quadrature` and then reused every MCMC
    step; a plain ``NamedTuple`` keeps it a JAX pytree without needing to be
    stored as model state, since the callables that built it are only
    consumed at construction time.
    """

    grid: jax.Array
    """``(K,)`` values of the marginalized parameter :math:`\\varphi`."""

    merger_rate_amplitude: jax.Array
    """``(K,)`` :math:`g_R(\\varphi_k)`, the merger-rate ratio at each node."""

    mean_energy_flux_amplitude: jax.Array
    """``(K,)`` :math:`g_F(\\varphi_k)`, the mean-energy-flux ratio at each node."""

    log_prior: jax.Array
    """``(K,)`` :math:`\\ln\\pi(\\varphi_k)`, the caller's prior density on the grid."""

    @property
    def amplitude(self) -> jax.Array:
        """``(K,)`` :math:`f(\\varphi_k) = g_R(\\varphi_k) \\cdot g_F(\\varphi_k)`, by construction."""
        return self.merger_rate_amplitude * self.mean_energy_flux_amplitude


def make_amplitude_quadrature(
    *,
    grid: jax.Array,
    log_prior: jax.Array,
    merger_rate_amplitude: MergerRateAmplitudeFn,
    mean_energy_flux_amplitude: MeanEnergyFluxAmplitudeFn,
) -> AmplitudeQuadrature:
    """Build the fixed quadrature grid consumed by every MCMC step.

    Called once, before sampling starts, with concrete (non-traced) arrays --
    like :func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.make_merger_rate_and_log_weights_fn`,
    this is a factory, not something invoked inside ``jax.jit``.

    Parameters
    ----------
    grid:
        Strictly increasing 1D array of :math:`\\varphi` values, length at
        least 2.
    log_prior:
        :math:`\\ln\\pi(\\varphi_k)` at each grid node, same shape as ``grid``.
        Must already be normalized on the grid
        (:math:`\\int \\pi(\\varphi)\\, d\\varphi \\approx 1` under the
        trapezoid rule); typical source is a NumPyro
        ``Distribution.log_prob`` evaluated on a grid that covers the prior
        support. This factory does not renormalize.
    merger_rate_amplitude:
        Maps the grid to :math:`g_R(\\varphi_k)`, the merger-rate ratio.
    mean_energy_flux_amplitude:
        Maps the grid to :math:`g_F(\\varphi_k)`, the mean-energy-flux ratio.

    Raises
    ------
    ValueError
        If ``grid`` is not 1D, has fewer than 2 points, is not strictly
        increasing, or does not match the shape of ``log_prior``.
    """
    grid = jnp.asarray(grid)
    log_prior = jnp.asarray(log_prior)
    if grid.ndim != 1:
        raise ValueError(f"grid must be 1D, got shape {grid.shape}")
    if grid.shape[0] < 2:
        raise ValueError(f"grid must have at least 2 points, got {grid.shape[0]}")
    if log_prior.shape != grid.shape:
        raise ValueError(
            f"log_prior shape {log_prior.shape} must match grid shape {grid.shape}"
        )
    if not bool(jnp.all(jnp.diff(grid) > 0)):
        raise ValueError("grid must be strictly increasing")

    return AmplitudeQuadrature(
        grid=grid,
        merger_rate_amplitude=merger_rate_amplitude(grid),
        mean_energy_flux_amplitude=mean_energy_flux_amplitude(grid),
        log_prior=log_prior,
    )


def merger_rate_amplitude_at(
    marginalized_parameter: ArrayLike, *, quadrature: AmplitudeQuadrature
) -> jax.Array:
    """:math:`g_R(\\varphi)` at arbitrary :math:`\\varphi`, linearly interpolated on the quadrature grid.

    Self-consistent with :meth:`AmplitudeConditional.icdf`, which already
    returns a linear interpolant between adjacent grid nodes: the same
    piecewise-linear model is reused here to recover the physical merger rate
    in post-processing, :math:`R(\\varphi) = g_R(\\varphi) \\cdot R_{\\mathrm{fid}}`.
    """
    return jnp.interp(
        marginalized_parameter, quadrature.grid, quadrature.merger_rate_amplitude
    )


def _amplitude_log_integrand(
    amplitude_mle: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
) -> jax.Array:
    r"""Shared ``(..., K)`` log-integrand behind every ``AmplitudeConditional`` computation.

    Returns :math:`\ln\pi(\varphi_k) - \tfrac12[\rho(f_k - \hat A)]^2` -- the
    continuous density on the grid, before trapezoid integration. Routing the
    normalizer, the density evaluation, and the inverse-CDF draw through this
    one implementation is what keeps them from drifting apart.
    """
    scaled_residual = jnp.expand_dims(template_optimal_snr, -1) * (
        quadrature.amplitude - jnp.expand_dims(amplitude_mle, -1)
    )
    return quadrature.log_prior - 0.5 * scaled_residual**2


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
    r"""Conditional posterior of the marginalized parameter on the quadrature grid.

    Given the amplitude sufficient statistics :math:`\hat A` and :math:`\rho`
    published by
    :func:`~astrogwb.sampling.models.amplitude_marginalized_model`, this is
    the density

    .. math::

        p(\varphi \mid d, \theta) \propto
        \pi(\varphi)\,
        \exp\!\left[-\tfrac12\bigl(\rho\,(f(\varphi) - \hat A)\bigr)^2\right]

    tabulated on the fixed grid in ``quadrature``. One object owns everything
    derived from that integrand, so the pieces cannot drift apart:

    - :attr:`log_normalizer` -- :math:`\ln Z` of the conditional under the
      trapezoid rule; this *is* the marginalization factor the model adds to
      its ``numpyro.factor`` site.
    - :meth:`sample` / :meth:`icdf` -- inverse-transform draws of
      :math:`\varphi` for post-processing reconstruction.
    - :attr:`effective_nodes` -- grid-adequacy diagnostic.

    Two approximations of the grid density are exposed deliberately.
    :meth:`log_prob` is the log of the piecewise-*linear* interpolation of the
    density on the grid -- exactly what :attr:`log_normalizer` integrates, so
    the trapezoid rule integrates the normalized density to exactly 1.
    :meth:`icdf` inverts the cumulative trapezoid of that density, i.e. a
    piecewise-*constant* density; the two agree at node resolution and differ
    sub-cell. ``icdf`` is byte-for-byte the pre-Distribution implementation,
    so reconstructed draws are bit-identical across the refactor.

    The ``batch_shape`` is the broadcast of the two statistics' shapes, so a
    ``(chain, draw)`` posterior feeds in directly. This distribution is meant
    for :class:`~numpyro.infer.Predictive` (generative-only use in
    :func:`~astrogwb.sampling.models.amplitude_reconstruction_model`); its
    ``support`` is a
    :class:`~numpyro.distributions.constraints.dependent_property`, which
    routes any use as a *latent* site through NumPyro's dynamic-support path
    -- do not ``numpyro.sample`` it inside a NUTS model without revisiting
    that choice.

    Parameters
    ----------
    amplitude_mle, template_optimal_snr:
        The amplitude sufficient statistics :math:`\hat A` and :math:`\rho`,
        broadcast against each other.
    quadrature:
        Precomputed grid from :func:`make_amplitude_quadrature`.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`.
    """

    # Plain dict like every NumPyro distribution: annotating ClassVar would
    # violate ty's override rules against Distribution's instance annotation.
    arg_constraints = {  # noqa: RUF012
        "amplitude_mle": constraints.real,
        "template_optimal_snr": constraints.positive,
    }
    pytree_data_fields = ("amplitude_mle", "template_optimal_snr", "quadrature")

    def __init__(
        self,
        amplitude_mle: jax.Array,
        template_optimal_snr: jax.Array,
        *,
        quadrature: AmplitudeQuadrature,
        validate_args: bool | None = None,
    ) -> None:
        self.amplitude_mle = amplitude_mle
        self.template_optimal_snr = template_optimal_snr
        self.quadrature = quadrature
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(amplitude_mle), jnp.shape(template_optimal_snr)
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """The closed grid interval ``[grid[0], grid[-1]]``."""
        return constraints.interval(self.quadrature.grid[0], self.quadrature.grid[-1])

    @property
    def _log_integrand(self) -> jax.Array:
        """``batch_shape + (K,)`` log-integrand on the grid."""
        return _amplitude_log_integrand(
            self.amplitude_mle, self.template_optimal_snr, quadrature=self.quadrature
        )

    @property
    def log_normalizer(self) -> jax.Array:
        r""":math:`\ln Z` of the conditional -- the marginalization factor itself.

        Stable :math:`\ln\int\exp(\ell)\,d\varphi` over the grid via
        :func:`_log_trapezoid`. The amplitude-marginalized model adds exactly
        this to the log-likelihood at the MLE: the factor *is* the normalizing
        constant of the conditional that post-processing later samples.
        """
        return _log_trapezoid(self._log_integrand, self.quadrature.grid)

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
        return relative_ess(self._log_integrand) * self.quadrature.grid.shape[0]

    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """Log of the piecewise-linear density on the grid, normalized.

        ``jnp.interp`` only takes a 1D ``fp``, so the batched integrand is
        interpolated manually with the same linear rule. Values outside the
        grid interval return ``-inf``, matching :attr:`support`.
        """
        grid = self.quadrature.grid
        value = jnp.asarray(value)
        integrand = jnp.broadcast_to(
            jnp.exp(self._log_integrand), value.shape + grid.shape
        )
        idx = jnp.clip(
            jnp.searchsorted(grid, value, side="right"), 1, grid.shape[0] - 1
        )
        x_lo = grid[idx - 1]
        x_hi = grid[idx]
        y_lo = jnp.take_along_axis(integrand, (idx - 1)[..., None], axis=-1)[..., 0]
        y_hi = jnp.take_along_axis(integrand, idx[..., None], axis=-1)[..., 0]
        density = y_lo + (y_hi - y_lo) * ((value - x_lo) / (x_hi - x_lo))
        log_density = jnp.log(density) - self.log_normalizer
        in_support = (value >= grid[0]) & (value <= grid[-1])
        return jnp.where(in_support, log_density, -jnp.inf)

    def icdf(self, q: ArrayLike) -> ArrayLike:
        r"""Inverse CDF by linear-in-CDF inversion on the trapezoid CDF.

        The CDF is the cumulative trapezoid of the same integrand that
        :attr:`log_normalizer` integrates, so draws follow precisely the
        density that was marginalized. Kept byte-for-byte compatible with the
        pre-Distribution implementation: same shifted integrand, same CDF,
        same flat-plateau guard.

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
        cdf = _cumulative_trapezoid(shifted, self.quadrature.grid)
        cdf = cdf / cdf[..., -1:]

        grid = self.quadrature.grid
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
    ) -> ArrayLike:
        """Inverse-transform draw: one uniform per element, through :meth:`icdf`."""
        # `None` only exists to match the base-class signature; handlers
        # always pass a real key.
        assert key is not None
        return self.icdf(jax.random.uniform(key, sample_shape + self.batch_shape))
