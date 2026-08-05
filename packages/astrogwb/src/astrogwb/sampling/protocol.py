"""Protocols for amplitude-marginalization callbacks.

Mirrors :mod:`astrogwb.importance.protocol`: a callback contract expressed as a
:class:`~typing.Protocol` so that callers can supply either of two
implementations -- analytic (:func:`~astrogwb.sampling.amplitude.amplitude_log_evidence`)
or numerical quadrature
(:func:`~astrogwb.sampling.amplitude_quadrature.quadrature_log_evidence`) --
to :func:`~astrogwb.sampling.models.amplitude_marginalized_model` without the
model needing to know which one it got.
"""

from __future__ import annotations

from typing import Protocol

import jax


class AmplitudeScalingFn(Protocol):
    """Map the marginalized parameter to the multiplicative amplitude :math:`A = f(\\varphi)`."""

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


class LogEvidenceFn(Protocol):
    """Marginalize the amplitude direction out of the Gaussian likelihood.

    Satisfied by :func:`~astrogwb.sampling.amplitude.amplitude_log_evidence`
    and :func:`~astrogwb.sampling.amplitude_quadrature.quadrature_log_evidence`
    once their extra keyword-only argument (``prior`` and ``quadrature``
    respectively) is bound, e.g. via :func:`functools.partial`.
    """

    def __call__(
        self,
        amplitude_ml: jax.Array,
        template_optimal_snr: jax.Array,
        *,
        residual: jax.Array,
        log_norm: jax.Array,
    ) -> jax.Array: ...
