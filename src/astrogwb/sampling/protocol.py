"""Callable boundary between spectral predictions and sampling models."""

from collections.abc import Mapping
from typing import Protocol

import jax
from jax.typing import ArrayLike


class SpectralDensityFn(Protocol):
    """Predict a spectrum of shape ``(F,)`` and optional diagnostics.

    An analytic calculator may return ``{}`` for diagnostics. Diagnostic keys
    and shapes must stay stable during JAX tracing; keys must not collide with
    prior sites or likelihood-owned sites. Values are recorded unchanged as
    NumPyro deterministics by the sampling model.
    """

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]: ...


def with_renamed_diagnostics(
    fn: SpectralDensityFn, names: Mapping[str, str]
) -> SpectralDensityFn:
    """Return ``fn`` with selected diagnostic keys renamed.

    ``names`` maps a key the wrapped function returns to the name it should be
    published under, leaving every other extra untouched. The canonical use is
    an amplitude-marginalized run relabelling an estimator's
    ``total_merger_rate`` as ``template_merger_rate``, so the trace never
    carries a template quantity under the name of a physical one.

    Raises ``ValueError`` if a source key is absent from the returned extras,
    or if a target name collides with a key that survives the rename. A silent
    no-op would publish a template rate as ``total_merger_rate`` and be
    indistinguishable from the physical rate downstream.
    """

    def evaluate(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
        prediction, extras = fn(params)
        missing = sorted(names.keys() - extras.keys())
        if missing:
            raise ValueError(f"spectrum diagnostics are missing {missing}")
        renamed = {name: value for name, value in extras.items() if name not in names}
        for source, target in names.items():
            if target in renamed:
                raise ValueError(
                    f"renaming diagnostic {source!r} to {target!r} would "
                    "collide with an existing diagnostic"
                )
            renamed[target] = extras[source]
        return prediction, renamed

    return evaluate


__all__ = ["SpectralDensityFn", "with_renamed_diagnostics"]
