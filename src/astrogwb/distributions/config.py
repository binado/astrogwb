"""Serializable configuration for constructing NumPyro distributions.

The wire format, as in every paper-config ``[priors]`` entry::

    {"dist": "<numpyro.distributions class name>", "kwargs": {...}}

The class is looked up on :mod:`numpyro.distributions` by name. NumPyro is
imported lazily, and construction evaluates no JAX operation, so building a
config does not initialize the XLA backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpyro.distributions as dist
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

__all__ = ["DistributionConfig"]


class DistributionConfig(BaseModel):
    """One NumPyro distribution, named by class and constructor kwargs.

    There is no positional ``args`` form: :meth:`from_distribution` emits only
    kwargs, and a second spelling would hash one distribution to two keys.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    dist: str
    kwargs: dict[str, float]

    def build(self) -> Distribution:
        """Materialize the live distribution.

        Every failure raises ``ValueError``: this runs under a pydantic
        ``BeforeValidator``, which converts only ``ValueError`` and
        ``AssertionError`` into a ``ValidationError``.
        """

        cls = getattr(dist, self.dist, None)
        # Restrict a config-supplied `getattr` to distribution classes.
        if not (isinstance(cls, type) and issubclass(cls, dist.Distribution)):
            raise ValueError(f"{self.dist!r} is not a numpyro distribution")  # noqa: TRY004

        expected = set(cls.arg_constraints)
        missing = sorted(expected - set(self.kwargs))
        if missing:
            raise ValueError(
                f"missing required key(s) {missing} for a {self.dist} distribution"
            )
        unexpected = sorted(set(self.kwargs) - expected)
        if unexpected:
            raise ValueError(
                f"Extra inputs are not permitted for a {self.dist} distribution: {unexpected}"
            )
        return cls(**{key: float(value) for key, value in self.kwargs.items()})

    @classmethod
    def from_distribution(cls, distribution: Distribution) -> Self:
        """Return the config a live distribution was built from.

        Constructor keyword names are the class's ``arg_constraints``.
        """

        if not isinstance(distribution, dist.Distribution):
            raise TypeError(
                f"cannot serialize {type(distribution).__name__!r} as a distribution config"
            )
        distribution_type = type(distribution)
        return cls(
            dist=distribution_type.__name__,
            kwargs={
                key: float(getattr(distribution, key))
                for key in distribution_type.arg_constraints
            },
        )
