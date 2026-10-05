"""Serializable configuration for constructing NumPyro distributions.

The wire format is the one every paper-config ``[priors]`` entry has always
used::

    {"dist": "<numpyro.distributions class name>", "kwargs": {...}}

The class is looked up on :mod:`numpyro.distributions` by name, so adding a
distribution needs no registry entry here. NumPyro is imported inside
:meth:`DistributionConfig.build` and :meth:`DistributionConfig.from_distribution`
only. Construction wraps plain Python floats and evaluates no JAX operation, so
building a distribution config does not initialize the XLA backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

__all__ = ["DistributionConfig"]


class DistributionConfig(BaseModel):
    """One NumPyro distribution, named by class and constructor kwargs.

    There is deliberately no positional ``args`` form:
    :meth:`from_distribution` can only ever emit kwargs, so a second spelling
    would make a round trip non-canonical -- and a non-canonical record hashes
    to a second key for the same distribution.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    dist: str
    kwargs: dict[str, float]

    def build(self) -> Distribution:
        """Materialize the live distribution.

        Every failure raises ``ValueError``, even where the fault is a wrong
        name rather than a wrong value: this runs under the paper config's
        pydantic ``BeforeValidator``, and pydantic converts only ``ValueError``
        and ``AssertionError`` into a ``ValidationError``.
        """
        import numpyro.distributions as dist

        cls = getattr(dist, self.dist, None)
        # `getattr` on a config-supplied string: the guard is what keeps it to
        # distribution classes rather than any attribute the module exposes.
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

        Constructor keyword names are recovered from the class's own
        ``arg_constraints``, so this works for any distribution without a
        branch per type. A config-built distribution holds plain Python floats,
        so ``float(...)`` never touches JAX.
        """
        import numpyro.distributions as dist

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
