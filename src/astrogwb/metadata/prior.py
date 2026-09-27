"""A prior over one hyperparameter, as a serializable record.

The wire format is the one every paper-config ``[priors]`` entry has always used::

    {"dist": "<numpyro.distributions class name>", "kwargs": {...}}

It lives here rather than in the paper's config layer because an artifact now
records it: a spectral-density draw names, per hyperparameter, either the value
it was fixed at or the prior it was sampled from, and that record is part of
the draw's cache key. Keeping the prior as data -- not as a registered model --
is what lets an edited bound re-key the artifact without a version bump.

The class is looked up on :mod:`numpyro.distributions` by name, so adding a
distribution needs no registry entry here. numpyro is imported inside
:meth:`PriorSpec.build` and :meth:`PriorSpec.from_distribution` only, so this
module keeps :mod:`astrogwb.metadata` free of JAX. Construction wraps plain
Python floats and evaluates no JAX op, so building a prior does not initialize
the XLA backend either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

__all__ = ["PriorSpec"]


class PriorSpec(BaseModel):
    """One numpyro distribution, named by class and constructor kwargs.

    There is deliberately no positional ``args`` form:
    :meth:`from_distribution` can only ever emit kwargs, so a second spelling
    would make a round trip non-canonical -- and a non-canonical record hashes
    to a second key for the same prior.
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
                f"missing required key(s) {missing} for a {self.dist} prior"
            )
        unexpected = sorted(set(self.kwargs) - expected)
        if unexpected:
            raise ValueError(
                f"Extra inputs are not permitted for a {self.dist} prior: {unexpected}"
            )
        return cls(**{key: float(value) for key, value in self.kwargs.items()})

    @classmethod
    def from_distribution(cls, prior: Distribution) -> Self:
        """The spec a live distribution was built from.

        The constructor keyword names are recovered from the class's own
        ``arg_constraints``, so this is correct for any distribution without a
        branch per type. A spec-built distribution holds plain Python floats,
        so ``float(...)`` never touches JAX.
        """
        import numpyro.distributions as dist

        if not isinstance(prior, dist.Distribution):
            raise TypeError(
                f"cannot serialize {type(prior).__name__!r} as a prior spec"
            )
        distribution = type(prior)
        return cls(
            dist=distribution.__name__,
            kwargs={
                key: float(getattr(prior, key)) for key in distribution.arg_constraints
            },
        )
