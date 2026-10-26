"""Eager checks that a population fits a waveform generator.

Generation is trace-safe and trusts its inputs: under ``jax.jit`` the source
arrays are tracers, and a tracer cannot drive a Python exception. These checks
are how a caller earns that trust, once, before the population reaches a
compiled forward model.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpyro

from astrogwb.populations._types import PopulationModel
from astrogwb.waveform.generator.base import PolarizationPowerGenerator

__all__ = [
    "LUMINOSITY_DISTANCE",
    "require_luminosity_distance",
    "validate_source_model",
]

#: The source output naming the effective distance governing waveform
#: amplitude. Required in every source model's returned mapping; see
#: :mod:`astrogwb.gwb.importance`, which reads the same key.
LUMINOSITY_DISTANCE = "luminosity_distance"


def require_luminosity_distance(sources: Mapping[str, object]) -> None:
    """Raise unless the source mapping names the distance scaling the waveform."""
    if LUMINOSITY_DISTANCE not in sources:
        raise KeyError(
            f"source model must return {LUMINOSITY_DISTANCE!r}: it is the "
            "distance governing waveform amplitude"
        )


def validate_source_model(
    model: PopulationModel,
    *,
    generator: PolarizationPowerGenerator,
    rng_key: jax.Array | int,
) -> None:
    """Check a population against a generator's waveform family, eagerly.

    One eager draw of a single source, checked against the approximant,
    before handing the model to NUTS or :class:`~numpyro.infer.Predictive`.

    No waveform is evaluated, so this is cheap enough to call unconditionally.

    Raises:
        KeyError: If ``model`` does not return ``luminosity_distance``.
        ValueError: If a drawn value names a degree of freedom the generator's
            approximant does not carry -- a tidal population against an
            aligned-spin model, say, whose deformabilities would otherwise be
            silently ignored.
    """
    with numpyro.handlers.seed(rng_seed=rng_key), numpyro.plate("events", 1):
        sources = dict(model())
    require_luminosity_distance(sources)
    generator.check_sources(sources)
