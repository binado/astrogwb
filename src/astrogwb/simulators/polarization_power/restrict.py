"""Restricting a polarization-power draw to a redshift window.

A draw is a :class:`~astrogwb.simulators.polarization_power.PolarizationPowerData`
plus the :class:`~astrogwb.simulators.polarization_power.CatalogMetadata` that
describes the density that drew it. Dropping samples without narrowing the
recorded density would leave it normalized over a window the samples no longer
span, so the two halves are transformed together by :func:`restrict_redshift`.
"""

from __future__ import annotations

import numpy as np

from astrogwb.simulators.polarization_power.metadata import CatalogMetadata
from astrogwb.simulators.polarization_power.simulator import PolarizationPowerData

__all__ = ["REDSHIFT_SITE", "REDSHIFT_WINDOW_KWARGS", "restrict_redshift"]

#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

#: Construction kwargs a population model must take for a draw from it to
#: support :func:`restrict_redshift`. Narrowing the window changes the
#: *normalization* of the generating density, so the arrays and the model kwargs
#: have to move together or the recorded density stops describing the samples.
REDSHIFT_WINDOW_KWARGS = ("minimum_redshift", "maximum_redshift")


def restrict_redshift(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    minimum_redshift: float,
    maximum_redshift: float,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """Restrict a draw to sources inside a redshift window, narrowing the population.

    Both halves move together, which is the whole reason this is one function:
    dropping samples without narrowing the generating model would leave the
    recorded density normalized over a window the samples no longer span, and
    every importance weight would be off by that normalization. Draws truncated
    to a sub-window follow the same law as draws made directly from it, so only
    the support changes.

    Returns a new ``(data, metadata)`` pair; the inputs are untouched. The
    returned metadata records the narrowed window and the kept sample count, so
    it no longer names a cached draw -- it is derived from one.

    Raises
    ------
    ValueError
        If the population takes no redshift window, the window is not inside
        the generation support, or no sample falls in it.
    """
    population = metadata.population
    model_kwargs = population.model_kwargs
    missing = [name for name in REDSHIFT_WINDOW_KWARGS if name not in model_kwargs]
    if missing:
        raise ValueError(
            f"population {population.model_name!r} takes no "
            f"{missing} construction setting(s), so its redshift window cannot "
            "be narrowed"
        )
    generated_min = float(model_kwargs["minimum_redshift"])
    generated_max = float(model_kwargs["maximum_redshift"])
    if not generated_min <= minimum_redshift < maximum_redshift <= generated_max:
        raise ValueError(
            f"analysis redshift support [{minimum_redshift:.4g}, "
            f"{maximum_redshift:.4g}] must lie within the catalog generation "
            f"support [{generated_min:.4g}, {generated_max:.4g}]"
        )

    redshift = np.asarray(data["source_parameters"][REDSHIFT_SITE])
    keep = np.flatnonzero(
        (redshift >= minimum_redshift) & (redshift <= maximum_redshift)
    )
    if keep.size == 0:
        raise ValueError(
            f"catalog has no samples in the "
            f"redshift window [{minimum_redshift:.4g}, {maximum_redshift:.4g}]"
        )
    restricted = PolarizationPowerData(
        frequencies=data["frequencies"],
        polarization_power=np.asarray(data["polarization_power"])[:, keep],
        source_parameters={
            name: np.asarray(values)[keep]
            for name, values in data["source_parameters"].items()
        },
    )
    narrowed = metadata.model_copy(
        update={
            "population": population.with_model_kwargs(
                minimum_redshift=float(minimum_redshift),
                maximum_redshift=float(maximum_redshift),
            ),
            "num_samples": int(keep.size),
        }
    )
    return restricted, narrowed
