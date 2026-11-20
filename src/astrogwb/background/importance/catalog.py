"""Reference catalogs of compact-binary waveform intensities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from pydantic import BaseModel

from astrogwb.cosmology import gw_distance_ratio, luminosity_distance
from astrogwb.populations._types import Parameters
from astrogwb.waveform.base import WaveformGenerator

__all__ = ["ReferenceCatalog"]


#: The inclination site, fixed rather than drawn.
_INCLINATION_SITE = "inclination"

#: Reference waveforms are face-on.
_REFERENCE_INCLINATION = 0.0

# Analytic averaging and redshift rescaling require aligned-spin,
# quasi-circular dominant-quadrupole waveforms, excluding higher modes.
_REFERENCE_APPROXIMANTS = frozenset(
    {
        "AnalyticInspiral",
        "TaylorF2",
        "IMRPhenomD",
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv3",
    }
)


def _require_quadrupole_waveform(generator: WaveformGenerator) -> None:
    """Restrict analytic inclination conversion to supported quadrupole models."""
    if generator.metadata.approximant not in _REFERENCE_APPROXIMANTS:
        raise ValueError(
            "reference integration requires a supported dominant-quadrupole, "
            "aligned-spin approximant; higher modes and precession are unsupported"
        )


#: The luminosity distance declared by a source model.
_LUMINOSITY_DISTANCE = "luminosity_distance"

#: The redshift site.
_REDSHIFT = "redshift"


@dataclass(frozen=True)
class ReferenceCatalog:
    """Waveform intensities at one distance and redshift.

    Reference waveforms are face-on, quasi-circular, and aligned-spin with
    only the dominant quadrupole mode. Intrinsic columns exclude redshift,
    luminosity distance, and inclination.

    Attributes
    ----------
    frequencies
        Positive, increasing reference frequency grid in Hz, shape
        ``(F_ref,)``. Cover the redshifted query frequencies through the
        waveform's nonzero support.
    intensity
        Sum of the squared polarization amplitudes, shape ``(F_ref, N)``.
    intrinsic
        Source-frame intrinsic columns, in intensity column order.
    proposal_log_prob
        Finite joint log density of each draw under the proposal, shape
        ``(N,)``. Use the same variables and measure as the target log density.
    reference_redshift
        Redshift at which all reference waveforms were generated.
    reference_distance
        Positive gravitational-wave luminosity distance in Mpc used for the
        waveforms.
    """

    frequencies: ArrayLike
    intensity: ArrayLike
    intrinsic: Mapping[str, ArrayLike]
    proposal_log_prob: ArrayLike
    reference_redshift: float
    reference_distance: float

    @classmethod
    def from_samples(
        cls,
        generator: WaveformGenerator,
        samples: Mapping[str, ArrayLike],
        *,
        intrinsic_sites: Sequence[str],
        proposal_log_prob: ArrayLike,
        reference_redshift: float,
        parameters: Parameters,
        chunk_size: int | None = None,
    ) -> ReferenceCatalog:
        """Place intrinsic ``samples`` at a reference redshift and generate their intensity.

        ``samples`` holds one ``(N,)`` column per intrinsic site, drawn by the
        caller from the proposal. A copy is made with face-on inclination, redshift set to ``reference_redshift`` and the
        luminosity distance to the cosmological one at that redshift times the
        modified-propagation ratio when ``xi_0`` is in ``parameters``. No
        extrinsic parameters are used. ``chunk_size`` is
        :meth:`~astrogwb.waveform.WaveformGenerator.intensity`'s.

        Parameters
        ----------
        generator
            Waveform generator. Its approximant must be a supported
            dominant-quadrupole, aligned-spin one.
        samples
            Source columns by name, each ``(N,)``.
        intrinsic_sites
            Names of the columns of ``samples`` the importance weights are
            evaluated on. Must exclude redshift, luminosity distance and
            inclination.
        proposal_log_prob
            Joint log density of each draw under the proposal, shape ``(N,)``,
            over ``intrinsic_sites``.
        reference_redshift
            Redshift all waveforms are generated at.
        parameters
            Must carry ``H0`` and ``Omega_m``; ``xi_0`` and ``xi_n`` are optional.
        chunk_size
            Number of sources per chunk.

        Raises
        ------
        ValueError
            If the approximant is unsupported, ``intrinsic_sites`` names a
            reserved or missing column, or ``proposal_log_prob`` is not ``(N,)``.
        """
        # Before the draw: ripplegw enables x64 on import, but only once the
        # generator runs, after the sources are sampled.
        jax.config.update("jax_enable_x64", True)
        _require_quadrupole_waveform(generator)
        reserved = {_INCLINATION_SITE, _LUMINOSITY_DISTANCE, _REDSHIFT}
        if reserved & set(intrinsic_sites):
            raise ValueError(
                f"intrinsic_sites must exclude {sorted(reserved)}; "
                f"got {sorted(reserved & set(intrinsic_sites))}"
            )
        columns = {name: np.asarray(values) for name, values in samples.items()}
        if missing := set(intrinsic_sites) - columns.keys():
            raise ValueError(f"intrinsic_sites missing from samples: {sorted(missing)}")
        num_samples = len(next(iter(columns.values())))
        log_prob = np.asarray(proposal_log_prob)
        if log_prob.shape != (num_samples,):
            raise ValueError(
                f"proposal_log_prob must have shape ({num_samples},), got {log_prob.shape}"
            )
        distance = float(
            luminosity_distance(
                reference_redshift, parameters["H0"], parameters["Omega_m"]
            )
            * gw_distance_ratio(reference_redshift, parameters)
        )
        columns[_INCLINATION_SITE] = np.full(num_samples, _REFERENCE_INCLINATION)
        columns[_REDSHIFT] = np.full(num_samples, reference_redshift)
        columns[_LUMINOSITY_DISTANCE] = np.full(num_samples, distance)
        generator.check_sources(columns)
        # Fuse eager waveform generation to avoid retaining its intermediates.
        generate = jax.jit(partial(generator.intensity, chunk_size=chunk_size))
        return cls(
            frequencies=np.asarray(generator.frequencies),
            intensity=np.asarray(generate(columns)).T,
            intrinsic={name: columns[name] for name in intrinsic_sites},
            proposal_log_prob=log_prob,
            reference_redshift=reference_redshift,
            reference_distance=distance,
        )

    def take(self, n: int) -> ReferenceCatalog:
        """The first ``n`` draws, as a catalog.

        Parameters
        ----------
        n
            Number of draws to keep, at most the catalog's.

        Returns
        -------
        ReferenceCatalog
            A catalog whose per-draw arrays (``intensity``, ``intrinsic`` and
            ``proposal_log_prob``) are cut to ``n`` draws.

        Raises
        ------
        ValueError
            If ``n`` is negative or exceeds the number of draws.
        """
        size = int(np.shape(self.proposal_log_prob)[0])
        if not 0 <= n <= size:
            raise ValueError(f"cannot take {n} draws from a catalog of {size}")
        return replace(
            self,
            intensity=jnp.asarray(self.intensity)[:, :n],
            intrinsic={
                name: jnp.asarray(column)[:n] for name, column in self.intrinsic.items()
            },
            proposal_log_prob=jnp.asarray(self.proposal_log_prob)[:n],
        )

    def write(self, path: str | Path, metadata: BaseModel, **io_kwargs: Any) -> Path:
        """Write the catalog and the metadata that produced it, atomically.

        A thin wrapper of :func:`astrogwb.utils.io.write`, which needs the
        ``io`` extra.

        Parameters
        ----------
        path
            File path.
        metadata
            Metadata describing how the catalog was made.
        **io_kwargs
            Passed to :func:`astrogwb.utils.io.write` (``seed``, ``batch_size``).

        Returns
        -------
        pathlib.Path
            The file written.
        """
        from astrogwb.utils.io import write

        tree = {
            "frequencies": np.asarray(self.frequencies),
            "intensity": np.asarray(self.intensity),
            "intrinsic": {
                name: np.asarray(column) for name, column in self.intrinsic.items()
            },
            "proposal_log_prob": np.asarray(self.proposal_log_prob),
            "reference_redshift": np.float64(self.reference_redshift),
            "reference_distance": np.float64(self.reference_distance),
        }
        return write(path, tree, metadata, **io_kwargs)

    @classmethod
    def read[M: BaseModel](
        cls, path: str | Path, metadata_type: type[M]
    ) -> tuple[ReferenceCatalog, M]:
        """Read a catalog written by :meth:`write`.

        Parameters
        ----------
        path
            File path.
        metadata_type
            Pydantic model that validates the recorded metadata.

        Returns
        -------
        ReferenceCatalog
            The catalog, with NumPy array fields.
        M
            The metadata it was written with.
        """
        from astrogwb.utils.io import load

        tree, metadata, _ = load(path, metadata_type)
        data: dict[str, Any] = dict(tree)
        catalog = cls(
            frequencies=data["frequencies"],
            intensity=data["intensity"],
            intrinsic=dict(data["intrinsic"]),
            proposal_log_prob=data["proposal_log_prob"],
            reference_redshift=float(data["reference_redshift"]),
            reference_distance=float(data["reference_distance"]),
        )
        return catalog, metadata
