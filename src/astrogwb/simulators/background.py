r"""The background stage: per-pair cross-power spectra of one realization.

Source :math:`i` reaches detector :math:`a` as
:math:`H_{a,i} = F^+_{a,i} \tilde h_{+,i} + F^\times_{a,i} \tilde h_{\times,i}`,
with antenna patterns of its sky position and polarization angle, and leaves
in the pair :math:`(a, b)` the real cross-power

.. math::

    K_{ab,i} = \mathrm{Re}\left[e^{i\phi_{ab,i}} H_{a,i}^* H_{b,i}\right],

:math:`\phi_{ab,i}` the inter-site delay phase. It is computed through the
Stokes contraction :func:`~astrogwb.detector.stokes_cross_power` of
:func:`~astrogwb.waveform.polarization_stokes`, which is the same product with
every polarization term kept. Sources are summed without cross terms between
them (no source interference), :math:`D_{ab} = \sum_i K_{ab,i}`, and normalized
by the observation time :math:`T` as

.. math::

    Y_{ab} = \frac{5\, D_{ab}}{T}.

Over an isotropic sky and uniform polarization angle
:math:`\langle K_{ab} \rangle = I \gamma_{ab} / 5`, with :math:`I` the
polarization power and :math:`\gamma_{ab}` the normalized isotropic overlap
reduction function, so for an isotropic, unpolarized population
:math:`E[Y_{ab}] = \gamma_{ab} S_h`. This is convention version
:data:`NORMALIZATION_VERSION`, recorded in every ``[background]`` table. The sum
is never divided by the realized count.

The background is deterministic given its inputs and carries no detector
noise. Detectors are static: the Earth's rotation during a signal is not
modelled. Pair spectra are the result; :func:`combine_pair_spectra` combines a
network's pairs into one optimally weighted spectrum, separately, so changing
the weights needs no new projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from astrogwb.simulators._io import atomic_h5
from astrogwb.utils.io import content_key, h5py_module

if TYPE_CHECKING:
    import jax

    from astrogwb.detector import Sensitivity
    from astrogwb.simulators.waveforms import WaveformChunks

__all__ = [
    "NORMALIZATION_VERSION",
    "BackgroundArtifact",
    "BackgroundSimulator",
    "DetectorConfiguration",
    "background_key",
    "background_table",
    "combine_pair_spectra",
    "union_configuration",
]

SCHEMA_VERSION = 1

#: :math:`\langle K_{ab} \rangle = I \gamma_{ab}` over this, the factor in
#: :math:`Y_{ab} = 5 D_{ab} / T`.
PAIR_RESPONSE_DENOMINATOR = 5.0

#: The version of the pair-spectrum convention :math:`E[Y_{ab}] = \gamma_{ab} S_h`.
NORMALIZATION_VERSION = 1

_NORMALIZATION = {
    "convention": "E[Y_ab] = gamma_ab * S_h",
    "pair_spectrum": "Y_ab = 5 * D_ab / T",
    "factor": PAIR_RESPONSE_DENOMINATOR,
    "version": NORMALIZATION_VERSION,
}


class DetectorConfiguration(BaseModel):
    """A detector network: site codes and, for combining pairs, noise curves.

    ``detectors`` are two or more site codes of
    :func:`~astrogwb.detector.resolve_detector`; their pairs :math:`a < b` are
    the background's pair axis. ``psd_references`` is the
    :class:`~astrogwb.detector.Sensitivity` reference of each, keyed by site
    code. The pair spectra do not depend on it, only the optimal weights of
    :func:`combine_pair_spectra` do, so it is not part of a background's key.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    detectors: Annotated[list[str], Field(min_length=2)]
    psd_references: dict[str, str] = {}

    def pairs(self) -> list[tuple[str, str]]:
        """The site codes of every pair :math:`a < b`, in the order of the pair axis."""
        return [
            (first, second)
            for index, first in enumerate(self.detectors)
            for second in self.detectors[index + 1 :]
        ]

    def sensitivities(self) -> dict[str, Sensitivity]:
        """The noise curve of every detector, keyed by site code.

        Raises
        ------
        KeyError
            If a detector has no ``psd_references`` entry.
        """
        from astrogwb.detector import Sensitivity

        missing = [name for name in self.detectors if name not in self.psd_references]
        if missing:
            raise KeyError(f"no psd_references for {missing}")
        return {name: Sensitivity(self.psd_references[name]) for name in self.detectors}


def union_configuration(
    networks: Sequence[DetectorConfiguration],
) -> DetectorConfiguration:
    """Every detector of ``networks`` once, in first-seen order.

    A site code shared by two networks must name the same detector and noise
    curve; this is trusted, not checked.
    """
    references: dict[str, str] = {}
    for network in networks:
        for name in network.detectors:
            references.setdefault(name, network.psd_references.get(name, ""))
    return DetectorConfiguration(
        detectors=list(references),
        psd_references={name: ref for name, ref in references.items() if ref},
    )


def background_key(waveform_key: str, detectors: DetectorConfiguration) -> str:
    """The content hash of the spectra of the waveforms ``waveform_key`` on ``detectors``.

    The noise curves are not part of it: the pair spectra do not depend on them.
    """
    return content_key(
        {
            "waveforms": waveform_key,
            "detectors": detectors.detectors,
            "normalization_version": NORMALIZATION_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
    )


def background_table(
    waveform_key: str, detectors: DetectorConfiguration
) -> dict[str, Any]:
    """The ``[background]`` manifest table of the spectra :func:`background_key` names."""
    return {
        "metadata": {"detectors": detectors.model_dump(mode="json")},
        "key": background_key(waveform_key, detectors),
        "normalization": _NORMALIZATION,
        "units": {"frequency": "Hz", "pair_spectral_density": "1 / Hz"},
    }


@dataclass(frozen=True)
class BackgroundArtifact:
    """The per-pair spectra of one realization, in memory or read from disk.

    ``pair_spectral_density`` is :math:`Y_{ab}` in :math:`\\mathrm{Hz}^{-1}`,
    ``(P, F)``, the pairs in the order of ``detectors.pairs()``. ``count``,
    ``total_merger_rate`` (per second) and ``hyperparameters`` are the
    realization's, carried over from its population. ``waveform_key`` names
    the polarizations the spectra were projected from.
    """

    detectors: DetectorConfiguration
    frequencies: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    observation_seconds: float
    count: int
    total_merger_rate: float
    hyperparameters: dict[str, float]
    waveform_key: str
    chunk_size: int | None = None

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """The pair axis, as site-code pairs."""
        return self.detectors.pairs()

    def key(self) -> str:
        """The content hash of these spectra: their waveforms and detectors."""
        return background_key(self.waveform_key, self.detectors)

    def pair_spectra(self, network: DetectorConfiguration) -> NDArray[np.float64]:
        """The rows of ``network``'s pairs, ``(P_network, F)``, in its own order.

        :math:`K_{ab} = K_{ba}`, so a pair is found whichever way round it is.

        Raises
        ------
        KeyError
            If a pair of ``network`` is not one of these spectra's.
        """
        rows = {frozenset(pair): index for index, pair in enumerate(self.pairs)}
        missing = [pair for pair in network.pairs() if frozenset(pair) not in rows]
        if missing:
            raise KeyError(f"pairs {missing} are not in this background")
        return self.pair_spectral_density[
            [rows[frozenset(pair)] for pair in network.pairs()]
        ]

    def table(self) -> dict[str, Any]:
        """The ``[background]`` manifest table."""
        return background_table(self.waveform_key, self.detectors)

    def write(self, path: str | Path) -> Path:
        """Write the spectra to the HDF5 file ``path``, atomically."""
        h5py = h5py_module()
        with atomic_h5(path) as file:
            file.create_dataset("frequencies", data=self.frequencies)
            file.create_dataset(
                "pairs", data=np.asarray(self.pairs, dtype=h5py.string_dtype())
            )
            file.create_dataset(
                "pair_spectral_density", data=self.pair_spectral_density
            )
        return Path(path)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        table: Mapping[str, Any],
        realization: Mapping[str, Any],
        waveform_key: str,
    ) -> Self:
        """Read the spectra at ``path`` with their manifest tables.

        ``table`` is the ``[background]`` table, ``realization`` the
        ``[realization]`` table and ``waveform_key`` the ``[waveforms]`` key.

        Raises
        ------
        ValueError
            If the table uses another normalization convention, records
            another key, or the spectra do not match it.
        """
        version = table["normalization"].get("version")
        if version != NORMALIZATION_VERSION:
            raise ValueError(
                f"{path} uses pair-spectrum normalization version {version}, "
                f"not {NORMALIZATION_VERSION}"
            )
        detectors = DetectorConfiguration.model_validate(table["metadata"]["detectors"])
        h5py = h5py_module()
        with h5py.File(path, "r") as file:
            frequencies = np.asarray(file["frequencies"][()], dtype=np.float64)
            pairs = [tuple(row) for row in file["pairs"].asstr()[()].tolist()]
            spectra = np.asarray(file["pair_spectral_density"][()], dtype=np.float64)
        if pairs != detectors.pairs() or spectra.shape != (
            len(pairs),
            frequencies.shape[0],
        ):
            raise ValueError(f"{path}: spectra do not match the manifest")
        artifact = cls(
            detectors=detectors,
            frequencies=frequencies,
            pair_spectral_density=spectra,
            observation_seconds=float(realization["observation_seconds"]),
            count=int(realization["count"]),
            total_merger_rate=float(realization["total_merger_rate"]),
            hyperparameters={
                name: float(value)
                for name, value in realization["hyperparameters"].items()
            },
            waveform_key=waveform_key,
        )
        if artifact.key() != table["key"]:
            raise ValueError(
                f"the manifest records background {table['key']}, but its "
                f"waveforms and detectors hash to {artifact.key()}"
            )
        return artifact


class BackgroundSimulator:
    """Projects polarizations onto a detector network and sums the pair cross-powers.

    ``simulator(waveforms)`` loops over ``waveforms.iter_chunks(chunk_size)``
    -- a :class:`~astrogwb.simulators.WaveformArtifact` read from disk, or
    :class:`~astrogwb.simulators.Waveforms` generated on the fly -- and
    accumulates each chunk's ``(P, F)`` cross-powers. One chunk shape is one
    compile, whatever the realized count, and peak memory is one chunk of
    polarizations. The observation time comes from the waveforms' population.
    Build once and call often.
    """

    def __init__(
        self, detectors: DetectorConfiguration, *, chunk_size: int = 1024
    ) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._detectors = detectors
        self._chunk_size = chunk_size
        self._pair_sums = jax.jit(self._chunk_pair_sums)

    @property
    def detectors(self) -> DetectorConfiguration:
        """The network these spectra are recorded by."""
        return self._detectors

    def _chunk_pair_sums(
        self,
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        plus: jax.Array,
        cross: jax.Array,
    ) -> jax.Array:
        """The ``(P, F)`` cross-power summed over one chunk's sources."""
        import jax.numpy as jnp

        from astrogwb.detector import stokes_cross_power
        from astrogwb.waveform.polarization_power import polarization_stokes

        # (4, F, C) -> (4, C, F)
        stokes = jnp.swapaxes(polarization_stokes(plus, cross), 1, 2)
        return jnp.stack(
            [
                stokes_cross_power(
                    frequencies,
                    first,
                    second,
                    sources["polar_cosine"],
                    sources["azimuth"],
                    sources["polarization"],
                    stokes,
                ).sum(axis=0)
                for first, second in self._detectors.pairs()
            ]
        )

    def __call__(self, waveforms: WaveformChunks) -> BackgroundArtifact:
        """The per-pair spectra of ``waveforms``.

        Raises
        ------
        KeyError
            If the sources carry no sky position or polarization angle.
        """
        import jax.numpy as jnp

        population = waveforms.population
        frequencies = np.asarray(waveforms.frequencies, dtype=np.float64)
        missing = {"polar_cosine", "azimuth", "polarization"} - set(
            population.source_parameters
        )
        if missing:
            raise KeyError(f"population has no extrinsic columns {sorted(missing)}")
        total = jnp.zeros((len(self._detectors.pairs()), frequencies.shape[0]))
        grid = jnp.asarray(frequencies)
        for sources, plus, cross in waveforms.iter_chunks(self._chunk_size):
            extrinsic = {
                name: sources[name]
                for name in ("polar_cosine", "azimuth", "polarization")
            }
            total = total + self._pair_sums(grid, extrinsic, plus, cross)
        spectra = (
            PAIR_RESPONSE_DENOMINATOR
            * np.asarray(total, dtype=np.float64)
            / population.observation_seconds
        )
        return BackgroundArtifact(
            detectors=self._detectors,
            frequencies=frequencies,
            pair_spectral_density=spectra,
            observation_seconds=population.observation_seconds,
            count=population.count,
            total_merger_rate=population.total_merger_rate,
            hyperparameters=dict(population.hyperparameters),
            waveform_key=waveforms.key(),
            chunk_size=self._chunk_size,
        )


def _pair_weights(
    frequencies: NDArray[np.float64], detectors: DetectorConfiguration
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The pair weights ``(P, F)`` and their ORF-weighted sum ``(F,)`` of a network."""
    from astrogwb.detector import pair_terms

    terms = pair_terms(frequencies, detectors.detectors, detectors.sensitivities())
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)
    return weights, denominator


def combine_pair_spectra(
    pair_spectral_density: ArrayLike,
    frequencies: ArrayLike,
    detectors: DetectorConfiguration,
) -> NDArray[np.float64]:
    r"""The network spectrum of per-pair spectra ``(..., P, F)``, ``(..., F)``.

    :math:`\sum_{a<b} w_{ab} Y_{ab} / \sum_{a<b} w_{ab} \gamma_{ab}` with the
    optimal weights :math:`w_{ab} = \gamma_{ab} / (S_a S_b)` of
    :func:`~astrogwb.detector.pair_terms`, the pairs in the order of
    :meth:`DetectorConfiguration.pairs`; its mean is :math:`S_h` for an
    isotropic, unpolarized population. Bins where no pair has a finite weight
    return zero.
    """
    weights, denominator = _pair_weights(np.asarray(frequencies), detectors)
    numerator = np.sum(weights * np.asarray(pair_spectral_density), axis=-2)
    usable = denominator > 0.0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)
