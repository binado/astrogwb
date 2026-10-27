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

The stage is configured by named networks, a :data:`DetectorRegistry`.
Every pair of the union of their detectors is projected once per chunk, so a
pair two networks share costs one projection, and each network keeps its own
rows, ``(P_n, F)``, with their :func:`combine_pair_spectra`, the optimally
weighted network spectrum ``(F,)``. The combination weights come from the
networks' noise curves, so the background depends on them and they are part
of its key.

The background is deterministic given its inputs and carries no detector
noise. Detectors are static: the Earth's rotation during a signal is not
modelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector.config import DetectorRegistry
from astrogwb.simulators._io import atomic_h5
from astrogwb.utils.io import content_key, h5py_module

if TYPE_CHECKING:
    import jax

    from astrogwb.simulators.waveforms import WaveformChunks

__all__ = [
    "NORMALIZATION_VERSION",
    "BackgroundArtifact",
    "BackgroundSimulator",
    "DetectorRegistry",
    "NetworkSpectrum",
    "background_key",
    "background_table",
    "combine_pair_spectra",
]

SCHEMA_VERSION = 3

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


class NetworkSpectrum(NamedTuple):
    """One network's background: its pair spectra and their combination.

    ``pair_spectral_density`` is :math:`Y_{ab}` in :math:`\\mathrm{Hz}^{-1}`,
    ``(P_n, F)``, the pairs in the order of
    :meth:`DetectorRegistry.pairs`; ``spectral_density`` is their
    :func:`combine_pair_spectra`, ``(F,)``.
    """

    pair_spectral_density: NDArray[np.float64]
    spectral_density: NDArray[np.float64]


def _network_names(
    registry: DetectorRegistry, names: Sequence[str] | None = None
) -> tuple[str, ...]:
    selected = tuple(sorted(registry.networks if names is None else names))
    for name in selected:
        members = registry.networks[name]
        if len(members) < 2:
            raise ValueError(f"network {name!r} must contain at least two detectors")
    return selected


def _scientific_payload(
    registry: DetectorRegistry, names: Sequence[str]
) -> dict[str, Any]:
    used = sorted({detector for name in names for detector in registry.networks[name]})
    return {
        "networks": {name: list(registry.networks[name]) for name in sorted(names)},
        "detectors": {
            name: {
                "geometry": registry.detectors[name].geometry.model_dump(mode="json"),
                "psd_reference": registry.detectors[name].psd_reference,
            }
            for name in used
        },
    }


def background_key(
    waveform_key: str, registry: DetectorRegistry, names: Sequence[str] | None = None
) -> str:
    """The content hash of the spectra of the waveforms ``waveform_key`` on ``networks``.

    Only used network membership, geometry and PSD references affect it.
    """
    return content_key(
        {
            "waveforms": waveform_key,
            **_scientific_payload(registry, _network_names(registry, names)),
            "normalization_version": NORMALIZATION_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
    )


def background_table(
    waveform_key: str, registry: DetectorRegistry, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """The ``[background]`` manifest table of the spectra :func:`background_key` names."""
    return {
        "metadata": {"networks": list(_network_names(registry, names))},
        "key": background_key(waveform_key, registry, names),
        "normalization": _NORMALIZATION,
        "units": {
            "frequency": "Hz",
            "pair_spectral_density": "1 / Hz",
            "spectral_density": "1 / Hz",
        },
    }


def _read_networks(table: Mapping[str, Any]) -> tuple[str, ...]:
    """The recorded network names of a ``[background]`` table."""
    names = table["metadata"]["networks"]
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("background metadata format is old; regenerate realizations")
    return tuple(names)


@dataclass(frozen=True)
class BackgroundArtifact:
    """The per-network spectra of one realization, in memory or read from disk.

    ``spectra`` holds each network's :class:`NetworkSpectrum`, keyed and
    ordered as ``network_names``. ``count``, ``total_merger_rate`` (per second) and
    ``hyperparameters`` are the realization's, carried over from its
    population. ``waveform_key`` names the polarizations the spectra were
    projected from.
    """

    network_names: tuple[str, ...]
    frequencies: NDArray[np.float64]
    spectra: dict[str, NetworkSpectrum]
    observation_seconds: float
    count: int
    total_merger_rate: float
    hyperparameters: dict[str, float]
    waveform_key: str
    identity: str | None = None
    chunk_size: int | None = None

    def network(self, name: str) -> NetworkSpectrum:
        """The pair spectra ``(P_n, F)`` and network spectrum ``(F,)`` of ``name``."""
        return self.spectra[name]

    def key(self) -> str:
        """The recorded background identity."""
        if self.identity is None:
            raise ValueError("background identity has not been assigned")
        return self.identity

    def table(self, registry: DetectorRegistry) -> dict[str, Any]:
        """The ``[background]`` table using the supplied registry."""
        return background_table(self.waveform_key, registry, self.network_names)

    def write(self, path: str | Path) -> Path:
        """Write the spectra to the HDF5 file ``path``, atomically.

        Datasets ``frequencies`` and, per network,
        ``networks/<name>/pair_spectral_density`` and
        ``networks/<name>/spectral_density``. The pair axis is in the order of
        :meth:`DetectorRegistry.pairs` of the network the manifest records.
        """
        with atomic_h5(path) as file:
            file.create_dataset("frequencies", data=self.frequencies)
            for name, spectrum in self.spectra.items():
                group = file.create_group(f"networks/{name}")
                group.create_dataset(
                    "pair_spectral_density", data=spectrum.pair_spectral_density
                )
                group.create_dataset("spectral_density", data=spectrum.spectral_density)
        return Path(path)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        table: Mapping[str, Any],
        realization: Mapping[str, Any],
        waveform_key: str,
        registry: DetectorRegistry,
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
        network_names = _read_networks(table)
        for name in network_names:
            if name not in registry.networks:
                raise ValueError(f"registry has no recorded network {name!r}")
            if len(registry.networks[name]) < 2:
                raise ValueError(
                    f"network {name!r} must contain at least two detectors"
                )
        h5py = h5py_module()
        with h5py.File(path, "r") as file:
            frequencies = np.asarray(file["frequencies"][()], dtype=np.float64)
            stored = file.get("networks", {})
            if set(stored) != set(network_names):
                raise ValueError(f"{path}: networks do not match the manifest")
            spectra = {
                name: NetworkSpectrum(
                    np.asarray(
                        stored[name]["pair_spectral_density"][()], dtype=np.float64
                    ),
                    np.asarray(stored[name]["spectral_density"][()], dtype=np.float64),
                )
                for name in network_names
            }
        for name, spectrum in spectra.items():
            shape = (len(registry.pairs(name)), frequencies.shape[0])
            if (
                spectrum.pair_spectral_density.shape != shape
                or spectrum.spectral_density.shape != shape[1:]
            ):
                raise ValueError(
                    f"{path}: the spectra of {name!r} do not match the manifest"
                )
        artifact = cls(
            network_names=network_names,
            frequencies=frequencies,
            spectra=spectra,
            observation_seconds=float(realization["observation_seconds"]),
            count=int(realization["count"]),
            total_merger_rate=float(realization["total_merger_rate"]),
            hyperparameters={
                name: float(value)
                for name, value in realization["hyperparameters"].items()
            },
            waveform_key=waveform_key,
            identity=str(table["key"]),
        )
        return artifact


class BackgroundSimulator:
    """Projects polarizations onto detector networks and sums the pair cross-powers.

    ``simulator(waveforms)`` loops over ``waveforms.iter_chunks(chunk_size)``
    -- a :class:`~astrogwb.simulators.WaveformArtifact` read from disk, or
    :class:`~astrogwb.simulators.Waveforms` generated on the fly -- and
    accumulates each chunk's cross-powers over the detector union's pairs,
    so a pair shared by networks is projected
    once. Each network then takes its rows and their
    :func:`combine_pair_spectra`, using the registry's resolved geometry and
    PSD references. One chunk shape is one compile, whatever the realized
    count, and peak memory is one chunk of polarizations. The observation time
    comes from the waveforms' population. Build once and call often.
    """

    def __init__(self, registry: DetectorRegistry, *, chunk_size: int = 1024) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._registry = registry
        self._network_names = _network_names(registry)
        self._detector_names = tuple(
            sorted({d for n in self._network_names for d in registry.networks[n]})
        )
        if len(self._network_names) != len(registry.networks):
            raise ValueError(
                "registry networks must all be valid for background simulation"
            )
        pair_names = [
            (a, b)
            for i, a in enumerate(self._detector_names)
            for b in self._detector_names[i + 1 :]
        ]
        rows = {pair: index for index, pair in enumerate(pair_names)}
        self._rows = {
            name: [rows[pair] for pair in registry.pairs(name)]
            for name in self._network_names
        }
        self._detectors, self._sensitivities = registry.build_detectors(
            self._detector_names
        )
        self._detector_objects = dict(
            zip(self._detector_names, self._detectors, strict=True)
        )
        self._pairs = [
            (self._detector_objects[a], self._detector_objects[b])
            for a, b in pair_names
        ]
        self._network_runtime = {
            name: (
                tuple(self._detector_objects[d] for d in registry.networks[name]),
                {d: self._sensitivities[d] for d in registry.networks[name]},
            )
            for name in self._network_names
        }
        self._chunk_size = chunk_size
        self._pair_sums = jax.jit(self._chunk_pair_sums)

    @property
    def registry(self) -> DetectorRegistry:
        """The registry used to configure this simulator."""
        return self._registry

    def _chunk_pair_sums(
        self,
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        plus: jax.Array,
        cross: jax.Array,
    ) -> jax.Array:
        """The ``(P_union, F)`` cross-power summed over one chunk's sources."""
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
                for first, second in self._pairs
            ]
        )

    def __call__(self, waveforms: WaveformChunks) -> BackgroundArtifact:
        """The per-network spectra of ``waveforms``.

        Raises
        ------
        KeyError
            If the sources carry no sky position or polarization angle, or a
            detector has no ``psd_references`` entry.
        """
        import jax.numpy as jnp

        population = waveforms.population
        frequencies = np.asarray(waveforms.frequencies, dtype=np.float64)
        missing = {"polar_cosine", "azimuth", "polarization"} - set(
            population.source_parameters
        )
        if missing:
            raise KeyError(f"population has no extrinsic columns {sorted(missing)}")
        total = jnp.zeros((len(self._pairs), frequencies.shape[0]))
        grid = jnp.asarray(frequencies)
        for sources, plus, cross in waveforms.iter_chunks(self._chunk_size):
            extrinsic = {
                name: sources[name]
                for name in ("polar_cosine", "azimuth", "polarization")
            }
            total = total + self._pair_sums(grid, extrinsic, plus, cross)
        union = (
            PAIR_RESPONSE_DENOMINATOR
            * np.asarray(total, dtype=np.float64)
            / population.observation_seconds
        )
        spectra = {}
        for name in self._network_names:
            pairs = union[self._rows[name]]
            detectors, sensitivities = self._network_runtime[name]
            weights, denominator = _pair_weights_from_objects(
                frequencies, detectors, sensitivities
            )
            numerator = np.sum(weights * pairs, axis=-2)
            usable = denominator > 0.0
            combined = np.where(
                usable, numerator / np.where(usable, denominator, 1.0), 0.0
            )
            spectra[name] = NetworkSpectrum(pairs, combined)
        return BackgroundArtifact(
            network_names=self._network_names,
            frequencies=frequencies,
            spectra=spectra,
            observation_seconds=population.observation_seconds,
            count=population.count,
            total_merger_rate=population.total_merger_rate,
            hyperparameters=dict(population.hyperparameters),
            waveform_key=waveforms.key(),
            identity=background_key(
                waveforms.key(), self._registry, self._network_names
            ),
            chunk_size=self._chunk_size,
        )


def _pair_weights(
    frequencies: NDArray[np.float64], registry: DetectorRegistry, network_name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The pair weights ``(P, F)`` and their ORF-weighted sum ``(F,)`` of a network."""

    detectors, sensitivities = registry.build_network(network_name)
    return _pair_weights_from_objects(frequencies, detectors, sensitivities)


def _pair_weights_from_objects(
    frequencies: NDArray[np.float64],
    detectors: Sequence[Any],
    sensitivities: Mapping[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pair weights and denominator using already-resolved runtime objects."""
    from astrogwb.detector import pair_terms

    terms = pair_terms(frequencies, detectors, sensitivities)
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)
    return weights, denominator


def combine_pair_spectra(
    pair_spectral_density: ArrayLike,
    frequencies: ArrayLike,
    registry: DetectorRegistry,
    network_name: str,
) -> NDArray[np.float64]:
    r"""The network spectrum of per-pair spectra ``(..., P, F)``, ``(..., F)``.

    :math:`\sum_{a<b} w_{ab} Y_{ab} / \sum_{a<b} w_{ab} \gamma_{ab}` with the
    optimal weights :math:`w_{ab} = \gamma_{ab} / (S_a S_b)` of
    :func:`~astrogwb.detector.pair_terms`, the pairs in the order of
    :meth:`DetectorRegistry.pairs`; its mean is :math:`S_h` for an
    isotropic, unpolarized population. Bins where no pair has a finite weight
    return zero.
    """
    weights, denominator = _pair_weights(
        np.asarray(frequencies), registry, network_name
    )
    numerator = np.sum(weights * np.asarray(pair_spectral_density), axis=-2)
    usable = denominator > 0.0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)
