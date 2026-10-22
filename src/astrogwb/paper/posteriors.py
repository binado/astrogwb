"""Setup shared by the posterior notebooks.

The cache digest, the reference catalog, the cached injection, the per-network
Gaussian data and the shot-noise variants: everything a grid or a sampler needs
before it evaluates a likelihood. Only the stdlib, NumPy and :mod:`astrogwb`
are imported at module scope, and ``jax.numpy`` inside the functions that use
it, so importing this module leaves the XLA backend alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import (
    effective_psd,
    gaussian_bin_scale,
    pair_bin_scale,
    pair_terms,
)
from astrogwb.frequency import frequency_mask
from astrogwb.gwb.importance import reference_catalog, reference_catalog_stem
from astrogwb.inference import ImportanceGaussianLikelihood
from astrogwb.paper.config.detectors import DetectorRegistry
from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerData,
)
from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityMetadata,
    BackgroundSpectralDensitySimulator,
    NetworkSpectralDensityMetadata,
)

__all__ = [
    "ensure_reference_catalog",
    "injection_spectrum",
    "network_data",
    "network_injection_metadata",
    "network_pair_data",
    "noise_digest",
    "pad_pairs",
    "settings_digest",
    "shot_noise_variant",
]


def settings_digest(**settings: object) -> str:
    """Short hash of everything a cached result depends on.

    Parameters
    ----------
    **settings
        JSON-serializable (or ``str``-able) values; key order is irrelevant.

    Returns
    -------
    str
        Twelve hex characters, for use in a cache file name.
    """
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def ensure_reference_catalog(
    metadata: CatalogMetadata,
    seed: int,
    directory: Path,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The reference catalog of ``metadata`` at ``seed``, drawn on a miss.

    The file is ``<directory>/reference_catalog-<key>-<seed>.h5``. A hit is
    checked against the request it records; a miss is drawn at
    ``batch_keys(seed, 1)[0]`` and written.

    Parameters
    ----------
    metadata
        The catalog's record.
    seed
        The draw's seed.
    directory
        Cache directory.
    chunk_size
        Waveforms per ``lax.map`` batch on a miss; cost only.

    Returns
    -------
    PolarizationPowerData
        Power ``(F_ref, N)``, every draw at the window's lower edge.

    Raises
    ------
    ValueError
        If the cached file records another metadata or seed.
    """
    path = directory / f"{reference_catalog_stem(metadata, seed)}.h5"
    if path.is_file():
        data, recorded, attrs = load(path, CatalogMetadata)
        if recorded.key() != metadata.key() or attrs.get("seed") != seed:
            raise ValueError(
                f"{path} records {recorded.key()} at seed {attrs.get('seed')}, "
                f"not the requested {metadata.key()} at seed {seed}"
            )
        return data  # ty: ignore[invalid-return-type]
    data = reference_catalog(metadata, batch_keys(seed, 1)[0], chunk_size=chunk_size)
    write(path, data, metadata, seed=seed)
    return data


def injection_spectrum(
    metadata: BackgroundSpectralDensityMetadata,
    seed: int,
    path: Path,
    *,
    chunk_size: int = 128,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """The zero-noise injection of ``metadata`` at ``seed``, drawn on a miss.

    The data are the Poisson spectrum itself, with no Gaussian noise added. The
    file records none of the settings behind it, so the caller puts the
    metadata key and the seed in ``path``.

    Parameters
    ----------
    metadata
        The injection's record.
    seed
        The draw's seed.
    path
        ``.npz`` cache file.
    chunk_size
        Waveforms per reduced chunk on a miss; cost only.

    Returns
    -------
    tuple
        The spectral density ``(F,)``, its frequencies ``(F,)`` and the number
        of injected events.
    """
    if path.is_file():
        with np.load(path) as cached:
            return (
                cached["spectral_density"],
                cached["frequencies"],
                int(cached["n_events"]),
            )
    drawn = BackgroundSpectralDensitySimulator(metadata, chunk_size=chunk_size)(
        batch_keys(seed, 1)[0]
    )
    # The draw axis is leading even for one draw.
    observed = np.asarray(drawn["spectral_density"][0], dtype=np.float64)
    frequencies = np.asarray(drawn["frequencies"], dtype=np.float64)
    num_injected = int(drawn["n_events"][0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, spectral_density=observed, frequencies=frequencies, n_events=num_injected
    )
    return observed, frequencies, num_injected


def network_injection_metadata(
    metadata: BackgroundSpectralDensityMetadata,
    registry: DetectorRegistry,
    name: str,
) -> NetworkSpectralDensityMetadata:
    """The injection ``metadata`` as the network ``name`` records it.

    The registry's detectors of the network, in declared order, with their
    ``psd_reference``. Its key differs from the sky-averaged record's, so the
    two caches never mix. The record names detectors by site code, which
    resolves to the packaged geometry: a registry geometry override is not
    carried into it.
    """
    detectors = list(registry.networks[name])
    return NetworkSpectralDensityMetadata(
        **metadata.model_dump(),
        detectors=detectors,
        psd_references={
            detector: registry.detectors[detector].psd_reference
            for detector in detectors
        },
    )


def network_data(
    registry: DetectorRegistry,
    name: str,
    frequencies: NDArray[np.float64],
    *,
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
) -> dict[str, object]:
    """The per-bin Gaussian scale and usable-bin mask of one network.

    Bins no detector pair measures have infinite effective PSD; they are
    dropped from the mask and given a finite scale of one, as
    ``InferenceInputs.model_kwargs`` does.

    Returns
    -------
    dict
        ``scale`` and ``frequency_mask``, shape ``(F,)``, as model keyword
        arguments.
    """
    import jax.numpy as jnp

    grid = jnp.asarray(frequencies)
    geometry, sensitivities = registry.build_network(name)
    psd = jnp.asarray(effective_psd(grid, geometry, sensitivities))
    mask = (
        frequency_mask(grid, fmin=minimum_frequency, fmax=maximum_frequency)
        & jnp.isfinite(psd)
        & (psd > 0.0)
    )
    scale = gaussian_bin_scale(psd, observation_time, grid)
    return {"scale": jnp.where(mask, scale, 1.0), "frequency_mask": mask}


#: The most detector pairs of a paper network (ET triangle + CE); networks
#: with fewer are padded to it so they stack.
MAX_PAIRS = 6


def pad_pairs(values: ArrayLike, num_pairs: int = MAX_PAIRS) -> NDArray:
    """Pad the pair axis (second to last) of ``values`` with zeros to ``num_pairs``."""
    values = np.asarray(values)
    widths = [(0, 0)] * values.ndim
    widths[-2] = (0, num_pairs - values.shape[-2])
    return np.pad(values, widths)


def network_pair_data(
    registry: DetectorRegistry,
    name: str,
    frequencies: NDArray[np.float64],
    *,
    observation_time: float,
    minimum_frequency: float,
    maximum_frequency: float,
    num_pairs: int = MAX_PAIRS,
) -> dict[str, object]:
    """The per-pair Gaussian scale, usable-entry mask and ORF of one network.

    The stacked-pair counterpart of :func:`network_data`: ``scale`` is
    :func:`~astrogwb.detector.pair_bin_scale`, ``pair_overlap`` the isotropic
    ORF of each pair, pairs in the order of
    :func:`~astrogwb.detector.pair_terms`, padded to ``num_pairs`` with masked
    rows of zero overlap and unit scale so networks stack. Entries outside the
    band or with an infinite scale are masked and given a scale of one.

    Returns
    -------
    dict
        ``scale``, ``frequency_mask`` and ``pair_overlap``, ``(num_pairs, F)``,
        as :class:`~astrogwb.inference.Network` fields.
    """
    import jax.numpy as jnp

    grid = np.asarray(frequencies)
    geometry, sensitivities = registry.build_network(name)
    scale = np.asarray(pair_bin_scale(grid, geometry, sensitivities, observation_time))
    overlap = np.stack(
        [term.overlap for term in pair_terms(grid, geometry, sensitivities)]
    )
    band = np.asarray(
        frequency_mask(
            jnp.asarray(grid), fmin=minimum_frequency, fmax=maximum_frequency
        )
    )
    mask = band & np.isfinite(scale) & (scale > 0.0)
    padded_scale = np.ones((num_pairs, grid.size))
    padded_scale[: scale.shape[0]] = np.where(mask, scale, 1.0)
    return {
        "scale": jnp.asarray(padded_scale),
        "frequency_mask": jnp.asarray(pad_pairs(mask, num_pairs)),
        "pair_overlap": jnp.asarray(pad_pairs(overlap, num_pairs)),
    }


def noise_digest(per_network: dict[str, dict[str, object]]) -> dict[str, str]:
    """Short hash of each network's per-bin noise, for the cache digest.

    Network names alone do not identify the noise: editing a detector's
    geometry or PSD changes the scale under the same name. Hashing the
    :func:`network_data` arrays keys cached results to the noise itself.

    Parameters
    ----------
    per_network
        :func:`network_data` by network name.

    Returns
    -------
    dict
        Twelve hex characters per network name.
    """
    digests = {}
    for name, data in per_network.items():
        hasher = hashlib.sha256()
        for key in ("scale", "frequency_mask", "pair_overlap"):
            if key not in data:
                continue
            hasher.update(np.asarray(data[key]).tobytes())
        digests[name] = hasher.hexdigest()[:12]
    return digests


def shot_noise_variant(
    variant: str,
    likelihood: ImportanceGaussianLikelihood,
    fiducial_relative_variance: NDArray[np.float64],
) -> tuple[ImportanceGaussianLikelihood, NDArray[np.float64] | None, dict[str, object]]:
    r"""The likelihood, per-network :math:`s^2` and cache settings of one variant.

    Parameters
    ----------
    variant
        ``"detector"``, ``"amplitude"``, ``"fixed"`` or ``"per_frequency"``.
    likelihood
        Built with ``shot_noise="amplitude"``, so the squared power is held
        for every variant.
    fiducial_relative_variance
        :math:`s^2` at the fiducials per network, in ``per_network`` order;
        used by ``"fixed"`` only.

    Returns
    -------
    tuple
        The likelihood with the variant's mode, the ``"fixed"`` variant's
        per-network :math:`s^2` (else ``None``), and the settings that set the
        variant apart in a cache digest. The detector variant adds none, so
        its caches predate shot noise and stay valid.
    """
    if variant == "detector":
        return dataclasses.replace(likelihood, shot_noise=None), None, {}
    if variant == "fixed":
        values = [float(value) for value in fiducial_relative_variance]
        return (
            dataclasses.replace(likelihood, shot_noise="fixed"),
            np.asarray(values),
            {"shot_noise": variant, "relative_variance": values},
        )
    return (
        dataclasses.replace(likelihood, shot_noise=variant),
        None,
        {"shot_noise": variant},
    )
