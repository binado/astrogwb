"""Network optimal SNR for resolved compact-binary events.

Composes gwmock building blocks into one batched entry point: time-domain
waveform generation (gwmock's LALSimulation backend), projection onto a
detector network with Earth rotation, and a Whittle inner-product
contraction against the network noise. The inverse spectral noise matrix is
built once per call and reused across events; only the waveform generation,
projection, and the final quadratic form run per event.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import (
    DetectorSpec,
    evaluate_psd,
    load_sensitivity,
    resolve_detector,
)

logger = logging.getLogger(__name__)

#: Parameters consumed by this module itself (segment placement and sky
#: position); everything else is forwarded to the waveform backend.
_RESERVED_PARAMETER_KEYS = frozenset({"tc", "ra", "dec", "psi"})

#: Parameters every event must carry: coalescence time and sky/polarization
#: angles for the projection, plus the backend's required intrinsic set.
_REQUIRED_PARAMETER_KEYS = (
    "tc",
    "ra",
    "dec",
    "psi",
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
)


def _normalize_parameters(
    source_parameters: Mapping[str, ArrayLike],
) -> tuple[dict[str, NDArray[np.float64]], dict[str, float], int]:
    """Split event parameters into per-event arrays and broadcastable scalars.

    Returns ``(event_arrays, scalars, n_events)`` where ``event_arrays``
    holds every parameter given as an array of shape ``(n_events,)`` and
    ``scalars`` holds parameters given as Python floats (broadcast across
    events). Mixed lengths are rejected here so downstream code sees one
    consistent event count.
    """
    if not source_parameters:
        raise ValueError("source_parameters must not be empty.")

    missing = [key for key in _REQUIRED_PARAMETER_KEYS if key not in source_parameters]
    if missing:
        raise ValueError(f"Missing required source parameters: {missing}")

    event_arrays: dict[str, NDArray[np.float64]] = {}
    scalars: dict[str, float] = {}
    n_events: int | None = None
    for key, value in source_parameters.items():
        array = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if array.ndim != 1:
            raise ValueError(
                f"Source parameter {key!r} must be scalar or 1-dimensional."
            )
        if array.shape[0] > 1:
            if n_events is None:
                n_events = array.shape[0]
            elif array.shape[0] != n_events:
                raise ValueError(
                    f"Source parameter {key!r} has length {array.shape[0]}, "
                    f"expected {n_events} to match the other per-event arrays."
                )
            event_arrays[key] = array
        else:
            scalars[key] = float(array[0])

    return event_arrays, scalars, n_events if n_events is not None else 1


def _resolve_network(
    detectors: Sequence[DetectorSpec],
) -> list[tuple[str, DetectorSpec]]:
    """Resolve detector specs to unique ``(name, spec)`` pairs in input order."""
    if not detectors:
        raise ValueError("At least one detector is required.")
    resolved: list[tuple[str, DetectorSpec]] = []
    seen: set[str] = set()
    for spec in detectors:
        custom = resolve_detector(spec)
        name = custom.name
        if name in seen:
            raise ValueError(f"Duplicate detector name {name!r}.")
        seen.add(name)
        resolved.append((name, custom))
    return resolved


def _psd_references_for(
    names: Sequence[str],
    overrides: Mapping[str, str | Path] | None,
) -> dict[str, str | Path]:
    """Resolve per-detector PSD references: explicit overrides first, then TOML defaults."""
    references: dict[str, str | Path] = {}
    for name in names:
        if overrides is not None and name in overrides:
            references[name] = overrides[name]
            continue
        try:
            references[name] = load_sensitivity(name).psd_reference
        except KeyError as exc:
            raise KeyError(
                f"No PSD reference for detector {name!r}: not in sensitivity.toml and "
                "not given in psd_references. Pass psd_references={"
                f"'{name}': <reference>}} explicitly."
            ) from exc
    return references


def _noise_inverse(
    psds: dict[str, NDArray[np.float64]],
    cross_psds: Mapping[tuple[str, str], ArrayLike] | None,
    names: Sequence[str],
    frequencies: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.complex128]:
    """Build the masked inverse spectral noise matrix ``S_n^{-1}(f)``.

    Follows the construction in ``gwmock_signal.snr._network`` (diagonal
    one-sided PSDs, Hermitian cross-PSDs), but inverts only the in-band bins
    once so the result can be shared across events. Out-of-band bins are
    excluded before inversion, so zero-valued PSD entries there cannot cause
    singular matrices.
    """
    n_det = len(names)
    n_freq = frequencies.shape[0]
    index = {name: i for i, name in enumerate(names)}

    noise = np.zeros((n_det, n_det, n_freq), dtype=np.complex128)
    for name in names:
        i = index[name]
        noise[i, i, :] = psds[name]

    if cross_psds is not None:
        for (name_a, name_b), spectrum in cross_psds.items():
            if name_a == name_b:
                raise ValueError(
                    f"cross_psds key ({name_a!r}, {name_b!r}) is diagonal; "
                    "diagonal entries come from the PSDs."
                )
            if name_a not in index or name_b not in index:
                unknown = sorted({name_a, name_b} - index.keys())
                raise ValueError(f"cross_psds references unknown detectors: {unknown}")
            i, j = index[name_a], index[name_b]
            noise[i, j, :] = np.asarray(spectrum, dtype=np.complex128)
            noise[j, i, :] = np.conj(noise[i, j, :])

    return np.linalg.inv(noise.transpose(2, 0, 1)[mask])


def network_optimal_snr(
    source_parameters: Mapping[str, ArrayLike],
    detectors: Sequence[DetectorSpec],
    *,
    waveform_model: str,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None = None,
    psd_references: Mapping[str, str | Path] | None = None,
    cross_psds: Mapping[tuple[str, str], ArrayLike] | None = None,
    earth_rotation: bool = True,
    chunk_size: int = 0,
) -> NDArray[np.float64]:
    """Network optimal SNR ``sqrt((s|s))`` for every event in a catalog.

    For each event the pipeline is: generate time-domain plus/cross
    polarizations with gwmock's LALSimulation backend, project them onto the
    detectors (antenna patterns and delays evaluated at time-dependent GPS
    times when ``earth_rotation`` is on), then contract the projected strains
    against the network noise via the frequency-domain inner product of
    Equations 9 and 18 of Cireddu et al. 2025 (arXiv:2312.14614).

    All events share one analysis segment sized from the longest inspiral in
    the catalog (rounded up to a power-of-two seconds), so the frequency grid
    -- and with it the inverted spectral noise matrix -- is computed once and
    reused. Events are processed one at a time, so peak memory is one
    event's strain regardless of catalog size; ``chunk_size`` only controls
    how often progress is logged.

    Args:
        source_parameters: Per-event parameters as 1-dimensional arrays of
            equal length, or broadcastable scalars. Required keys: ``tc``,
            ``ra``, ``dec``, ``psi``, ``detector_frame_mass_1``,
            ``detector_frame_mass_2``, ``luminosity_distance``. Remaining
            keys (spins, ``inclination``, ``coa_phase``, ``lambda_1``,
            ``lambda_2``) are forwarded to the waveform backend, which
            rejects unknown ones.
        detectors: Detector specifications -- plain LAL site codes
            (resolved through astrogwb's ``geometry.toml``) and/or gwmock
            ``CustomDetector`` instances.
        waveform_model: LAL approximant name, e.g.
            ``"IMRPhenomXAS_NRTidalv3"``.
        sampling_frequency: Sample rate in Hz; sets the Nyquist frequency.
        minimum_frequency: Lower frequency bound of the SNR integral in Hz;
            also the waveform generation cutoff.
        maximum_frequency: Upper bound of the SNR integral in Hz; defaults
            to the Nyquist frequency.
        psd_references: Optional per-detector PSD references keyed by
            detector name (anything :func:`~astrogwb.detector.evaluate_psd`
            accepts). Defaults to each detector's entry in astrogwb's
            ``sensitivity.toml``.
        cross_psds: Optional off-diagonal cross-PSDs keyed by ordered
            detector-name tuples; Hermitian symmetry is applied
            automatically. ``None`` gives the uncorrelated limit.
        earth_rotation: Evaluate antenna patterns and delays at per-sample
            GPS times (recommended); ``False`` evaluates them once at the
            segment midpoint.
        chunk_size: Number of events between progress log records; ``0``
            logs once per call.

    Returns:
        Network optimal SNR per event, shape ``(n_events,)``,
        non-negative.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be > 0")
    if minimum_frequency <= 0:
        raise ValueError("minimum_frequency must be > 0")
    if maximum_frequency is not None and maximum_frequency <= minimum_frequency:
        raise ValueError("maximum_frequency must be > minimum_frequency")
    if chunk_size < 0:
        raise ValueError("chunk_size must be >= 0")

    event_arrays, scalars, n_events = _normalize_parameters(source_parameters)

    network = _resolve_network(detectors)
    names = [name for name, _ in network]

    # Heavy imports stay inside the function so importing this module stays cheap.
    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.waveform.backends.conditioning import segment_sample_count
    from gwmock_signal.waveform.backends.lal import LALSimulationBackend

    # One common segment: size it from the largest chirp mass in the catalog,
    # rounded up to a power-of-two seconds exactly as the backend would.
    mass_1 = event_arrays.get(
        "detector_frame_mass_1", np.array([scalars["detector_frame_mass_1"]])
    )
    mass_2 = event_arrays.get(
        "detector_frame_mass_2", np.array([scalars["detector_frame_mass_2"]])
    )
    chirp_masses = (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2
    heaviest = int(np.argmax(chirp_masses))
    sizing_backend = LALSimulationBackend()
    pre_coalescence_seconds = sizing_backend.pre_coalescence_duration(
        waveform_model,
        sampling_frequency,
        minimum_frequency,
        detector_frame_mass_1=float(mass_1[heaviest]),
        detector_frame_mass_2=float(mass_2[heaviest]),
        luminosity_distance=1.0,
    )
    if pre_coalescence_seconds is None:
        raise ValueError(
            f"Waveform model {waveform_model!r} cannot report its pre-coalescence "
            "duration; cannot size the shared analysis segment."
        )
    segment_duration = float(2.0 ** math.ceil(math.log2(pre_coalescence_seconds)))
    backend = LALSimulationBackend(segment_duration=segment_duration)

    n_samples = segment_sample_count(
        float(chirp_masses[heaviest]),
        minimum_frequency,
        sampling_frequency,
        segment_duration=segment_duration,
    )
    delta_f = sampling_frequency / n_samples
    frequencies = np.arange(n_samples // 2 + 1, dtype=np.float64) * delta_f
    dt = 1.0 / sampling_frequency

    f_high = (
        maximum_frequency if maximum_frequency is not None else float(frequencies[-1])
    )
    mask = (frequencies >= minimum_frequency) & (frequencies <= f_high)

    psds = {
        name: evaluate_psd(reference, frequencies, out_of_band="zero")
        for name, reference in _psd_references_for(names, psd_references).items()
    }
    noise_inverse = _noise_inverse(psds, cross_psds, names, frequencies, mask)

    waveform_keys = [key for key in event_arrays if key not in _RESERVED_PARAMETER_KEYS]
    snrs = np.empty(n_events, dtype=np.float64)
    log_every = chunk_size if chunk_size > 0 else n_events
    for event in range(n_events):
        params = {
            key: float(event_arrays[key][event])
            if key in event_arrays
            else scalars[key]
            for key in waveform_keys
        }
        polarizations = backend.generate_td_waveform(
            waveform_model,
            tc=float(event_arrays["tc"][event])
            if "tc" in event_arrays
            else scalars["tc"],
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            **params,
        )
        projected = project_polarizations_to_network(
            polarizations,
            [spec for _, spec in network],
            right_ascension=(
                float(event_arrays["ra"][event])
                if "ra" in event_arrays
                else scalars["ra"]
            ),
            declination=(
                float(event_arrays["dec"][event])
                if "dec" in event_arrays
                else scalars["dec"]
            ),
            polarization_angle=(
                float(event_arrays["psi"][event])
                if "psi" in event_arrays
                else scalars["psi"]
            ),
            earth_rotation=earth_rotation,
            backend="numpy",
        )

        strains = np.array([np.fft.rfft(projected[name].value) * dt for name in names])
        in_band = strains[:, mask]
        integrand = np.einsum(
            "ik,kij,jk->", in_band.conj(), noise_inverse, in_band
        ).real
        snrs[event] = np.sqrt(max(4.0 * delta_f * integrand, 0.0))

        if (event + 1) % log_every == 0 or event + 1 == n_events:
            logger.info("Computed SNR for %d / %d events", event + 1, n_events)

    return snrs
