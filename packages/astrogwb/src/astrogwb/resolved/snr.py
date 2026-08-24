"""Per-detector optimal SNR for resolved compact-binary events.

Composes gwmock building blocks into one batched entry point: waveform
generation, projection onto a detector network with Earth rotation, and a
Whittle inner-product contraction of each projected strain against that
detector's PSD. Two generation backends are supported; see
:func:`optimal_snr` for the selection rule. Inverse PSDs are built once
per duration batch and reused across its events. Progress is reported through an
optional caller-provided callback; the package performs no logging.

Like the rest of astrogwb, the API consumes materialized objects:
``CustomDetector`` instances carrying geometry (see
:func:`astrogwb.detector.resolve_detector`) and a name-keyed
:class:`astrogwb.detector.Sensitivity` mapping (see
:func:`astrogwb.detector.load_sensitivity_map`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from gwmock_signal.detector import CustomDetector
from gwmock_signal.jax_batch import simulate_cbc_batch
from gwmock_signal.projection.network import project_polarizations_to_network
from gwmock_signal.waveform.backends.conditioning import segment_sample_count
from gwmock_signal.waveform.backends.lal import LALSimulationBackend
from gwmock_signal.waveform.backends.ripple import RippleBackend
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import Sensitivity, in_band_inverse_psd

WaveformBackend = Literal["auto", "ripple", "lal"]

#: Parameters consumed by this module itself (segment placement and sky
#: position). Names are the gwmock-pop canonical vocabulary.
_RESERVED_PARAMETER_KEYS = frozenset(
    {"coa_time", "right_ascension", "declination", "polarization_angle"}
)

#: Canonical keys forwarded to the waveform backend. Catalog metadata
#: (``redshift``, ``source_frame_mass_*``, ...) is dropped.
_WAVEFORM_PARAMETER_KEYS = frozenset(
    {
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "luminosity_distance",
        "spin_1x",
        "spin_1y",
        "spin_1z",
        "spin_2x",
        "spin_2y",
        "spin_2z",
        "inclination",
        "coa_phase",
        "lambda_1",
        "lambda_2",
    }
)

_KEEP_PARAMETER_KEYS = _RESERVED_PARAMETER_KEYS | _WAVEFORM_PARAMETER_KEYS


def _normalize_parameters(
    source_parameters: Mapping[str, ArrayLike],
) -> tuple[dict[str, NDArray[np.float64]], int]:
    """Coerce kept event parameters to 1-D float64 arrays of one shared length.

    Returns ``(event_arrays, n_events)``. Only reserved projection keys and
    the waveform allow-list are retained, but every input value must be a
    non-empty 1-dimensional array of the same length.
    """
    if not source_parameters:
        raise ValueError("source_parameters must not be empty.")

    event_arrays: dict[str, NDArray[np.float64]] = {}
    n_events: int | None = None
    for key, value in source_parameters.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"Source parameter {key!r} must be 1-dimensional.")
        if n_events is None:
            n_events = array.shape[0]
            if n_events == 0:
                raise ValueError("Source parameter arrays must be non-empty.")
        elif array.shape[0] != n_events:
            raise ValueError(
                f"Source parameter {key!r} has length {array.shape[0]}, "
                f"expected {n_events} to match the other per-event arrays."
            )
        if key in _KEEP_PARAMETER_KEYS:
            event_arrays[key] = array

    assert n_events is not None  # non-empty input guarantees it was set
    return event_arrays, n_events


def _chirp_mass(
    mass_1: NDArray[np.float64], mass_2: NDArray[np.float64]
) -> NDArray[np.float64]:
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def _rfft_grid(
    n_samples: int, sampling_frequency: float
) -> tuple[NDArray[np.float64], float, float]:
    """Return ``(frequencies, delta_f, dt)`` for an ``n_samples`` rFFT grid."""
    delta_f = sampling_frequency / n_samples
    frequencies = np.arange(n_samples // 2 + 1, dtype=np.float64) * delta_f
    dt = 1.0 / sampling_frequency
    return frequencies, delta_f, dt


def matched_filter_snr(
    strain: NDArray[np.float64],
    inv_psd: NDArray[np.float64],
    mask: NDArray[np.bool_],
    delta_f: float,
    dt: float,
) -> NDArray[np.float64]:
    """CBC optimal SNR ``sqrt(4 Δf Σ |h|² / S)`` for rank-3 time-domain strain.

    ``strain`` has shape ``(n_events, n_detectors, n_time)``. ``inv_psd`` has
    shape ``(n_detectors, n_in_band)`` and ``mask`` selects those bins on the
    rFFT frequency axis. Returns ``(n_events, n_detectors)``.
    """
    spectra = np.fft.rfft(strain, axis=-1) * dt
    in_band = spectra[..., mask]
    integrand = (np.abs(in_band) ** 2 * inv_psd).sum(axis=-1).real
    return np.sqrt(np.maximum(4.0 * delta_f * integrand, 0.0))


def optimal_snr(
    source_parameters: Mapping[str, ArrayLike],
    detectors: Sequence[CustomDetector],
    sensitivities: Mapping[str, Sensitivity],
    *,
    waveform_model: str,
    sampling_frequency: float,
    minimum_frequency: float,
    batch_size: int,
    maximum_frequency: float | None = None,
    earth_rotation: bool = True,
    backend: WaveformBackend = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Single-detector optimal SNR ``sqrt((s|s))`` for every event and detector.

    Two generation backends are available. With ``backend="auto"`` (the
    default) the ripple (JAX) backend is used when it is installed and
    supports ``waveform_model``: each duration batch is generated under
    ``jax.vmap`` and projected on device. Otherwise
    the LALSimulation backend is used as a per-event, time-domain fallback.
    ``backend="ripple"`` requires a usable ripple backend;
    ``backend="lal"`` skips ripple. Either way each event/detector strain
    is contracted against that detector's PSD via the frequency-domain inner
    product of Equations 9 and 18 of Cireddu et al. 2025 (arXiv:2312.14614).
    Detector antenna patterns and delays are evaluated at time-dependent GPS
    times when ``earth_rotation`` is on. Detectors are treated as
    uncorrelated; the uncorrelated network SNR is
    ``np.sqrt((rho ** 2).sum(axis=1))``.

    Events are stably sorted by their backend-specific required duration,
    longest first. Each group of at most ``batch_size`` adjacent events is
    both a waveform batch and a duration bucket, sharing the shortest grid
    that contains that bucket's longest signal. Results are restored to
    input order before return. The LAL path still holds only one event's
    strain at a time.

    Parameters
    ----------
    source_parameters:
        Per-event parameters as 1-dimensional arrays, all of equal length,
        named in the gwmock-pop canonical vocabulary. Waveform keys (masses,
        distance, spins, ``inclination``, ``coa_phase``, ``lambda_1``, and
        ``lambda_2``) are forwarded to gwmock, which validates its required
        inputs. Other columns (catalog metadata such as ``redshift``) are
        validated for shape and then ignored.
    detectors:
        Resolved detector geometries (gwmock ``CustomDetector`` instances,
        e.g. from :func:`astrogwb.detector.resolve_detector`). Names must
        be unique. Column ``j`` of the returned array corresponds to
        ``detectors[j]``.
    sensitivities:
        Noise curves keyed by detector name (e.g. from
        :func:`astrogwb.detector.load_sensitivity_map`); must contain an
        entry for every detector in ``detectors``.
    waveform_model:
        Approximant name, e.g. ``"IMRPhenomXAS_NRTidalv3"``.
    sampling_frequency:
        Sample rate in Hz; sets the Nyquist frequency.
    minimum_frequency:
        Lower frequency bound of the SNR integral in Hz; also the waveform
        generation cutoff.
    batch_size:
        Maximum events sharing one duration-sized grid. Must be positive.
    maximum_frequency:
        Upper bound of the SNR integral in Hz; defaults to the Nyquist
        frequency.
    earth_rotation:
        Evaluate antenna patterns and delays at per-sample GPS times
        (recommended); ``False`` evaluates them once at the segment
        midpoint.
    backend:
        ``"auto"`` uses ripple when it supports ``waveform_model``, else
        LALSimulation. ``"ripple"`` requires a usable ripple backend.
        ``"lal"`` always uses LALSimulation.
    progress_callback:
        Optional callable invoked as ``(done, total)`` after each completed
        duration batch. ``done`` is the
        number of events finished so far. The package itself performs no
        logging; callers may log from the callback (throttling there if
        desired).

    Returns
    -------
    numpy.ndarray
        Optimal SNR per event and detector, shape
        ``(n_events, n_detectors)``, non-negative. Column ``j`` is
        ``detectors[j]``.
    """
    if backend not in ("auto", "ripple", "lal"):
        raise ValueError(
            f"backend must be 'auto', 'ripple', or 'lal'; got {backend!r}."
        )
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be > 0")
    if minimum_frequency <= 0:
        raise ValueError("minimum_frequency must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if maximum_frequency is not None and maximum_frequency <= minimum_frequency:
        raise ValueError("maximum_frequency must be > minimum_frequency")

    event_arrays, n_events = _normalize_parameters(source_parameters)

    if not detectors:
        raise ValueError("At least one detector is required.")
    names = [detector.name for detector in detectors]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate detector name(s): {duplicates}")
    missing_sensitivities = [name for name in names if name not in sensitivities]
    if missing_sensitivities:
        raise KeyError(
            f"No sensitivity for detector(s) {missing_sensitivities}: every "
            "detector must have an entry in the sensitivities mapping."
        )

    ripple_backend: RippleBackend | None = None
    if backend != "lal":
        ripple_backend = _select_ripple_backend(waveform_model)
        if ripple_backend is None and backend == "ripple":
            raise ValueError(
                f"Ripple backend is not available for waveform model "
                f"{waveform_model!r}."
            )
    if ripple_backend is not None:
        return _optimal_snr_ripple_batch(
            ripple_backend,
            event_arrays,
            n_events,
            detectors,
            names,
            sensitivities,
            waveform_model,
            batch_size=batch_size,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            earth_rotation=earth_rotation,
            progress_callback=progress_callback,
        )
    return _optimal_snr_lal(
        event_arrays,
        n_events,
        detectors,
        names,
        sensitivities,
        waveform_model,
        batch_size=batch_size,
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        earth_rotation=earth_rotation,
        progress_callback=progress_callback,
    )


def _select_ripple_backend(waveform_model: str) -> RippleBackend | None:
    """Return a ripple backend able to generate ``waveform_model``, else ``None``.

    ``None`` covers both "ripple/JAX is not installed" and "ripple has no
    such approximant"; callers then fall back to the LAL path. An
    installed ripple whose interface is incompatible raises from the
    backend's own guard rather than being silently skipped.
    """
    try:
        backend = RippleBackend()
    except ImportError:
        return None
    if waveform_model not in backend.available_approximants():
        return None
    return backend


def _duration_sorted_batches(
    durations: NDArray[np.float64], batch_size: int
) -> list[tuple[NDArray[np.intp], float]]:
    """Plan longest-first batches; each batch is its own duration bucket."""
    order = np.argsort(-durations, kind="stable")
    batches: list[tuple[NDArray[np.intp], float]] = []
    start = 0
    while start < order.size:
        duration = float(durations[order[start]])
        stop = min(start + batch_size, order.size)
        batches.append((order[start:stop].astype(np.intp, copy=False), duration))
        start = stop
    return batches


def _optimal_snr_ripple_batch(
    backend: RippleBackend,
    event_arrays: Mapping[str, NDArray[np.float64]],
    n_events: int,
    detectors: Sequence[CustomDetector],
    names: Sequence[str],
    sensitivities: Mapping[str, Sensitivity],
    waveform_model: str,
    *,
    batch_size: int,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    earth_rotation: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> NDArray[np.float64]:
    """Ripple path with one duration-sized grid per vmapped batch."""
    mass_1 = event_arrays["detector_frame_mass_1"]
    mass_2 = event_arrays["detector_frame_mass_2"]
    chirp_masses = _chirp_mass(mass_1, mass_2)
    eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2
    durations = np.asarray(
        [
            backend.segment_duration_for(
                float(chirp_mass),
                minimum_frequency,
                sampling_frequency,
                eta=float(symmetric_mass_ratio),
            )
            for chirp_mass, symmetric_mass_ratio in zip(chirp_masses, eta, strict=True)
        ],
        dtype=np.float64,
    )

    snrs = np.empty((n_events, len(names)), dtype=np.float64)
    done = 0
    for indices, segment_duration in _duration_sorted_batches(durations, batch_size):
        batch_backend = backend.with_segment_duration(segment_duration)
        batch = simulate_cbc_batch(
            waveform_model,
            list(detectors),
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            parameters={key: values[indices] for key, values in event_arrays.items()},
            backend=batch_backend,
            earth_rotation=earth_rotation,
        )
        strain = np.asarray(batch.strain)
        frequencies, delta_f, dt = _rfft_grid(strain.shape[-1], sampling_frequency)
        in_band_mask, inverse_psd = in_band_inverse_psd(
            frequencies,
            names,
            sensitivities,
            f_min=minimum_frequency,
            f_max=maximum_frequency,
        )
        snrs[indices] = matched_filter_snr(
            strain, inverse_psd, in_band_mask, delta_f, dt
        )

        done += indices.size
        if progress_callback is not None:
            progress_callback(done, n_events)

    return snrs


def _optimal_snr_lal(
    event_arrays: Mapping[str, NDArray[np.float64]],
    n_events: int,
    detectors: Sequence[CustomDetector],
    names: Sequence[str],
    sensitivities: Mapping[str, Sensitivity],
    waveform_model: str,
    *,
    batch_size: int,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    earth_rotation: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> NDArray[np.float64]:
    """LAL path with sequential generation inside duration-sized batches."""
    mass_1 = event_arrays["detector_frame_mass_1"]
    mass_2 = event_arrays["detector_frame_mass_2"]
    chirp_masses = _chirp_mass(mass_1, mass_2)
    durations = np.asarray(
        [
            segment_sample_count(
                float(chirp_mass), minimum_frequency, sampling_frequency
            )
            / sampling_frequency
            for chirp_mass in chirp_masses
        ],
        dtype=np.float64,
    )

    waveform_keys = [key for key in event_arrays if key not in _RESERVED_PARAMETER_KEYS]
    snrs = np.empty((n_events, len(names)), dtype=np.float64)
    done = 0
    for indices, segment_duration in _duration_sorted_batches(durations, batch_size):
        backend = LALSimulationBackend(segment_duration=segment_duration)
        n_samples = round(segment_duration * sampling_frequency)
        frequencies, delta_f, dt = _rfft_grid(n_samples, sampling_frequency)
        mask, inv_psd = in_band_inverse_psd(
            frequencies,
            names,
            sensitivities,
            f_min=minimum_frequency,
            f_max=maximum_frequency,
        )

        for event in indices:
            params = {key: float(event_arrays[key][event]) for key in waveform_keys}
            polarizations = backend.generate_td_waveform(
                waveform_model,
                tc=float(event_arrays["coa_time"][event]),
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                **params,
            )
            projected = project_polarizations_to_network(
                polarizations,
                list(detectors),
                right_ascension=float(event_arrays["right_ascension"][event]),
                declination=float(event_arrays["declination"][event]),
                polarization_angle=float(event_arrays["polarization_angle"][event]),
                earth_rotation=earth_rotation,
                backend="numpy",
            )
            strain = np.stack(
                [np.asarray(projected[name].value, dtype=np.float64) for name in names]
            )[None, ...]
            snrs[event] = matched_filter_snr(strain, inv_psd, mask, delta_f, dt)[0]

        done += indices.size
        if progress_callback is not None:
            progress_callback(done, n_events)

    return snrs
