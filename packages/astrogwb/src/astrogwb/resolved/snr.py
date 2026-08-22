"""Per-detector optimal SNR for resolved compact-binary events.

Composes gwmock building blocks into one batched entry point: waveform
generation, projection onto a detector network with Earth rotation, and a
Whittle inner-product contraction of each projected strain against that
detector's PSD. Two generation backends are supported; see
:func:`optimal_snr` for the selection rule. Inverse PSDs are built once
per call and reused across events. Progress is reported through an
optional caller-provided callback; the package performs no logging.

Like the rest of astrogwb, the API consumes materialized objects:
``CustomDetector`` instances carrying geometry (see
:func:`astrogwb.detector.resolve_detector`) and a name-keyed
:class:`astrogwb.detector.Sensitivity` mapping (see
:func:`astrogwb.detector.load_sensitivity_map`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from gwmock_signal.detector import CustomDetector
from gwmock_signal.jax_batch import recommend_chunk_size, simulate_cbc_batch
from gwmock_signal.projection.network import project_polarizations_to_network
from gwmock_signal.waveform.backends.conditioning import segment_sample_count
from gwmock_signal.waveform.backends.lal import LALSimulationBackend
from gwmock_signal.waveform.backends.ripple import RippleBackend
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import Sensitivity

WaveformBackend = Literal["auto", "ripple", "lal"]

#: Parameters consumed by this module itself (segment placement and sky
#: position); everything else is forwarded to the waveform backend. Names are
#: the gwmock-pop canonical vocabulary.
_RESERVED_PARAMETER_KEYS = frozenset(
    {"coa_time", "right_ascension", "declination", "polarization_angle"}
)

#: Parameters every event must carry: coalescence time and sky/polarization
#: angles for the projection, plus the backend's required intrinsic set.
REQUIRED_SOURCE_PARAMETERS = (
    "coa_time",
    "right_ascension",
    "declination",
    "polarization_angle",
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
)


def _normalize_parameters(
    source_parameters: Mapping[str, ArrayLike],
) -> tuple[dict[str, NDArray[np.float64]], int]:
    """Coerce event parameters to 1-D float64 arrays of one shared length.

    Returns ``(event_arrays, n_events)``. Every value must be a
    non-empty 1-dimensional array of the same length, so downstream code
    sees one consistent event count.
    """
    if not source_parameters:
        raise ValueError("source_parameters must not be empty.")

    missing = [
        key for key in REQUIRED_SOURCE_PARAMETERS if key not in source_parameters
    ]
    if missing:
        raise ValueError(f"Missing required source parameters: {missing}")

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


def _in_band_inverse_psd(
    frequencies: NDArray[np.float64],
    names: Sequence[str],
    sensitivities: Mapping[str, Sensitivity],
    f_min: float,
    f_max: float | None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Boolean in-band mask and ``1/S(f)`` with shape ``(n_detectors, n_in_band)``."""
    f_high = f_max if f_max is not None else float(frequencies[-1])
    mask = (frequencies >= f_min) & (frequencies <= f_high)
    psd_stack = np.stack(
        [
            sensitivities[name].evaluate(frequencies, out_of_band="zero")
            for name in names
        ]
    )
    return mask, 1.0 / psd_stack[:, mask]


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
    maximum_frequency: float | None = None,
    earth_rotation: bool = True,
    backend: WaveformBackend = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Single-detector optimal SNR ``sqrt((s|s))`` for every event and detector.

    Two generation backends are available. With ``backend="auto"`` (the
    default) the ripple (JAX) backend is used when it is installed and
    supports ``waveform_model``: the whole catalog is generated under
    ``jax.vmap`` and projected on device in memory-sized chunks. Otherwise
    the LALSimulation backend is used as a per-event, time-domain fallback.
    ``backend="ripple"`` requires a usable ripple backend;
    ``backend="lal"`` skips ripple. Either way each event/detector strain
    is contracted against that detector's PSD via the frequency-domain inner
    product of Equations 9 and 18 of Cireddu et al. 2025 (arXiv:2312.14614).
    Detector antenna patterns and delays are evaluated at time-dependent GPS
    times when ``earth_rotation`` is on. Detectors are treated as
    uncorrelated; the uncorrelated network SNR is
    ``np.sqrt((rho ** 2).sum(axis=1))``.

    All events share one analysis segment sized from the longest inspiral
    in the catalog, so the frequency grid -- and with it the inverse
    PSDs -- is computed once and reused. Peak memory is one batch chunk on
    the ripple path, sized from the device-memory limit gwmock reports
    (the whole catalog when no limit is reported, e.g. on CPU); the LAL
    path holds one event's strain regardless of catalog size.

    Parameters
    ----------
    source_parameters:
        Per-event parameters as 1-dimensional arrays, all of equal length,
        named in the gwmock-pop canonical vocabulary. Required keys are
        :data:`REQUIRED_SOURCE_PARAMETERS`. Remaining keys (spins,
        ``inclination``, ``coa_phase``, ``lambda_1``, ``lambda_2``) are
        forwarded to the waveform backend, which rejects unknown ones.
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
        work unit (one ripple chunk, or one LAL event). ``done`` is the
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


def _optimal_snr_ripple_batch(
    backend: RippleBackend,
    event_arrays: Mapping[str, NDArray[np.float64]],
    n_events: int,
    detectors: Sequence[CustomDetector],
    names: Sequence[str],
    sensitivities: Mapping[str, Sensitivity],
    waveform_model: str,
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    earth_rotation: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> NDArray[np.float64]:
    """Batched ripple (JAX) path: vmapped generation, on-device projection.

    The segment duration is pinned once for the whole catalog, so every
    chunk lands on the same frequency grid and the inverse PSDs built from
    the first chunk's grid serve all chunks. Chunk size follows the
    device-memory limit gwmock reports; when no limit is reported (e.g.
    CPU) the whole catalog is processed as one batch.
    """
    mass_1 = event_arrays["detector_frame_mass_1"]
    mass_2 = event_arrays["detector_frame_mass_2"]
    chirp_masses = _chirp_mass(mass_1, mass_2)
    eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2
    segment_duration = backend.segment_duration_for(
        chirp_masses, minimum_frequency, sampling_frequency, eta=eta
    )
    backend = backend.with_segment_duration(segment_duration)

    # Conservative overestimate of the shared grid length (the exact
    # 5-smooth even length is at most twice), used only to size the
    # memory-limited chunk: overestimating costs a smaller chunk,
    # underestimating costs an out-of-memory abort.
    n_samples_bound = 2 * math.ceil(segment_duration * sampling_frequency) + 2
    chunk_size = recommend_chunk_size(
        len(names), n_samples_bound, earth_rotation=earth_rotation
    )
    if chunk_size is None:
        chunk_size = n_events

    snrs = np.empty((n_events, len(names)), dtype=np.float64)
    inverse_psd: NDArray[np.float64] | None = None
    in_band_mask: NDArray[np.bool_] | None = None
    delta_f = 0.0
    dt = 0.0
    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        batch = simulate_cbc_batch(
            waveform_model,
            list(detectors),
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            parameters={
                key: values[start:stop] for key, values in event_arrays.items()
            },
            backend=backend,
            earth_rotation=earth_rotation,
        )
        strain = np.asarray(batch.strain)
        if inverse_psd is None:
            frequencies, delta_f, dt = _rfft_grid(strain.shape[-1], sampling_frequency)
            in_band_mask, inverse_psd = _in_band_inverse_psd(
                frequencies,
                names,
                sensitivities,
                minimum_frequency,
                maximum_frequency,
            )
        assert inverse_psd is not None and in_band_mask is not None  # set together

        snrs[start:stop] = matched_filter_snr(
            strain, inverse_psd, in_band_mask, delta_f, dt
        )

        if progress_callback is not None:
            progress_callback(stop, n_events)

    return snrs


def _optimal_snr_lal(
    event_arrays: Mapping[str, NDArray[np.float64]],
    n_events: int,
    detectors: Sequence[CustomDetector],
    names: Sequence[str],
    sensitivities: Mapping[str, Sensitivity],
    waveform_model: str,
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    earth_rotation: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> NDArray[np.float64]:
    """Per-event LALSimulation path; peak memory is one event's strain."""
    # One common segment: size it from the largest chirp mass in the catalog,
    # rounded up to a power-of-two seconds exactly as the backend would.
    mass_1 = event_arrays["detector_frame_mass_1"]
    mass_2 = event_arrays["detector_frame_mass_2"]
    chirp_masses = _chirp_mass(mass_1, mass_2)
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
    frequencies, delta_f, dt = _rfft_grid(n_samples, sampling_frequency)
    mask, inv_psd = _in_band_inverse_psd(
        frequencies, names, sensitivities, minimum_frequency, maximum_frequency
    )

    waveform_keys = [key for key in event_arrays if key not in _RESERVED_PARAMETER_KEYS]
    snrs = np.empty((n_events, len(names)), dtype=np.float64)
    for event in range(n_events):
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

        if progress_callback is not None:
            progress_callback(event + 1, n_events)

    return snrs
