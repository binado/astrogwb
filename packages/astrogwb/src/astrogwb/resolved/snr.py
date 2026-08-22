"""Per-detector optimal SNR for resolved compact-binary events.

Composes gwmock building blocks into one batched entry point: time-domain
waveform generation (gwmock's LALSimulation backend), projection onto a
detector network with Earth rotation, and a Whittle inner-product
contraction of each projected strain against that detector's PSD. Inverse
PSDs are built once per call and reused across events; only the waveform
generation, projection, and the final inner product run per event.
Progress is reported through an optional caller-provided callback; the
package performs no logging.

Like the rest of astrogwb, the API consumes materialized objects:
``CustomDetector`` instances carrying geometry (see
:func:`astrogwb.detector.resolve_detector`) and a name-keyed
:class:`astrogwb.detector.Sensitivity` mapping (see
:func:`astrogwb.detector.load_sensitivity_map`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from gwmock_signal.detector import CustomDetector
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import Sensitivity

#: Parameters consumed by this module itself (segment placement and sky
#: position); everything else is forwarded to the waveform backend. Names are
#: the gwmock-pop canonical vocabulary.
_RESERVED_PARAMETER_KEYS = frozenset(
    {"coa_time", "right_ascension", "declination", "polarization_angle"}
)

#: Parameters every event must carry: coalescence time and sky/polarization
#: angles for the projection, plus the backend's required intrinsic set.
_REQUIRED_PARAMETER_KEYS = (
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

    missing = [key for key in _REQUIRED_PARAMETER_KEYS if key not in source_parameters]
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
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Single-detector optimal SNR ``sqrt((s|s))`` for every event and detector.

    For each event the pipeline is: generate time-domain plus/cross
    polarizations with gwmock's LALSimulation backend, project them onto
    the detectors (antenna patterns and delays evaluated at time-dependent
    GPS times when ``earth_rotation`` is on), then contract each projected
    strain against that detector's PSD via the frequency-domain inner
    product of Equations 9 and 18 of Cireddu et al. 2025
    (arXiv:2312.14614). Detectors are treated as uncorrelated; the
    uncorrelated network SNR is ``np.sqrt((rho ** 2).sum(axis=1))``.

    All events share one analysis segment sized from the longest inspiral
    in the catalog (rounded up to a power-of-two seconds), so the
    frequency grid -- and with it the inverse PSDs -- is computed once and
    reused. Events are processed one at a time, so peak memory is one
    event's strain regardless of catalog size.

    Parameters
    ----------
    source_parameters:
        Per-event parameters as 1-dimensional arrays, all of equal length,
        named in the gwmock-pop canonical vocabulary. Required keys:
        ``coa_time``, ``right_ascension``, ``declination``,
        ``polarization_angle``, ``detector_frame_mass_1``,
        ``detector_frame_mass_2``, ``luminosity_distance``. Remaining keys (spins, ``inclination``,
        ``coa_phase``, ``lambda_1``, ``lambda_2``) are forwarded to the
        waveform backend, which rejects unknown ones.
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
        LAL approximant name, e.g. ``"IMRPhenomXAS_NRTidalv3"``.
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
    progress_callback:
        Optional callable invoked as ``(done, total)`` after each event.
        The package itself performs no logging; callers may log from the
        callback (throttling there if desired).

    Returns
    -------
    numpy.ndarray
        Optimal SNR per event and detector, shape
        ``(n_events, n_detectors)``, non-negative. Column ``j`` is
        ``detectors[j]``.
    """
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

    # Heavy imports stay inside the function so importing this module stays cheap.
    from gwmock_signal.projection.network import project_polarizations_to_network
    from gwmock_signal.waveform.backends.conditioning import segment_sample_count
    from gwmock_signal.waveform.backends.lal import LALSimulationBackend

    # One common segment: size it from the largest chirp mass in the catalog,
    # rounded up to a power-of-two seconds exactly as the backend would.
    mass_1 = event_arrays["detector_frame_mass_1"]
    mass_2 = event_arrays["detector_frame_mass_2"]
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

    psd_stack = np.stack(
        [
            sensitivities[name].evaluate(frequencies, out_of_band="zero")
            for name in names
        ]
    )
    inv_psd = 1.0 / psd_stack[:, mask]

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

        strains = np.array([np.fft.rfft(projected[name].value) * dt for name in names])
        in_band = strains[:, mask]
        integrand = (np.abs(in_band) ** 2 * inv_psd).sum(axis=1).real
        snrs[event] = np.sqrt(np.maximum(4.0 * delta_f * integrand, 0.0))

        if progress_callback is not None:
            progress_callback(event + 1, n_events)

    return snrs
