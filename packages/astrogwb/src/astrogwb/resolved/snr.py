"""Network optimal SNR for resolved compact-binary events.

Composes gwmock building blocks into one batched entry point: time-domain
waveform generation (gwmock's LALSimulation backend), projection onto a
detector network with Earth rotation, and a Whittle inner-product
contraction against the network noise. The inverse spectral noise matrix is
built once per call and reused across events; only the waveform generation,
projection, and the final quadratic form run per event. Progress is reported
through an optional caller-provided callback; the package performs no
logging.

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


def _noise_inverse(
    psds: dict[str, NDArray[np.float64]],
    names: Sequence[str],
    mask: NDArray[np.bool_],
    *,
    cross_psds: Mapping[tuple[str, str], ArrayLike] | None = None,
) -> NDArray[np.complex128]:
    """Build the masked inverse spectral noise matrix ``S_n^{-1}(f)``.

    With ``cross_psds=None`` the covariance is diagonal, so the inverse is
    the multiplicative inverse of the PSDs and no matrix inversion is
    performed. Otherwise the full Hermitian matrix (diagonal one-sided
    PSDs, Hermitian cross-PSDs, as in ``gwmock_signal.snr._network``) is
    inverted on the in-band bins only. Out-of-band bins are excluded before
    inversion, so zero-valued PSD entries there cannot cause singular
    matrices.
    """
    n_det = len(names)
    psd_stack = np.stack([psds[name] for name in names])
    diagonal = np.arange(n_det)

    if cross_psds is None:
        n_inband = int(mask.sum())
        inverse = np.zeros((n_inband, n_det, n_det), dtype=np.complex128)
        inverse[:, diagonal, diagonal] = (1.0 / psd_stack[:, mask]).T
        return inverse

    n_freq = mask.shape[0]
    index = {name: i for i, name in enumerate(names)}
    noise = np.zeros((n_det, n_det, n_freq), dtype=np.complex128)
    noise[diagonal, diagonal, :] = psd_stack
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
    detectors: Sequence[CustomDetector],
    sensitivities: Mapping[str, Sensitivity],
    *,
    waveform_model: str,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None = None,
    cross_psds: Mapping[tuple[str, str], ArrayLike] | None = None,
    earth_rotation: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Network optimal SNR ``sqrt((s|s))`` for every event in a catalog.

    For each event the pipeline is: generate time-domain plus/cross
    polarizations with gwmock's LALSimulation backend, project them onto
    the detectors (antenna patterns and delays evaluated at time-dependent
    GPS times when ``earth_rotation`` is on), then contract the projected
    strains against the network noise via the frequency-domain inner
    product of Equations 9 and 18 of Cireddu et al. 2025
    (arXiv:2312.14614).

    All events share one analysis segment sized from the longest inspiral
    in the catalog (rounded up to a power-of-two seconds), so the
    frequency grid -- and with it the inverted spectral noise matrix -- is
    computed once and reused. Events are processed one at a time, so peak
    memory is one event's strain regardless of catalog size.

    Parameters
    ----------
    source_parameters:
        Per-event parameters as 1-dimensional arrays, all of equal length.
        Required keys: ``tc``, ``ra``, ``dec``, ``psi``,
        ``detector_frame_mass_1``, ``detector_frame_mass_2``,
        ``luminosity_distance``. Remaining keys (spins, ``inclination``,
        ``coa_phase``, ``lambda_1``, ``lambda_2``) are forwarded to the
        waveform backend, which rejects unknown ones.
    detectors:
        Resolved detector geometries (gwmock ``CustomDetector`` instances,
        e.g. from :func:`astrogwb.detector.resolve_detector`). Names must
        be unique.
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
    cross_psds:
        Optional off-diagonal cross-PSDs keyed by ordered detector-name
        tuples; Hermitian symmetry is applied automatically. ``None``
        (default) gives the uncorrelated limit, where the inverse noise
        matrix is diagonal and no matrix inversion is performed.
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
        Network optimal SNR per event, shape ``(n_events,)``,
        non-negative.
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

    psds = {
        name: sensitivities[name].evaluate(frequencies, out_of_band="zero")
        for name in names
    }
    noise_inverse = _noise_inverse(psds, names, mask, cross_psds=cross_psds)

    waveform_keys = [key for key in event_arrays if key not in _RESERVED_PARAMETER_KEYS]
    snrs = np.empty(n_events, dtype=np.float64)
    for event in range(n_events):
        params = {key: float(event_arrays[key][event]) for key in waveform_keys}
        polarizations = backend.generate_td_waveform(
            waveform_model,
            tc=float(event_arrays["tc"][event]),
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            **params,
        )
        projected = project_polarizations_to_network(
            polarizations,
            list(detectors),
            right_ascension=float(event_arrays["ra"][event]),
            declination=float(event_arrays["dec"][event]),
            polarization_angle=float(event_arrays["psi"][event]),
            earth_rotation=earth_rotation,
            backend="numpy",
        )

        strains = np.array([np.fft.rfft(projected[name].value) * dt for name in names])
        in_band = strains[:, mask]
        integrand = np.einsum(
            "ik,kij,jk->", in_band.conj(), noise_inverse, in_band
        ).real
        snrs[event] = np.sqrt(max(4.0 * delta_f * integrand, 0.0))

        if progress_callback is not None:
            progress_callback(event + 1, n_events)

    return snrs
