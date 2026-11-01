"""Ripple adapter: grid arithmetic, parameter mapping, and the waveform kernel.

Everything that knows about ``ripplegw`` lives here. It reimplements what
astrogwb needs from Ripple's public API rather than using ``gwmock-signal``'s
GPL-3 backend, so the package stays MIT; ``gwmock-signal`` is the tests' parity
reference.

No cutoff window is applied: the usual raised cosine is exactly 1 on every
in-band bin, and astrogwb never inverse-transforms.

With ``"linear"`` spacing Ripple sees the full one-sided grid above DC and the
caller slices the result, not the input: ``IMRPhenomXAS_NRTidalv3`` reads
``f_final = f[-1] + df`` and ``df = f[1] - f[0]``. The ``"log"`` and
``"loglinear"`` spacings hand Ripple the band itself; the only reads of the
grid's shape that affect the power are (see
``tests/core/test_waveform_generator.py``):

* ``IMRPhenomD`` and ``IMRPhenomD_NRTidalv2`` snap the merger cutoff to
  ``floor(f_cut / df) * df`` with ``df = f[1] - f[0]``. On a lin-log grid that is
  the linear ``df``, a finer lattice than the grid above the turn, so the
  cutoff is effectively exact.
* ``IMRPhenomXAS_NRTidalv3``'s ``f_final`` enters only the phase slope
  ``linb``; the amplitude never sees it, and the power
  :math:`|h_+|^2 + |h_\times|^2` drops the phase.
* ``IMRPhenomHM`` passes ``f[1] - f[0]`` on as ``deltaF`` and never reads it.

On the linear path the DC bin, which Ripple evaluates to NaN, is dropped from
the input; the top of the grid and its spacing are unchanged.

``ripplegw`` is imported inside function bodies: importing it enables JAX x64
globally, a side effect ``import astrogwb`` must not have.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "build_polarization_kernel",
    "build_power_kernel",
    "check_sources",
    "next_smooth_even",
    "ripple_parameters",
]


class _WaveformNames(Sequence[str]):
    """Lazy view over Ripple's registered waveform names."""

    def __init__(self, **filters: Any) -> None:
        self._filters = filters

    def _names(self) -> tuple[str, ...]:
        # Imported here, not at module scope -- see the module docstring.
        import ripplegw

        return tuple(
            name
            for name in ripplegw.list_waveforms(**self._filters)
            if name not in _EXCLUDED_APPROXIMANTS
        )

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        return self._names()[index]

    def __len__(self) -> int:
        return len(self._names())

    def __repr__(self) -> str:
        return repr(self._names())


_CBC_FILTERS = {"domain": "FD", "source_type": "cbc"}

#: Ripple registers this model; gwmock does not implement it, so it is omitted
#: from every supported-approximant view.
_EXCLUDED_APPROXIMANTS = frozenset({"IMRPhenomXP_NRTidalv3"})

SUPPORTED_APPROXIMANTS = _WaveformNames(**_CBC_FILTERS)
ALIGNED_SPIN_MODELS = _WaveformNames(
    **_CBC_FILTERS, is_precessing=False, is_tidal=False
)
TIDAL_MODELS = _WaveformNames(**_CBC_FILTERS, is_tidal=True)
PRECESSING_MODELS = _WaveformNames(**_CBC_FILTERS, is_precessing=True)

#: Canonical source names this adapter requires of every catalog. The two
#: detector-frame masses may instead be given as source-frame masses and a
#: redshift; see :func:`_canonical_arrays`.
REQUIRED_PARAMETERS = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
)

#: Source-frame names from which the detector-frame masses are derived when
#: the latter are absent: ``m_detector = m_source * (1 + z)``.
_SOURCE_FRAME_MASSES = (
    ("detector_frame_mass_1", "source_frame_mass_1"),
    ("detector_frame_mass_2", "source_frame_mass_2"),
)

#: Canonical source names that default to zero when a catalog omits them.
OPTIONAL_PARAMETERS = (
    "inclination",
    "coa_phase",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
)

#: In-plane spin components, which only a precessing model may carry.
_IN_PLANE_SPINS = ("spin_1x", "spin_1y", "spin_2x", "spin_2y")

#: Prime factors an FFT length may contain.
_SMOOTH_FACTORS = (2, 3, 5)


def _approximant_metadata(approximant: str) -> dict[str, Any]:
    """Return Ripple metadata for a supported frequency-domain CBC model."""
    # Imported here, not at module scope -- see the module docstring.
    import ripplegw

    available = tuple(SUPPORTED_APPROXIMANTS)
    if approximant not in available:
        raise ValueError(
            f"unsupported approximant {approximant!r}; available: {available}"
        )
    return ripplegw.get_waveform_metadata(approximant)


def next_smooth_even(minimum: int) -> int:
    """Return the smallest even integer at least ``minimum`` that is 5-smooth.

    Ripple sizes its segment by this rule, so astrogwb must agree bin for bin.
    """
    if minimum <= _SMOOTH_FACTORS[0]:
        return _SMOOTH_FACTORS[0]
    # Start from the power-of-two ceiling, admissible by construction.
    best = 2
    while best < minimum:
        best *= 2
    power_of_two = 2
    while power_of_two < best:
        times_three = power_of_two
        while times_three < best:
            candidate = times_three
            while candidate < best:
                if candidate >= minimum:
                    best = candidate
                    break
                candidate *= 5
            times_three *= 3
        power_of_two *= 2
    return int(best)


def _as_batch(
    source_parameters: Mapping[str, ArrayLike],
    name: str,
    n_events: int | None,
    *,
    default: float | None = None,
) -> jax.Array:
    """Return one parameter as a 1-D float64 array, validating name and shape.

    Name and shape are static, so these checks are free under tracing.
    """
    if name not in source_parameters:
        if default is None:
            raise ValueError(f"missing required source parameter: {name!r}")
        return jnp.full(n_events, default, dtype=jnp.float64)
    values = jnp.asarray(source_parameters[name], dtype=jnp.float64)
    if values.ndim != 1:
        raise ValueError(
            f"source parameter {name!r} must be 1-D; got shape {values.shape}"
        )
    if n_events is not None and values.shape[0] != n_events:
        raise ValueError(
            f"source parameter {name!r} has length {values.shape[0]}, "
            f"expected {n_events}"
        )
    return values


def _canonical_arrays(
    approximant: str, source_parameters: Mapping[str, ArrayLike]
) -> dict[str, jax.Array]:
    """Return every canonical parameter as a 1-D float64 array, zeros included.

    An absent detector-frame mass is derived as
    :math:`m_{\\mathrm{det}} = m_{\\mathrm{src}} (1 + z)`.

    Raises
    ------
    ValueError
        If a detector-frame mass is absent and cannot be derived, or if any
        other required name is missing or has the wrong shape.
    """
    _approximant_metadata(approximant)
    first, *rest = REQUIRED_PARAMETERS
    arrays = {first: _detector_frame_mass(source_parameters, first, None)}
    n_events = arrays[first].shape[0]
    for name in rest:
        arrays[name] = _detector_frame_mass(source_parameters, name, n_events)
    for name in OPTIONAL_PARAMETERS:
        arrays[name] = _as_batch(source_parameters, name, n_events, default=0.0)
    return arrays


def _detector_frame_mass(
    source_parameters: Mapping[str, ArrayLike], name: str, n_events: int | None
) -> jax.Array:
    """``name`` as a batch, derived from source-frame mass and redshift if absent."""
    source_names = dict(_SOURCE_FRAME_MASSES)
    if name in source_parameters or name not in source_names:
        return _as_batch(source_parameters, name, n_events)
    source_name = source_names[name]
    missing = [
        needed
        for needed in (source_name, "redshift")
        if needed not in source_parameters
    ]
    if missing:
        raise ValueError(
            f"missing required source parameter: {name!r}; give it directly, or "
            f"give {source_name!r} and 'redshift' to derive it "
            f"(also missing: {missing})"
        )
    source_mass = _as_batch(source_parameters, source_name, n_events)
    redshift = _as_batch(source_parameters, "redshift", source_mass.shape[0])
    return source_mass * (1.0 + redshift)


def ripple_parameters(
    approximant: str, source_parameters: Mapping[str, ArrayLike]
) -> dict[str, jax.Array]:
    """Map canonical source names onto the Ripple parameters ``approximant`` takes.

    Validates names and shapes only; physical values are checked eagerly by
    :func:`check_sources`.
    """
    # Imported here, not at module scope -- see the module docstring.
    from ripplegw.conversions import ms_to_Mc_eta

    arrays = _canonical_arrays(approximant, source_parameters)
    metadata = _approximant_metadata(approximant)
    masses = jnp.stack(
        [arrays["detector_frame_mass_1"], arrays["detector_frame_mass_2"]], axis=-1
    )
    chirp_mass, eta = jax.vmap(ms_to_Mc_eta)(masses)
    parameters = {
        "M_c": chirp_mass,
        "eta": eta,
        "s1_z": arrays["spin_1z"],
        "s2_z": arrays["spin_2z"],
        "d_L": arrays["luminosity_distance"],
        "phase_c": arrays["coa_phase"],
        "iota": arrays["inclination"],
    }
    if metadata.get("is_precessing", False):
        parameters |= {
            "s1_x": arrays["spin_1x"],
            "s1_y": arrays["spin_1y"],
            "s2_x": arrays["spin_2x"],
            "s2_y": arrays["spin_2y"],
        }
    if metadata.get("is_tidal", False):
        parameters |= {
            "lambda_1": arrays["lambda_1"],
            "lambda_2": arrays["lambda_2"],
        }
    return parameters


def check_sources(approximant: str, source_parameters: Mapping[str, ArrayLike]) -> None:
    """Check source *values* against what ``approximant`` can represent.

    Eager only; call it on a concrete catalog before inference.

    Raises
    ------
    ValueError
        If a value is unphysical, or names a degree of freedom the approximant
        does not carry (which generation would silently ignore).
    """
    metadata = _approximant_metadata(approximant)
    arrays = _canonical_arrays(approximant, source_parameters)
    if not metadata.get("is_precessing", False):
        for name in _IN_PLANE_SPINS:
            if bool(jnp.any(arrays[name] != 0.0)):
                raise ValueError(
                    f"{approximant} is an aligned-spin model; {name} must be "
                    "zero for every event"
                )
    for name in ("lambda_1", "lambda_2"):
        if bool(jnp.any(arrays[name] < 0.0)):
            raise ValueError(f"{name} must be non-negative")


def build_power_kernel(
    approximant: str,
    reference_frequency: float,
    *,
    use_taper_in_tidal_corrections: bool = True,
) -> Callable[[jax.Array, Mapping[str, jax.Array]], jax.Array]:
    """Return a vmapped ``(frequencies, events) -> power`` evaluator.

    The callable returns frequency-first power ``(F, N)``. For a Ripple
    :class:`~ripplegw.interfaces.AmplitudePhaseWaveform` the phase cancels from
    the power, so only ``amplitude`` is evaluated; other models evaluate full
    polarizations.

    Not jitted: it runs inside NumPyro models, which are already jitted; eager
    callers supply their own jit. ``nan_to_num`` guards the forward sum only;
    it cannot make a NaN gradient finite.
    """
    # Imported here, not at module scope -- see the module docstring.
    from ripplegw.interfaces import AmplitudePhaseWaveform

    waveform = _build_ripple_waveform(
        approximant,
        reference_frequency,
        use_taper_in_tidal_corrections=use_taper_in_tidal_corrections,
    )

    if isinstance(waveform, AmplitudePhaseWaveform):

        def amplitude_power(
            frequencies: jax.Array, events: Mapping[str, jax.Array]
        ) -> jax.Array:
            amplitudes = jax.vmap(
                lambda event: waveform.amplitude(frequencies, dict(event)),
                in_axes=0,
            )(events)
            cosine = jnp.cos(events["iota"])
            inclination_factor = ((1.0 + cosine**2) / 2.0) ** 2 + cosine**2
            power = amplitudes**2 * inclination_factor[:, None]
            return jnp.nan_to_num(power).T.astype(jnp.float64)

        return amplitude_power

    evaluate_polarizations = _polarization_evaluator(waveform)

    def full_polarization_power(
        frequencies: jax.Array, events: Mapping[str, jax.Array]
    ) -> jax.Array:
        plus, cross = evaluate_polarizations(frequencies, events)
        return polarization_power(plus, cross)

    return full_polarization_power


def build_polarization_kernel(
    approximant: str,
    reference_frequency: float,
    *,
    use_taper_in_tidal_corrections: bool = True,
) -> Callable[[jax.Array, Mapping[str, jax.Array]], tuple[jax.Array, jax.Array]]:
    """Return a vmapped ``(frequencies, events) -> (plus, cross)`` evaluator, each ``(N, F)``.

    Every model, amplitude-phase ones included, evaluates its full complex
    polarizations here. Not jitted, and NaN-guarded, as
    :func:`build_power_kernel` is.
    """
    return _polarization_evaluator(
        _build_ripple_waveform(
            approximant,
            reference_frequency,
            use_taper_in_tidal_corrections=use_taper_in_tidal_corrections,
        )
    )


def _build_ripple_waveform(
    approximant: str,
    reference_frequency: float,
    *,
    use_taper_in_tidal_corrections: bool,
) -> Any:
    """Construct the Ripple waveform object once, shared by both kernels."""
    # Imported here, not at module scope -- see the module docstring.
    import ripplegw

    metadata = _approximant_metadata(approximant)
    if not use_taper_in_tidal_corrections and not metadata.get("is_tidal", False):
        raise ValueError(
            "use_taper_in_tidal_corrections=False is only supported for tidal "
            "Ripple approximants"
        )

    config: dict[str, float | bool] = {"f_ref": reference_frequency}
    if not use_taper_in_tidal_corrections:
        config["no_taper"] = True
    try:
        return ripplegw.waveform(approximant, **config)
    except TypeError as error:
        if not use_taper_in_tidal_corrections:
            raise ValueError(
                f"{approximant} is tidal but does not support "
                "use_taper_in_tidal_corrections=False"
            ) from error
        raise


def _polarization_evaluator(
    waveform: Any,
) -> Callable[[jax.Array, Mapping[str, jax.Array]], tuple[jax.Array, jax.Array]]:
    """``(frequencies, events) -> (plus, cross)``, each ``(N, F)`` and NaN-guarded."""

    def one_event(
        frequencies: jax.Array, event: Mapping[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        polarizations = waveform(frequencies, dict(event))
        return (
            jnp.nan_to_num(polarizations["p"]),
            jnp.nan_to_num(polarizations["c"]),
        )

    return jax.vmap(one_event, in_axes=(None, 0))
