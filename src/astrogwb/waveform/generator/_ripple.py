"""Ripple adapter: grid arithmetic, parameter mapping, and the waveform kernel.

Everything that knows about ``ripplegw`` lives here so
:mod:`astrogwb.waveform.generator.ripple` stays a thin descriptor. The split is
also a licensing boundary: ``gwmock-signal`` ships a GPL-3 Ripple backend, and
this module reimplements the parts astrogwb needs from Ripple's public API, so
the package stays MIT. ``gwmock-signal`` remains the parity reference the tests
compare against.

Two deliberate differences from a time-domain backend:

*No cutoff window.* A backend that inverse-transforms has to taper the
amplitude below ``minimum_frequency``, because truncating a nonzero function
rings across the buffer. astrogwb never leaves the frequency domain -- it forms
:math:`|h_+|^2 + |h_\\times|^2` on the in-band slice -- and the usual raised
cosine reaches exactly 1 at ``minimum_frequency``, so on every retained bin the
window is the identity. Applying it would be arithmetic with no effect;
``tests/core/test_waveform_generator.py`` pins that against gwmock rather than
leaving it as a claim here.

*The full grid above DC still reaches Ripple.* Several models --
``IMRPhenomXAS_NRTidalv3`` and ``IMRPhenomXPHM`` among them -- do not evaluate
pointwise in the frequency argument, so handing Ripple only the in-band bins
changes the in-band values. NRTidalv3 shows why: it reads the top of the grid
as ``f_final = f[-1] + df`` and the spacing as ``df = f[1] - f[0]``, and
``f_final`` sets a linear-in-frequency phase slope. Truncate the grid and the
in-band phase winds away from the untruncated answer at fixed amplitude. The
caller slices the result, never the input.

The DC bin is the one exception, and it is dropped at the input rather than
sliced from the output: Ripple evaluates ``f = 0`` to NaN, and while
``nan_to_num`` keeps that out of the forward sum, no output-side treatment
rescues reverse mode (see :func:`build_power_kernel`). Dropping it is safe precisely
because the quantities above are read off the *top* of the grid and off the
spacing, both of which one fewer leading bin leaves untouched.

``ripplegw`` is imported inside function bodies, never at module scope:
importing it enables JAX x64 globally, and ``import astrogwb`` must not carry
that side effect (see the runtime-configuration rule in ``CLAUDE.md`` and
``tests/paper/test_cli.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

__all__ = [
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

        return tuple(ripplegw.list_waveforms(**self._filters))

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

# Compatibility views for callers that used the former module-level groups.
SUPPORTED_APPROXIMANTS = _WaveformNames(**_CBC_FILTERS)
ALIGNED_SPIN_MODELS = _WaveformNames(
    **_CBC_FILTERS, is_precessing=False, is_tidal=False
)
TIDAL_MODELS = _WaveformNames(**_CBC_FILTERS, is_tidal=True)
PRECESSING_MODELS = _WaveformNames(**_CBC_FILTERS, is_precessing=True)

#: Canonical source names this adapter requires of every catalog.
REQUIRED_PARAMETERS = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
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

    5-smooth means no prime factor above 5, which is what transform libraries
    are efficient at; even, because a real transform of ``n`` samples has
    ``n // 2 + 1`` one-sided bins. This is the rule Ripple applies to size its
    segment, so astrogwb has to agree with it bin for bin.

    Every 5-smooth candidate below ``2 * minimum`` is enumerated and the
    smallest admissible one wins. The bound is safe because doubling any
    candidate below ``minimum`` lands below it, so a valid answer always exists
    inside the range.
    """
    if minimum <= _SMOOTH_FACTORS[0]:
        return _SMOOTH_FACTORS[0]
    # The power-of-two ceiling: 5-smooth and even by construction, so the
    # search always starts from an admissible answer and only improves on it.
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

    Name and shape are static under tracing, so these checks cost nothing in a
    traced call and are the only ones this module performs on the hot path.
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
    """Return every canonical parameter as a 1-D float64 array, zeros included."""
    _approximant_metadata(approximant)
    first, *rest = REQUIRED_PARAMETERS
    arrays = {first: _as_batch(source_parameters, first, None)}
    n_events = arrays[first].shape[0]
    for name in rest:
        arrays[name] = _as_batch(source_parameters, name, n_events)
    for name in OPTIONAL_PARAMETERS:
        arrays[name] = _as_batch(source_parameters, name, n_events, default=0.0)
    return arrays


def ripple_parameters(
    approximant: str, source_parameters: Mapping[str, ArrayLike]
) -> dict[str, jax.Array]:
    """Map canonical source names onto the Ripple parameters ``approximant`` takes.

    Validates names and shapes only. Physical values are checked by
    :func:`check_sources`, which a caller runs eagerly -- a traced call cannot
    raise on a value, and forcing one would put a host sync on the hot path.

    The component masses become ``(M_c, eta)`` through
    ``ripplegw.conversions.ms_to_Mc_eta``, which stays in JAX so the mapping is
    differentiable and traceable.
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

    Eager only, and never called from the generate path: a traced array cannot
    drive a Python exception, and converting one to decide would sync the host
    on every batch. Call it once on a concrete catalog -- or on one eager draw
    from a population, via
    :func:`~astrogwb.sampling.validate_source_model` -- before handing the
    model to inference.

    Raises:
        ValueError: If a value is unphysical, or names a degree of freedom the
            approximant does not carry. Silently ignoring the latter is the
            failure this guards: an aligned-spin model hands back a waveform
            for a precessing binary without ever seeing its in-plane spins.
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
    if not metadata.get("is_tidal", False):
        for name in ("lambda_1", "lambda_2"):
            if bool(jnp.any(arrays[name] != 0.0)):
                raise ValueError(
                    f"{approximant} carries no tidal deformability; {name} "
                    "must be zero for every event -- use an NRTidal "
                    "approximant or TaylorF2"
                )
    for name in ("lambda_1", "lambda_2"):
        if bool(jnp.any(arrays[name] < 0.0)):
            raise ValueError(f"{name} must be non-negative")


def build_power_kernel(
    approximant: str, reference_frequency: float
) -> Callable[[jax.Array, Mapping[str, jax.Array]], jax.Array]:
    """Return a vmapped ``(frequencies, events) -> power`` evaluator.

    Constructs the Ripple waveform once; a run has one approximant and one
    reference frequency, so the returned callable is built per generator
    instance and there is nothing left to key a cache on. The callable returns
    frequency-first power with shape ``(F, N)``.

    Ripple's :class:`~ripplegw.interfaces.AmplitudePhaseWaveform` marks models
    whose pre-polarization strain has one real amplitude and the standard
    quadrupolar inclination factors. For those models the phase cancels from
    polarization power exactly, so the selected kernel evaluates only
    ``amplitude``. Other models -- including higher-mode and precessing
    families -- retain the full plus/cross evaluation. The ``isinstance``
    dispatch happens here, before a caller traces the selected array function.

    Deliberately **not** wrapped in :func:`jax.jit`. This is a building block
    called from inside a NumPyro model, which NumPyro already jits under
    inference, so an inner jit would only add a boundary for XLA to inline.
    Callers that evaluate eagerly at catalog scale supply their own jit, where
    it is visible -- see ``scripts/generate_catalog.py``.

    ``nan_to_num`` guards the summed spectrum against a NaN from one event.
    It is not what makes the gradient finite, and it never was: a NaN reaching
    it has a NaN derivative, and masking or slicing the result only zeros that
    bin's cotangent, which leaves ``0 * NaN``. Because source parameters
    broadcast across frequency, their VJP sums every bin and the NaN returns.
    The one NaN astrogwb actually provoked was the DC bin, and
    :class:`~astrogwb.waveform.RippleGenerator` now keeps it out of the input
    grid; this stays as a guard, not as a cure.

    Above a model's cutoff Ripple returns exact zeros, not NaNs -- checked for
    ``IMRPhenomD``, ``TaylorF2``, ``IMRPhenomXAS_NRTidalv3`` and
    ``IMRPhenomXPHM`` across the production band.
    """
    # Imported here, not at module scope -- see the module docstring.
    import ripplegw
    from ripplegw.interfaces import AmplitudePhaseWaveform

    from astrogwb.waveform.polarization_power import polarization_power

    _approximant_metadata(approximant)
    waveform = ripplegw.waveform(approximant, f_ref=reference_frequency)

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

    def one_event(
        frequencies: jax.Array, event: Mapping[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        polarizations = waveform(frequencies, dict(event))
        return (
            jnp.nan_to_num(polarizations["p"]),
            jnp.nan_to_num(polarizations["c"]),
        )

    evaluate_polarizations = jax.vmap(one_event, in_axes=(None, 0))

    def full_polarization_power(
        frequencies: jax.Array, events: Mapping[str, jax.Array]
    ) -> jax.Array:
        plus, cross = evaluate_polarizations(frequencies, events)
        return polarization_power(plus, cross)

    return full_polarization_power
