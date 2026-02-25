from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .grid import FrequencyGrid
from .generator import SourceType
from .polarizations import WaveformPolarizations

if TYPE_CHECKING:
    from bilby.gw.waveform_generator import WaveformGenerator as BilbyWFG

GWSIGNAL_WAVEFORM_APPROXIMANTS = frozenset(["SEOBNRv5HM", "SEOBNRv5PHM"])


class BilbyWaveformBackend:
    """Stateful bilby-backed waveform backend.

    The bilby WaveformGenerator is constructed lazily on first use and cached.
    All bilby imports are deferred so the module is importable without bilby.
    """

    def __init__(
        self, approximant: str, grid: FrequencyGrid, source_type: SourceType
    ) -> None:
        self._approximant = approximant
        self._grid = grid
        self._source_type = source_type

    @cached_property
    def _waveform_generator(self) -> BilbyWFG:
        from bilby.gw.conversion import (
            convert_to_lal_binary_black_hole_parameters,
            convert_to_lal_binary_neutron_star_parameters,
        )
        from bilby.gw.source import (
            gwsignal_binary_black_hole,
            lal_binary_black_hole,
            lal_binary_neutron_star,
        )
        from bilby.gw.waveform_generator import WaveformGenerator as BilbyWFG

        if self._source_type == "BBH":
            source_model = (
                gwsignal_binary_black_hole
                if self._approximant in GWSIGNAL_WAVEFORM_APPROXIMANTS
                else lal_binary_black_hole
            )
            parameter_conversion = convert_to_lal_binary_black_hole_parameters
        else:
            source_model = lal_binary_neutron_star
            parameter_conversion = convert_to_lal_binary_neutron_star_parameters

        waveform_arguments = {
            "waveform_approximant": self._approximant,
            "reference_frequency": self._grid.reference_frequency,
        }

        return BilbyWFG(
            parameters=None,
            frequency_domain_source_model=source_model,
            duration=self._grid.duration,
            sampling_frequency=self._grid.sampling_frequency,
            parameter_conversion=parameter_conversion,
            waveform_arguments=waveform_arguments,
        )

    def frequency_domain_polarizations(
        self, parameters: dict[str, float]
    ) -> WaveformPolarizations:
        result = self._waveform_generator.frequency_domain_strain(parameters)
        return WaveformPolarizations(
            grid=self._grid, hp=result["plus"], hc=result["cross"]
        )
