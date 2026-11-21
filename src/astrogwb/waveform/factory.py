"""Construct waveform generators from their configuration."""

from astrogwb.waveform.analytical import AnalyticInspiralGenerator
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.metadata import ANALYTIC_APPROXIMANT, WaveformConfig
from astrogwb.waveform.ripple import RippleGenerator


def waveform_generator(config: WaveformConfig) -> WaveformGenerator:
    """Construct the generator described by the waveform configuration.

    Parameters
    ----------
    config
        Waveform approximant and frequency settings.
    """
    if config.approximant == ANALYTIC_APPROXIMANT:
        return AnalyticInspiralGenerator(config)
    return RippleGenerator(config)
