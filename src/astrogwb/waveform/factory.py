"""Construct waveform generators from their metadata."""

from astrogwb.waveform.analytical import AnalyticInspiralGenerator
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.metadata import ANALYTIC_APPROXIMANT, WaveformMetadata
from astrogwb.waveform.ripple import RippleGenerator


def waveform_generator(metadata: WaveformMetadata) -> WaveformGenerator:
    """Construct the generator described by the waveform metadata."""
    if metadata.approximant == ANALYTIC_APPROXIMANT:
        return AnalyticInspiralGenerator(metadata)
    return RippleGenerator(metadata)
