"""Owned detector specifications."""

from .geometry import Detector

type DetectorSpec = str | Detector

__all__ = ["DetectorSpec"]
