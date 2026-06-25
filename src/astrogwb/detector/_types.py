"""Shared detector type aliases.

``DetectorSpec`` is the canonical way to name a detector in astrogwb's
SGWB machinery. It is re-exported from gwmock so the two ecosystems agree
on a single type: a ``str`` (a LAL site code such as ``"H1"`` resolved via
the geometry table) or a ``gwmock_signal`` ``CustomDetector`` carrying its
own geometry (as ET presets do).
"""

from __future__ import annotations

from gwmock_signal.stochastic.overlap import DetectorSpec

__all__ = ["DetectorSpec"]
