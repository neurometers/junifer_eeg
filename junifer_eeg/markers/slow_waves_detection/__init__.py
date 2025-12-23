"""Refactored Slow Waves Detection markers."""

from ._slow_waves_detection_base import (
    SLOW_WAVE_FEATURES,
    SlowWavesDetectionBase,
)
from .slow_waves_detection import SlowWavesDetection

__all__ = [
    "SLOW_WAVE_FEATURES",
    "SlowWavesDetection",
    "SlowWavesDetectionBase",
]
