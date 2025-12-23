"""Refactored Spindles Detection markers."""

from ._spindles_detection_base import SPINDLE_FEATURES, SpindlesDetectionBase
from .spindles_detection import SpindlesDetection

__all__ = ["SPINDLE_FEATURES", "SpindlesDetection", "SpindlesDetectionBase"]
