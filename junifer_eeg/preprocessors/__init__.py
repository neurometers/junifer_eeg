"""EEG preprocessors module."""

from .artifact_rejection import (
    BadChannelsHighFrequency,
    BadChannelsThreshold,
    BadChannelsVariance,
    BadEpochsThreshold,
)
from .eeg_epoching import EEGEpoching
from .eeg_filter import EEGFilter
from .eeg_reference import EEGReference
from .interpolation import EEGInterpolation

__all__ = [
    "BadChannelsHighFrequency",
    "BadChannelsThreshold",
    "BadChannelsVariance",
    "BadEpochsThreshold",
    "EEGEpoching",
    "EEGFilter",
    "EEGInterpolation",
    "EEGReference",
]
