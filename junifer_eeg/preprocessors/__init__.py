"""EEG preprocessors module."""

from .eeg_filter import EEGFilter
from .icm_lg_preprocessing import (
    ICMAdaptiveArtifactRejection,
    ICMEquipmentFilter,
    ICMLGEpoching,
)

__all__ = [
    "EEGFilter",
    "ICMAdaptiveArtifactRejection",
    "ICMEquipmentFilter",
    "ICMLGEpoching",
]
