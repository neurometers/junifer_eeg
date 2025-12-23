"""EEG markers module."""

# Core infrastructure
from .base import (
    EEGBaseMarker,
    EEGEpochsMarker,
    EEGRawMarker,
    format_marker_result,
)

# Standalone markers (not refactored to folder structure)
from .contingent_negative_variation import ContingentNegativeVariation

# Decoding markers
from .decoding import TimeDecoding, WindowDecoding
from .eeg_roi_aggregation import EEGROIAggregation

# Kolmogorov Complexity
from .kolmogorov_complexity import KolmogorovComplexity
from .kolmogorov_complexity._kolmogorov_complexity_base import (
    KolmogorovComplexityBase,
)

# Permutation Entropy
from .permutation_entropy import PermutationEntropy
from .permutation_entropy._permutation_entropy_base import (
    PermutationEntropyBase,
)

# Power Spectral Density
from .power_spectral_density import (
    PowerSpectralDensityEstimator,
    PowerSpectralDensitySummary,
)

# Slow Waves Detection
from .slow_waves_detection import (
    SLOW_WAVE_FEATURES,
    SlowWavesDetection,
    SlowWavesDetectionBase,
)

# Spectral Power
from .spectral_power import SpectralPowerBands, SpectralPowerBase

# Spindles Detection
from .spindles_detection import (
    SPINDLE_FEATURES,
    SpindlesDetection,
    SpindlesDetectionBase,
)

# Symbolic Mutual Information
from .symbolic_mutual_information import (
    SymbolicMutualInformation,
    SymbolicMutualInformationBase,
    SymbolicMutualInformationROIs,
)

# Time-Locked markers
from .time_locked import TimeLockedContrast, TimeLockedTopography

__all__ = [
    "SLOW_WAVE_FEATURES",
    "SPINDLE_FEATURES",
    "ContingentNegativeVariation",
    "EEGBaseMarker",
    "EEGEpochsMarker",
    "EEGROIAggregation",
    "EEGRawMarker",
    "KolmogorovComplexity",
    "KolmogorovComplexityBase",
    "PermutationEntropy",
    "PermutationEntropyBase",
    "PowerSpectralDensityEstimator",
    "PowerSpectralDensitySummary",
    "SlowWavesDetection",
    "SlowWavesDetectionBase",
    "SpectralPowerBands",
    "SpectralPowerBase",
    "SpindlesDetection",
    "SpindlesDetectionBase",
    "SymbolicMutualInformation",
    "SymbolicMutualInformationBase",
    "SymbolicMutualInformationROIs",
    "TimeDecoding",
    "TimeLockedContrast",
    "TimeLockedTopography",
    "WindowDecoding",
    "format_marker_result",
]
