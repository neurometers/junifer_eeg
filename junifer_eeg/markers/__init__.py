"""EEG markers module."""

# Core infrastructure
from .base import EEGBaseMarker, EEGEpochsMarker, EEGRawMarker

# Legacy markers (to be deprecated)
from .contingent_negative_variation import ContingentNegativeVariation
from .eeg_roi_aggregation import EEGROIAggregation
from .kolmogorov_complexity import KolmogorovComplexity
from .permutation_entropy import (
    PermutationEntropy as PermutationEntropyOld,  # Old marker
)

# Refactored hierarchical markers
from .permutation_entropy_new import (
    PermutationEntropy,  # New refactored marker (now includes aggregation)
)
from .permutation_entropy_new._permutation_entropy_base import (
    PermutationEntropyBase,
)
from .power_spectral_density import (
    PowerSpectralDensityEstimator,
    PowerSpectralDensitySummary,
)
from .slow_waves_detection import SlowWavesDetection
from .spectral_power import SpectralPower  # Old marker from .py file
from .spectral_power_new import (
    SpectralPowerBands,  # Now includes aggregation
    SpectralPowerBase,
)
from .spindles_detection import SpindlesDetection
from .symbolic_mutual_information import (
    SymbolicMutualInformation as SymbolicMutualInformationOld,
)
from .symbolic_mutual_information_new._symbolic_mutual_information_base import (
    SymbolicMutualInformationBase,
)
from .symbolic_mutual_information_new.symbolic_mutual_information import (
    SymbolicMutualInformation,
)
from .symbolic_mutual_information_new.symbolic_mutual_information_rois import (
    SymbolicMutualInformationROIs,
)
from .time_decoding import TimeDecoding
from .time_locked_contrast import (
    TimeLockedContrast as TimeLockedContrastOld,  # Legacy
)
from .time_locked_new.time_locked_contrast import (
    TimeLockedContrast,  # New refactored marker (primary)
)
from .time_locked_new.time_locked_topography import (
    TimeLockedTopography,  # New refactored marker (primary)
)
from .time_locked_topography import (
    TimeLockedTopography as TimeLockedTopographyOld,  # Legacy
)
from .window_decoding import WindowDecoding

__all__ = [
    "ContingentNegativeVariation",
    "EEGBaseMarker",
    "EEGEpochsMarker",
    "EEGROIAggregation",
    "EEGRawMarker",
    "KolmogorovComplexity",
    "PermutationEntropy",  # New refactored marker
    "PermutationEntropyBase",
    "PermutationEntropyOld",  # Legacy
    "PowerSpectralDensityEstimator",
    "PowerSpectralDensitySummary",
    "SlowWavesDetection",
    "SpectralPower",
    "SpectralPowerBands",
    "SpectralPowerBase",
    "SpindlesDetection",
    "SymbolicMutualInformation",  # New refactored marker
    "SymbolicMutualInformationBase",
    "SymbolicMutualInformationOld",  # Legacy
    "SymbolicMutualInformationROIs",
    "TimeDecoding",
    "TimeLockedContrast",  # New refactored marker
    "TimeLockedContrastOld",  # Legacy
    "TimeLockedTopography",  # New refactored marker
    "TimeLockedTopographyOld",  # Legacy
    "WindowDecoding",
]
