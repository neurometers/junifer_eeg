"""JUnifer EEG - EEG extension for the JUelich NeuroImaging FEature extractoR."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# Import main modules
from . import (
    datagrabber,
    markers,
    preprocessors,
    storage,
)
from .datagrabber import EEGDataGrabber
from .markers import (
    SpectralPowerBands,
    TimeLockedContrast,
    TimeLockedTopography,
    WindowDecoding,
)

# Import MNE data type dumpers
from .mne_asset_dumper import register_mne_dumpers
from .preprocessors import (
    BadChannelsHighFrequency,
    BadChannelsThreshold,
    BadChannelsVariance,
    BadEpochsThreshold,
    EEGEpoching,
    EEGFilter,
    EEGInterpolation,
    EEGReference,
)

# Register MNE data types with junifer's DataObjectDumper
register_mne_dumpers()

# Import main classes for convenience

__all__ = [
    "BadChannelsHighFrequency",
    "BadChannelsThreshold",
    "BadChannelsVariance",
    "BadEpochsThreshold",
    "EEGDataGrabber",
    "EEGEpoching",
    "EEGFilter",
    "EEGInterpolation",
    "EEGReference",
    "SpectralPowerBands",
    "TimeLockedContrast",
    "TimeLockedTopography",
    "WindowDecoding",
    "datagrabber",
    "markers",
    "preprocessors",
    "storage",
]
