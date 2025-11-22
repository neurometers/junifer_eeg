"""JUnifer EEG - EEG extension for the JUelich NeuroImaging FEature extractoR."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# Import main modules
from . import (
    datareader,
    markers,
    preprocessors,
)
from .datagrabber import EEGDataGrabber
from .datareader import EEGDataReader
from .markers import (
    SpectralPower,
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
    "EEGDataReader",
    "EEGEpoching",
    "EEGFilter",
    "EEGInterpolation",
    "EEGReference",
    "SpectralPower",
    "TimeLockedContrast",
    "TimeLockedTopography",
    "WindowDecoding",
    "datareader",
    "markers",
    "preprocessors",
]
