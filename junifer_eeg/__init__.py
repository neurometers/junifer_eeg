"""JUnifer EEG - EEG extension for the JUelich NeuroImaging FEature extractoR."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Giovanni Marraffini, Fede Raimondo"
__email__ = "g.marraffini@neurometers.ai"


# Import main modules
from . import (
    datareader,
    markers,
    preprocessors,
    storage,
)

# Import main classes for convenience
from .datareader import EEGDataReader, ICMLGDataReader
from .markers import (
    SpectralPower,
    TimeLockedContrast,
    TimeLockedTopography,
    WindowDecoding,
)
from .preprocessors import (
    EEGFilter,
    ICMAdaptiveArtifactRejection,
    ICMEquipmentFilter,
    ICMLGEpoching,
)

__all__ = [
    "EEGDataReader",
    "EEGFilter",
    "ICMAdaptiveArtifactRejection",
    "ICMEquipmentFilter",
    "ICMLGDataReader",
    "ICMLGEpoching",
    "SpectralPower",
    "TimeLockedContrast",
    "TimeLockedTopography",
    "WindowDecoding",
    "datareader",
    "markers",
    "preprocessors",
]
