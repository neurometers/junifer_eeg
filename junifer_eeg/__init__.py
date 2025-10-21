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
from .datagrabber import ICMLGDataGrabber
from .datareader import EEGDataReader, ICMLGDataReader, SARTDataReader
from .markers import (
    SpectralPower,
    TimeLockedContrast,
    TimeLockedTopography,
    WindowDecoding,
)

# Import MNE data type dumpers
from .mne_asset_dumper import register_mne_dumpers
from .preprocessors import (
    EEGFilter,
    ICMAdaptiveArtifactRejection,
    ICMEquipmentFilter,
    ICMLGEpoching,
)

# Register MNE data types with junifer's DataObjectDumper
register_mne_dumpers()

# Import main classes for convenience

__all__ = [
    "EEGDataReader",
    "EEGFilter",
    "ICMAdaptiveArtifactRejection",
    "ICMEquipmentFilter",
    "ICMLGDataGrabber",
    "ICMLGDataReader",
    "ICMLGEpoching",
    "SARTDataReader",
    "SpectralPower",
    "TimeLockedContrast",
    "TimeLockedTopography",
    "WindowDecoding",
    "datareader",
    "markers",
    "preprocessors",
    "storage",
]
