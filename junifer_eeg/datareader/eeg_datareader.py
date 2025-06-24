"""EEG data reader extending junifer's DefaultDataReader."""

from pathlib import Path
from typing import Any

import junifer.datagrabber.pattern_validation_mixin as validation_module

# Extend the global extension and reader mappings used by DefaultDataReader
# Import the modules to access the global variables
import junifer.datareader.default as default_module
import mne
from junifer.api.decorators import register_datareader
from junifer.datareader import DefaultDataReader


def _read_edf(file_path: Path, **kwargs) -> Any:
    """Read EDF file using MNE."""
    return mne.io.read_raw_edf(file_path, preload=True, verbose=False)


# Add EEG file extensions
default_module._extensions.update(
    {
        ".edf": "EDF",
        ".bdf": "EDF",
    }
)

# Add EEG reader function
default_module._readers["EDF"] = {"func": _read_edf, "params": None}


# Add EEG to the patterns schema
validation_module.PATTERNS_SCHEMA["EEG"] = {
    "mandatory": ["pattern", "space"],
    "optional": {
        "mask": {"mandatory": ["pattern", "space"], "optional": []},
    },
}


@register_datareader
class EEGDataReader(DefaultDataReader):
    """EEG data reader for EDF files.

    This class extends DefaultDataReader by adding support for EDF/BDF files.
    The file reading is handled by extending the global extension mappings.
    Also extends junifer's PatternDataGrabber validation to support EEG data type.
    """

    def __init__(self) -> None:
        """Initialize EEGDataReader."""
        super().__init__()
