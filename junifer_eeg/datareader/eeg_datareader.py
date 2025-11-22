"""EEG data reader extending junifer's DefaultDataReader."""

from pathlib import Path
from typing import Any, Dict, Optional

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


def _read_fif(file_path: Path, **kwargs) -> Any:
    """Read FIF file using MNE."""
    # Handle both raw and epochs FIF files
    if "_epo.fif" in str(file_path):
        # It's an epochs file - try preload=True first, fallback to preload=False
        try:
            return mne.read_epochs(file_path, preload=True, verbose=False)
        except (ValueError, RuntimeError) as e:
            # If preload fails due to calibration issues, try without preload
            print(f"Warning: Preload failed ({e}), trying without preload...")
            epochs = mne.read_epochs(file_path, preload=False, verbose=False)
            # Force load the data to ensure it's available
            epochs.load_data()
            return epochs
    # It's a raw file
    return mne.io.read_raw_fif(file_path, preload=True, verbose=False)


# Add EEG file extensions
default_module._extensions.update(
    {
        ".edf": "EDF",
        ".bdf": "EDF",
        ".fif": "FIF",
    },
)

# Add EEG reader functions
default_module._readers["EDF"] = {"func": _read_edf, "params": None}
default_module._readers["FIF"] = {"func": _read_fif, "params": None}


# Add EEG to the patterns schema using new registration system
try:
    # Try new registration system first
    validation_module.register_data_type(
        "EEG",
        {
            "mandatory": ["pattern", "space"],
            "optional": {
                "mask": {"mandatory": ["pattern", "space"], "optional": []},
            },
        },
    )
except AttributeError:
    # Fallback to old PATTERNS_SCHEMA if available
    if hasattr(validation_module, "PATTERNS_SCHEMA"):
        validation_module.PATTERNS_SCHEMA["EEG"] = {
            "mandatory": ["pattern", "space"],
            "optional": {
                "mask": {"mandatory": ["pattern", "space"], "optional": []},
            },
        }


@register_datareader
class EEGDataReader(DefaultDataReader):
    """EEG data reader for EEG files.

    This class extends DefaultDataReader by adding support for EDF/BDF/FIF files.
    Automatically detects whether FIF files contain raw data or epochs.
    The file reading is handled by extending the global extension mappings.
    Also extends junifer's PatternDataGrabber validation to support EEG data type.
    Applies common processing needed for markers: equipment detection and montage setting.
    """

    def __init__(self) -> None:
        """Initialize EEGDataReader."""
        super().__init__()

    def _fit_transform(
        self,
        input: Dict[str, Dict],
        params: Optional[Dict] = None,
    ) -> Dict:
        """Fit and transform EEG data with equipment detection.

        Applies common processing needed for markers:
        - Equipment detection from channel names
        - Montage setting

        Parameters
        ----------
        input : Dict[str, Dict]
            Input data dictionary
        params : Optional[Dict]
            Additional parameters (unused)

        Returns
        -------
        Dict
            Transformed data with equipment information
        """
        # Use parent class for basic file reading
        output = super()._fit_transform(input, params)

        # Apply equipment detection and montage setting for EEG data
        for data_type, data_info in output.items():
            if data_type == "EEG" and "data" in data_info:
                raw_or_epochs = data_info["data"]

                if isinstance(raw_or_epochs, (mne.io.BaseRaw, mne.BaseEpochs)):
                    from .utils import detect_and_set_equipment

                    # Detect equipment and set montage
                    detect_and_set_equipment(raw_or_epochs)

        return output
