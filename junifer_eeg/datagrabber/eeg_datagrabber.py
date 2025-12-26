"""Vanilla EEG Data Grabber for junifer_eeg.

Simple data grabber with default patterns for EEG files.
Montage setting and equipment detection are handled automatically
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import junifer.datagrabber.pattern_validation_mixin as validation_module
import junifer.datareader.default as default_module
import mne
from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber


def _read_edf(file_path: Path, **kwargs) -> Any:
    """Read EDF file using MNE."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Set montage for consistency with BrainVision format
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_brainvision(file_path: Path, **kwargs) -> Any:
    """Read BrainVision file using MNE."""
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_fif(file_path: Path, **kwargs) -> Any:
    """Read FIF file using MNE."""
    # Handle both raw and epochs FIF files
    if "_epo.fif" in str(file_path) or "-epo.fif" in str(file_path):
        # It's an epochs file - try preload=True first, fallback to preload=False
        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        except (ValueError, RuntimeError) as e:
            # If preload fails due to calibration issues, try without preload
            print(f"Warning: Preload failed ({e}), trying without preload...")
            epochs = mne.read_epochs(file_path, preload=False, verbose=False)
            # Force load the data to ensure it's available
            epochs.load_data()

        # Set montage for epochs
        from .utils import detect_and_set_equipment

        detect_and_set_equipment(epochs)
        return epochs

    # It's a raw file
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    # Set montage for raw files
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_egi_mff(file_path: Path, **kwargs) -> Any:
    """Read EGI MFF file using MNE."""
    raw = mne.io.read_raw_egi(file_path, preload=True, verbose=False)

    # Set montage for EGI data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


# Add EEG file extensions
default_module._extensions.update(
    {
        ".edf": "EDF",
        ".bdf": "EDF",
        ".fif": "FIF",
        ".vhdr": "BrainVision",  # BrainVision header files
        ".mff": "EGI_MFF",  # EGI MFF format (directory-based)
    },
)

# Add EEG reader functions
default_module._readers["EDF"] = {"func": _read_edf, "params": None}
default_module._readers["FIF"] = {"func": _read_fif, "params": None}
default_module._readers["BrainVision"] = {
    "func": _read_brainvision,
    "params": None,
}
default_module._readers["EGI_MFF"] = {
    "func": _read_egi_mff,
    "params": None,
}


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


@register_datagrabber
class EEGDataGrabber(PatternDataGrabber):
    """Vanilla EEG Data Grabber.

    Provides default patterns for EEG file types.
    Montage setting and equipment detection happen automatically
    via the global reader registry (see eeg_datareader.py).
    """

    def __init__(
        self,
        datadir: Union[str, Path],
        types: Optional[List[str]] = None,
        patterns: Optional[Dict] = None,
        replacements: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize EEG Data Grabber.

        Parameters
        ----------
        datadir : str or Path
            Path to the data directory.
        types : list of str, optional
            Data types to grab. If None, defaults to ["EEG"].
        patterns : dict, optional
            Patterns for each data type. If None, uses default EEG patterns.
        replacements : list of str, optional
            Replacement variables. If None, defaults to ["subject"].
        **kwargs
            Additional arguments passed to PatternDataGrabber.
        """
        # Set defaults
        if types is None:
            types = ["EEG"]

        if patterns is None:
            patterns = {
                "EEG": {"pattern": "{subject}/*.fif", "space": "native"}
            }

        if replacements is None:
            replacements = ["subject"]

        super().__init__(
            datadir=datadir,
            types=types,
            patterns=patterns,
            replacements=replacements,
            **kwargs,
        )
