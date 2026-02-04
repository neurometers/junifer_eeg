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


def _read_eeglab(file_path: Path, **kwargs) -> Any:
    """Read EEGLAB .set file using MNE.

    Handles both raw continuous and epoched data.
    """
    from .utils import detect_and_set_equipment

    # Try to read as raw continuous data first
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        detect_and_set_equipment(raw)
        return raw
    except (ValueError, TypeError) as e:
        # If reading as raw fails, try as epoched data
        try:
            epochs = mne.read_epochs_eeglab(file_path, verbose=False)
            # Ensure data is loaded
            if not epochs.preload:
                epochs.load_data()
            detect_and_set_equipment(epochs)
            return epochs
        except Exception:
            # Re-raise the original error if both fail
            raise e from None


def _read_cnt(file_path: Path, **kwargs) -> Any:
    """Read Neuroscan .cnt file using MNE."""
    raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)

    # Set montage for Neuroscan data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_gdf(file_path: Path, **kwargs) -> Any:
    """Read GDF (General Data Format) file using MNE."""
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

    # Set montage for GDF data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_nihon_kohden(file_path: Path, **kwargs) -> Any:
    """Read Nihon Kohden .eeg file using MNE."""
    raw = mne.io.read_raw_nihon(file_path, preload=True, verbose=False)

    # Set montage for Nihon Kohden data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_raw_fif(file_path: Path, **kwargs) -> Any:
    """Read standalone .raw file (FIF format) using MNE."""
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    # Set montage
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_kit(file_path: Path, **kwargs) -> Any:
    """Read KIT/Yokogawa .sqd/.con file using MNE."""
    raw = mne.io.read_raw_kit(file_path, preload=True, verbose=False)

    # Set montage for KIT data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_curry(file_path: Path, **kwargs) -> Any:
    """Read Curry .cdt/.dap file using MNE."""
    raw = mne.io.read_raw_curry(file_path, preload=True, verbose=False)

    # Set montage for Curry data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_nicolet(file_path: Path, **kwargs) -> Any:
    """Read Nicolet .data file using MNE."""
    raw = mne.io.read_raw_nicolet(file_path, preload=True, verbose=False)

    # Set montage for Nicolet data
    from .utils import detect_and_set_equipment

    detect_and_set_equipment(raw)

    return raw


def _read_fieldtrip(file_path: Path, **kwargs) -> Any:
    """Read FieldTrip .mat file using MNE.

    Handles both raw and epoched FieldTrip data.
    """
    from .utils import detect_and_set_equipment

    # FieldTrip can contain raw or epoched data
    try:
        # Try reading as epochs first (more common for preprocessed data)
        epochs = mne.read_epochs_fieldtrip(file_path, info=None, verbose=False)
        if not epochs.preload:
            epochs.load_data()
        detect_and_set_equipment(epochs)
        return epochs
    except (ValueError, TypeError, KeyError) as e:
        # If epochs fail, it might not be FieldTrip format or might need info
        # For now, raise informative error
        raise ValueError(
            f"Could not read {file_path} as FieldTrip format. "
            "FieldTrip .mat files may require additional info parameter."
        ) from e


# Add EEG file extensions
default_module._extensions.update(
    {
        ".edf": "EDF",
        ".bdf": "EDF",  # BioSemi variant of EDF
        ".gdf": "GDF",  # General Data Format (EDF successor)
        ".fif": "FIF",
        ".raw": "RAW_FIF",  # Standalone raw FIF files
        ".vhdr": "BrainVision",  # BrainVision header files
        ".eeg": "NIHON_KOHDEN",  # Nihon Kohden clinical EEG
        ".mff": "EGI_MFF",  # EGI MFF format (directory-based)
        ".set": "EEGLAB",  # EEGLAB format (with .fdt for data)
        ".cnt": "CNT",  # Neuroscan format
        ".sqd": "KIT",  # KIT/Yokogawa format
        ".con": "KIT",  # KIT/Yokogawa continuous format
        ".cdt": "CURRY",  # Curry 7/8 data file
        ".dap": "CURRY",  # Curry data file (older versions)
        ".data": "NICOLET",  # Nicolet clinical EEG
        ".mat": "FIELDTRIP",  # FieldTrip MATLAB format
    },
)

# Add EEG reader functions
default_module._readers["EDF"] = {"func": _read_edf, "params": None}
default_module._readers["GDF"] = {"func": _read_gdf, "params": None}
default_module._readers["FIF"] = {"func": _read_fif, "params": None}
default_module._readers["RAW_FIF"] = {"func": _read_raw_fif, "params": None}
default_module._readers["BrainVision"] = {
    "func": _read_brainvision,
    "params": None,
}
default_module._readers["NIHON_KOHDEN"] = {
    "func": _read_nihon_kohden,
    "params": None,
}
default_module._readers["EGI_MFF"] = {
    "func": _read_egi_mff,
    "params": None,
}
default_module._readers["EEGLAB"] = {
    "func": _read_eeglab,
    "params": None,
}
default_module._readers["CNT"] = {
    "func": _read_cnt,
    "params": None,
}
default_module._readers["KIT"] = {
    "func": _read_kit,
    "params": None,
}
default_module._readers["CURRY"] = {
    "func": _read_curry,
    "params": None,
}
default_module._readers["NICOLET"] = {
    "func": _read_nicolet,
    "params": None,
}
default_module._readers["FIELDTRIP"] = {
    "func": _read_fieldtrip,
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
