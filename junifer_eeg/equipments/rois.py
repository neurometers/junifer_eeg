"""ROI registry and lookup functions for EEG equipment.

This module provides a registry system for ROI definitions across different
EEG equipment types, adapted from the NICE pipeline architecture.
"""

from typing import Dict, List

import numpy as np
from junifer.utils import raise_error, warn_with_log

# Standard ROI names that should be defined for each equipment
_STANDARD_ROI_NAMES = [
    "Fz",
    "Cz",
    "Pz",
    "p3a",
    "p3b",
    "mmn",
    "cnv",
    "scalp",
    "nonscalp",
]

# Global registry mapping equipment name to ROI definitions
_rois_map: Dict[str, Dict[str, np.ndarray]] = {}

# Global registry for channel names per equipment
_ch_names_map: Dict[str, List[str]] = {}

# Global registry for channel mapping dictionaries (spatial downsampling)
_ch_mapping_map: Dict[str, Dict[str, str]] = {}


def define_rois(
    equipment: str,
    rois_map: Dict[str, np.ndarray],
    ch_names: List[str] | None = None,
) -> None:
    """Register ROI definitions for an equipment type.

    Parameters
    ----------
    equipment : str
        Equipment identifier (e.g., 'egi/256', 'egi/128', 'biosemi/64').
    rois_map : dict
        Dictionary mapping ROI names to numpy arrays of channel indices.
        Should include all standard ROI names defined in _STANDARD_ROI_NAMES.
    ch_names : list of str, optional
        Canonical channel names for this equipment (e.g., ['A1', 'A2', ...]).
        If provided, enables validation and mapping of channel name strings.

    Raises
    ------
    ValueError
        If not all standard ROIs are defined.

    Warns
    -----
    UserWarning
        If ROIs are already defined for this equipment (will overwrite).
    """
    # Check if all standard ROIs are defined
    missing_rois = [roi for roi in _STANDARD_ROI_NAMES if roi not in rois_map]
    if missing_rois:
        raise_error(
            f"Missing required ROI definitions for {equipment}: {missing_rois}"
        )

    # Warn if overwriting existing definitions
    if equipment in _rois_map:
        warn_with_log(f"ROIs already defined for {equipment}, overwriting")

    _rois_map[equipment] = rois_map

    # Store channel names if provided
    if ch_names is not None:
        _ch_names_map[equipment] = ch_names


def get_roi(equipment: str, roi_name: str) -> np.ndarray:
    """Get channel indices for a specific ROI.

    Parameters
    ----------
    equipment : str
        Equipment identifier (e.g., 'egi/256', 'egi/128').
    roi_name : str
        Name of the ROI (e.g., 'scalp', 'cnv', 'p3a').

    Returns
    -------
    np.ndarray
        Array of channel indices (0-indexed) for the requested ROI.

    Raises
    ------
    ValueError
        If equipment is not registered or ROI name is not defined.
    """
    if equipment not in _rois_map:
        raise_error(f"ROIs not defined for equipment '{equipment}'")

    rois = _rois_map[equipment]
    if roi_name not in rois:
        raise_error(
            f"ROI '{roi_name}' not defined for equipment '{equipment}'. "
            f"Available ROIs: {list(rois.keys())}"
        )

    return rois[roi_name]


def get_roi_ch_names(
    equipment: str, roi_name: str, all_ch_names: List[str]
) -> List[str]:
    """Get channel names for a specific ROI.

    Parameters
    ----------
    equipment : str
        Equipment identifier (e.g., 'egi/256', 'egi/128').
    roi_name : str
        Name of the ROI (e.g., 'scalp', 'cnv', 'p3a').
    all_ch_names : list of str
        Complete list of channel names in the data.

    Returns
    -------
    list of str
        List of channel names belonging to the ROI.

    Raises
    ------
    ValueError
        If equipment is not registered or ROI name is not defined.
    IndexError
        If ROI indices exceed available channels.
    """
    indices = get_roi(equipment, roi_name)

    # Handle empty ROIs (e.g., for averaged montages)
    if indices is None or len(indices) == 0:
        return []

    # Validate indices
    if np.max(indices) >= len(all_ch_names):
        raise_error(
            f"ROI '{roi_name}' contains index {np.max(indices)} but only "
            f"{len(all_ch_names)} channels available"
        )

    return [all_ch_names[i] for i in indices]


def list_available_equipments() -> List[str]:
    """List all registered equipment types.

    Returns
    -------
    list of str
        List of registered equipment identifiers.
    """
    return list(_rois_map.keys())


def list_available_rois(equipment: str) -> List[str]:
    """List all available ROIs for an equipment type.

    Parameters
    ----------
    equipment : str
        Equipment identifier (e.g., 'egi/256', 'egi/128').

    Returns
    -------
    list of str
        List of available ROI names for the equipment.

    Raises
    ------
    ValueError
        If equipment is not registered.
    """
    if equipment not in _rois_map:
        raise_error(f"ROIs not defined for equipment '{equipment}'")

    return list(_rois_map[equipment].keys())


def get_ch_names(equipment: str) -> List[str]:
    """Get canonical channel names for an equipment type.

    Parameters
    ----------
    equipment : str
        Equipment identifier (e.g., 'egi/256', 'biosemi/128').

    Returns
    -------
    list of str
        List of canonical channel names for the equipment.
        Returns None if channel names not defined.

    Examples
    --------
    >>> ch_names = get_ch_names('biosemi/128')
    >>> ch_names[:4]
    ['A1', 'A2', 'A3', 'A4']
    """
    return _ch_names_map.get(equipment, None)
