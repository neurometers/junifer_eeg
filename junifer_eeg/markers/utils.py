"""Utility functions for EEG markers."""

from typing import Any, Dict, List, Optional

import numpy as np


def get_roi_mapping() -> Dict[str, List[str]]:
    """Get mapping of ROI names to electrode names.

    Returns standard 10-20 system ROI definitions.

    Returns
    -------
    dict
        Mapping from ROI names to lists of electrode names.
    """
    roi_mapping = {
        # Frontal
        "Fp": ["Fp1", "Fp2"],
        "F": ["F3", "F4", "F7", "F8"],
        "FC": ["FC1", "FC2", "FC5", "FC6"],
        "Fz": ["Fz"],
        # Central
        "C": ["C3", "C4"],
        "CP": ["CP1", "CP2", "CP5", "CP6"],
        "Cz": ["Cz"],
        # Parietal
        "P": ["P3", "P4", "P7", "P8"],
        "Pz": ["Pz"],
        "PO": ["PO3", "PO4", "PO7", "PO8"],
        # Occipital
        "O": ["O1", "O2"],
        "Oz": ["Oz"],
        # Temporal
        "T": ["T7", "T8"],
        "TP": ["TP9", "TP10"],
        # Individual electrodes (for single electrode ROIs)
        "Fp1": ["Fp1"],
        "Fp2": ["Fp2"],
        "F3": ["F3"],
        "F4": ["F4"],
        "F7": ["F7"],
        "F8": ["F8"],
        "FC1": ["FC1"],
        "FC2": ["FC2"],
        "FC5": ["FC5"],
        "FC6": ["FC6"],
        "C3": ["C3"],
        "C4": ["C4"],
        "CP1": ["CP1"],
        "CP2": ["CP2"],
        "CP5": ["CP5"],
        "CP6": ["CP6"],
        "P3": ["P3"],
        "P4": ["P4"],
        "P7": ["P7"],
        "P8": ["P8"],
        "PO3": ["PO3"],
        "PO4": ["PO4"],
        "PO7": ["PO7"],
        "PO8": ["PO8"],
        "O1": ["O1"],
        "O2": ["O2"],
        "T7": ["T7"],
        "T8": ["T8"],
        "TP9": ["TP9"],
        "TP10": ["TP10"],
    }

    return roi_mapping


def get_data_for_rois(
    data: np.ndarray, ch_names: List[str], rois: List[str]
) -> Dict[str, np.ndarray]:
    """Extract data for specified ROIs.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_channels, ...).
    ch_names : list of str
        Channel names corresponding to first dimension of data.
    rois : list of str
        ROI names to extract.

    Returns
    -------
    dict
        Dictionary mapping ROI names to data arrays.
    """
    roi_mapping = get_roi_mapping()
    roi_data = {}

    for roi in rois:
        if roi not in roi_mapping:
            # If ROI not in mapping, treat as individual electrode
            if roi in ch_names:
                roi_idx = ch_names.index(roi)
                roi_data[roi] = data[roi_idx : roi_idx + 1]  # Keep 2D
            else:
                raise ValueError(
                    f"ROI '{roi}' not found in channel names or ROI mapping"
                )
        else:
            # Get electrodes for this ROI
            roi_electrodes = roi_mapping[roi]
            roi_indices = []

            for electrode in roi_electrodes:
                if electrode in ch_names:
                    roi_indices.append(ch_names.index(electrode))

            if not roi_indices:
                raise ValueError(f"No electrodes found for ROI '{roi}'")

            roi_data[roi] = data[roi_indices]

    return roi_data


def aggregate_data(
    data: np.ndarray, method: str, axis: Optional[int] = None
) -> np.ndarray:
    """Aggregate data using specified method.

    Parameters
    ----------
    data : np.ndarray
        Data to aggregate.
    method : str
        Aggregation method: 'mean', 'std', 'median', 'min', 'max'.
    axis : int, optional
        Axis along which to aggregate. If None, aggregate over all.

    Returns
    -------
    np.ndarray
        Aggregated data.
    """
    if method == "mean":
        return np.mean(data, axis=axis)
    elif method == "std":
        return np.std(data, axis=axis)
    elif method == "median":
        return np.median(data, axis=axis)
    elif method == "min":
        return np.min(data, axis=axis)
    elif method == "max":
        return np.max(data, axis=axis)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def apply_roi_trial_aggregation(
    data: Dict[str, np.ndarray],
    roi_aggregation_methods: Optional[List[str]] = None,
    trial_aggregation_methods: Optional[List[str]] = None,
    marker_name: str = "marker",
) -> Dict[str, Dict[str, Any]]:
    """Apply ROI and trial aggregation to marker data.

    Parameters
    ----------
    data : dict
        Dictionary mapping ROI names to data arrays of shape (n_electrodes, n_trials).
    roi_aggregation_methods : list of str, optional
        Methods to aggregate across ROI electrodes.
    trial_aggregation_methods : list of str, optional
        Methods to aggregate across trials.
    marker_name : str
        Base name for the marker.

    Returns
    -------
    dict
        Dictionary with aggregated results in junifer format.
    """
    results = {}

    # Use a single base output name that junifer recognizes
    base_output_name = (
        marker_name.lower().replace("spectralpower_", "").replace("_", "")
    )

    # Default aggregation if none specified
    if roi_aggregation_methods is None and trial_aggregation_methods is None:
        # Return per-electrode, per-trial data (or averaged if only one trial)
        all_values = []
        col_names = []

        for roi_name, roi_data in data.items():
            if roi_data.ndim == 1:
                # Single trial case
                for i in range(len(roi_data)):
                    all_values.append(roi_data[i])
                    col_names.append(f"{roi_name}_elec_{i}")
            else:
                # Multiple trials - average across trials by default
                trial_avg = np.mean(roi_data, axis=1)
                for i in range(len(trial_avg)):
                    all_values.append(trial_avg[i])
                    col_names.append(f"{roi_name}_elec_{i}")

        results[base_output_name] = {
            "data": np.array(all_values).reshape(1, -1),
            "col_names": col_names,
        }
        return results

    # Handle case with no ROI aggregation but trial aggregation
    if roi_aggregation_methods is None:
        for trial_agg in trial_aggregation_methods:
            all_values = []
            col_names = []

            for roi_name, roi_data in data.items():
                if roi_data.ndim == 1:
                    # Single trial
                    roi_values = roi_data
                else:
                    # Multiple trials
                    roi_values = aggregate_data(roi_data, trial_agg, axis=1)

                # Each electrode in ROI gets its own column
                for i in range(len(roi_values)):
                    col_names.append(f"{roi_name}_trial_{trial_agg}_elec_{i}")
                    all_values.append(roi_values[i])

            results[base_output_name] = {
                "data": np.array(all_values).reshape(1, -1),
                "col_names": col_names,
            }
        return results

    # Handle case with no trial aggregation but ROI aggregation
    if trial_aggregation_methods is None:
        for roi_agg in roi_aggregation_methods:
            all_values = []
            col_names = []

            for roi_name, roi_data in data.items():
                if roi_data.ndim == 1:
                    # Single trial
                    roi_value = aggregate_data(roi_data, roi_agg)
                else:
                    # Multiple trials - average across trials first
                    trial_avg = np.mean(roi_data, axis=1)
                    roi_value = aggregate_data(trial_avg, roi_agg)

                col_names.append(f"{roi_name}_roi_{roi_agg}")
                all_values.append(roi_value)

            results[base_output_name] = {
                "data": np.array(all_values).reshape(1, -1),
                "col_names": col_names,
            }
        return results

    # Full case with both ROI and trial aggregation
    all_values = []
    col_names = []

    for trial_agg in trial_aggregation_methods:
        for roi_agg in roi_aggregation_methods:
            for roi_name, roi_data in data.items():
                if roi_data.ndim == 1:
                    # Single trial
                    roi_value = aggregate_data(roi_data, roi_agg)
                else:
                    # Multiple trials
                    trial_aggregated = aggregate_data(
                        roi_data, trial_agg, axis=1
                    )
                    roi_value = aggregate_data(trial_aggregated, roi_agg)

                col_names.append(f"{roi_name}_trial_{trial_agg}_roi_{roi_agg}")
                all_values.append(roi_value)

    results[base_output_name] = {
        "data": np.array(all_values).reshape(1, -1),
        "col_names": col_names,
    }

    return results
