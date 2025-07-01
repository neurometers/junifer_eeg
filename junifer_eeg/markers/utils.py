"""Utility functions for EEG markers."""

from typing import Any, Dict, List, Optional

import numpy as np


def get_roi_mapping() -> Dict[str, List[str]]:
    """Get mapping of ROI names to electrode names.

    Returns standard 10-20 system ROI definitions and ICM-specific ROIs.

    Returns
    -------
    dict
        Mapping from ROI names to lists of electrode names.
    """
    roi_mapping = {
        # Standard 10-20 system ROIs
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


def get_icm_roi_mapping(equipment: str = "standard") -> Dict[str, List[str]]:
    """Get ICM-specific ROI mappings for different equipment configurations.

    Parameters
    ----------
    equipment : str
        Equipment configuration: 'standard', 'egi256', or 'egi128'.

    Returns
    -------
    dict
        Mapping from ICM ROI names to lists of electrode names.
    """

    if equipment == "egi256":
        # EGI 256-channel system ROI definitions for ICM Local Global paradigm
        icm_rois = {
            # Scalp ROI - all EEG channels excluding bad/reference channels
            "scalp": [
                f"E{i}"
                for i in range(1, 257)
                if i
                not in [
                    17,
                    128,
                    126,
                    127,
                    132,
                    133,
                    134,
                    135,
                    136,
                    137,
                    138,
                    139,
                    140,
                    141,
                    142,
                    143,
                    144,
                    145,
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                    159,
                    160,
                    161,
                    162,
                    163,
                    164,
                    165,
                    166,
                    167,
                    168,
                    169,
                    170,
                    171,
                    172,
                    173,
                    174,
                    175,
                    176,
                    177,
                    178,
                    179,
                    180,
                    181,
                    182,
                    183,
                    184,
                    185,
                    186,
                    187,
                    188,
                    189,
                    190,
                    191,
                    192,
                    193,
                    194,
                    195,
                    196,
                    197,
                    198,
                    199,
                    200,
                    201,
                    202,
                    203,
                    204,
                    205,
                    206,
                    207,
                    208,
                    209,
                    210,
                    211,
                    212,
                    213,
                    214,
                    215,
                    216,
                    217,
                    218,
                    219,
                    220,
                    221,
                    222,
                    223,
                    224,
                    225,
                    226,
                    227,
                    228,
                    229,
                    230,
                    231,
                    232,
                    233,
                    234,
                    235,
                    236,
                    237,
                    238,
                    239,
                    240,
                    241,
                    242,
                    243,
                    244,
                    245,
                    246,
                    247,
                    248,
                    249,
                    250,
                    251,
                    252,
                    253,
                    254,
                    255,
                    256,
                ]
            ],
            # CNV ROI - central and frontal-central electrodes for CNV analysis
            "cnv": [
                "E5",
                "E6",
                "E11",
                "E12",
                "E13",
                "E16",
                "E18",
                "E19",
                "E20",
                "E23",
                "E24",
                "E25",
                "E26",
                "E27",
                "E28",
                "E29",
                "E30",
                "E31",
                "E35",
                "E36",
                "E37",
                "E40",
                "E41",
                "E42",
                "E103",
                "E104",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
                "E115",
                "E116",
                "E117",
                "E118",
            ],
            # MMN ROI - fronto-central electrodes for mismatch negativity
            "mmn": [
                "E5",
                "E6",
                "E7",
                "E11",
                "E12",
                "E13",
                "E16",
                "E18",
                "E19",
                "E20",
                "E23",
                "E24",
                "E25",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
                "E115",
                "E116",
                "E117",
                "E118",
            ],
            # P3a ROI - fronto-central electrodes for P3a component
            "p3a": [
                "E5",
                "E6",
                "E11",
                "E12",
                "E13",
                "E16",
                "E18",
                "E19",
                "E20",
                "E23",
                "E24",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
                "E115",
                "E116",
            ],
            # P3b ROI - centro-parietal electrodes for P3b component
            "p3b": [
                "E7",
                "E31",
                "E37",
                "E40",
                "E41",
                "E42",
                "E47",
                "E53",
                "E54",
                "E55",
                "E60",
                "E61",
                "E62",
                "E67",
                "E72",
                "E77",
                "E78",
                "E79",
                "E80",
                "E85",
                "E86",
                "E87",
                "E106",
            ],
        }

    elif equipment == "egi128":
        # EGI 128-channel system ROI definitions
        icm_rois = {
            # Scalp ROI - all EEG channels for 128-channel system
            "scalp": [
                f"E{i}"
                for i in range(1, 129)
                if i not in [17, 125, 126, 127, 128]
            ],
            # CNV ROI - central and frontal-central electrodes
            "cnv": [
                "E3",
                "E4",
                "E5",
                "E6",
                "E9",
                "E10",
                "E11",
                "E12",
                "E13",
                "E15",
                "E16",
                "E18",
                "E19",
                "E20",
                "E22",
                "E23",
                "E24",
                "E103",
                "E104",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
            ],
            # MMN ROI - fronto-central electrodes
            "mmn": [
                "E3",
                "E4",
                "E5",
                "E6",
                "E9",
                "E10",
                "E11",
                "E12",
                "E13",
                "E15",
                "E16",
                "E103",
                "E104",
                "E105",
                "E106",
                "E109",
                "E110",
                "E111",
                "E112",
            ],
            # P3a ROI - fronto-central electrodes
            "p3a": [
                "E3",
                "E4",
                "E5",
                "E6",
                "E9",
                "E10",
                "E11",
                "E12",
                "E13",
                "E103",
                "E104",
                "E105",
                "E106",
            ],
            # P3b ROI - centro-parietal electrodes
            "p3b": [
                "E7",
                "E31",
                "E37",
                "E40",
                "E41",
                "E42",
                "E47",
                "E53",
                "E54",
                "E55",
                "E60",
                "E61",
                "E62",
                "E72",
                "E77",
                "E78",
                "E79",
                "E106",
            ],
        }

    else:
        # Standard 10-20 system ICM ROI approximations
        icm_rois = {
            # Scalp ROI - all standard electrodes
            "scalp": [
                "Fp1",
                "Fp2",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "TP9",
                "CP5",
                "CP1",
                "CP2",
                "CP6",
                "TP10",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "PO9",
                "O1",
                "Oz",
                "O2",
                "PO10",
            ],
            # CNV ROI - central and frontal-central region
            "cnv": ["Fz", "FC1", "FC2", "Cz", "C3", "C4"],
            # MMN ROI - fronto-central region for mismatch negativity
            "mmn": ["Fz", "F3", "F4", "FC1", "FC2", "FC5", "FC6"],
            # P3a ROI - fronto-central region
            "p3a": ["Fz", "F3", "F4", "FC1", "FC2", "Cz"],
            # P3b ROI - centro-parietal region
            "p3b": ["Cz", "CP1", "CP2", "Pz", "P3", "P4"],
        }

    return icm_rois


def get_data_for_rois(
    data: np.ndarray,
    ch_names: List[str],
    rois: List[str],
    equipment: str = "standard",
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
    equipment : str
        Equipment configuration for ICM-specific ROIs: 'standard', 'egi256', or 'egi128'.

    Returns
    -------
    dict
        Dictionary mapping ROI names to data arrays.
    """
    # Get both standard and ICM-specific ROI mappings
    standard_roi_mapping = get_roi_mapping()
    icm_roi_mapping = get_icm_roi_mapping(equipment)

    # Combine mappings (ICM ROIs take precedence)
    roi_mapping = {**standard_roi_mapping, **icm_roi_mapping}

    roi_data = {}

    for roi in rois:
        if roi not in roi_mapping:
            # If ROI not in mapping, treat as individual electrode
            if roi in ch_names:
                roi_idx = ch_names.index(roi)
                roi_data[roi] = data[roi_idx : roi_idx + 1]  # Keep 2D
            else:
                raise ValueError(
                    f"ROI '{roi}' not found in channel names or ROI mapping for equipment '{equipment}'"
                )
        else:
            # Get electrodes for this ROI
            roi_electrodes = roi_mapping[roi]
            roi_indices = []

            for electrode in roi_electrodes:
                if electrode in ch_names:
                    roi_indices.append(ch_names.index(electrode))

            if not roi_indices:
                raise ValueError(
                    f"No electrodes found for ROI '{roi}' in equipment '{equipment}'"
                )

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
