"""Utility functions for EEG markers."""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from ..equipments import get_roi, list_available_rois


def filter_to_eeg_channels(data_obj):
    """Filter data to only EEG channels (supports both EGI and standard naming).

    Parameters
    ----------
    data_obj : mne.Epochs or mne.Raw
        MNE data object

    Returns
    -------
    filtered_data : mne.Epochs or mne.Raw
        Data with only EEG channels
    eeg_channel_names : list
        Names of EEG channels
    eeg_channel_indices : list
        Indices of EEG channels
    """
    # Try EGI naming convention first (E1, E2, ..., E256)
    egi_channel_indices = [
        i
        for i, ch in enumerate(data_obj.ch_names)
        if ch.startswith("E") and ch[1:].isdigit()
    ]

    if egi_channel_indices:
        # Use EGI channels if found
        eeg_channel_indices = egi_channel_indices
        eeg_channel_names = [data_obj.ch_names[i] for i in eeg_channel_indices]
    else:
        # Fall back to MNE's channel type detection for standard EEG names
        import mne

        eeg_picks = mne.pick_types(data_obj.info, eeg=True, exclude=[])
        eeg_channel_indices = eeg_picks.tolist()
        eeg_channel_names = [data_obj.ch_names[i] for i in eeg_channel_indices]

    # Filter data to only EEG channels
    filtered_data = data_obj.copy().pick(eeg_channel_names)

    return filtered_data, eeg_channel_names, eeg_channel_indices


def get_icm_roi_mapping(equipment: str = "egi256") -> Dict[str, List[str]]:
    """Get ICM-specific ROI mappings for different equipment configurations.

    This function now uses the equipment knowledge base system to retrieve
    ROI definitions. The hardcoded ROI dictionaries have been replaced with
    dynamic lookups from the equipment registry.

    Parameters
    ----------
    equipment : str
        Equipment configuration: 'egi/256', 'egi/128', 'egi/64', 'standard', etc.
        If using old naming ('egi256', 'egi128'), it will be converted automatically.

    Returns
    -------
    dict
        Mapping from ICM ROI names to lists of electrode names.

    Notes
    -----
    This function retrieves ROI definitions from the equipment knowledge base
    located in junifer_eeg/equipments/. All ROI definitions are sourced from
    the NICE pipeline for consistency.

    Examples
    --------
    >>> rois = get_icm_roi_mapping("egi/256")
    >>> scalp_channels = rois["scalp"]  # Returns E1-E224
    >>> cnv_channels = rois["cnv"]  # Returns frontal-central channels
    """
    # Convert old naming convention to new format
    equipment_map = {
        "egi256": "egi/256",
        "egi128": "egi/128",
        "egi64": "egi/64",
        "standard": "egi/64",  # Map standard to 64-channel system
    }
    equipment = equipment_map.get(equipment, equipment)

    # Get all available ROI names for this equipment
    try:
        roi_names = list_available_rois(equipment)
    except ValueError as e:
        # Equipment not found, provide helpful error message
        from ..equipments import list_available_equipments

        available = list_available_equipments()
        raise ValueError(
            f"Equipment '{equipment}' not found in knowledge base. "
            f"Available equipment: {available}"
        ) from e

    # Build ROI mapping dictionary
    # We need channel names, so we'll return indices that can be resolved later
    # For now, return a mapping that can be used by downstream functions
    icm_rois = {}

    # Get indices for each ROI
    for roi_name in roi_names:
        indices = get_roi(equipment, roi_name)
        # Convert indices to channel names (E1-based for EGI, assuming sequential)
        if equipment.startswith("egi/"):
            # EGI naming: E1, E2, E3, ...
            if len(indices) > 0:
                icm_rois[roi_name] = [f"E{i + 1}" for i in indices]
            else:
                icm_rois[roi_name] = []
        else:
            # For other equipment types, return indices directly
            # (will need channel names from actual data)
            icm_rois[roi_name] = indices.tolist() if len(indices) > 0 else []

    return icm_rois


def get_data_for_rois(
    data: np.ndarray,
    ch_names: List[str],
    rois: List[str | int],
    equipment: str = "egi256",
) -> Dict[str, np.ndarray]:
    """Extract data for specified channels/ROIs.

    Resolves a flat list of channel specifications into actual channel indices.
    Each item can be:
    - int: channel index (e.g., 0, 1, 223)
    - str: channel name (e.g., 'E1', 'E224') OR semantic ROI name (e.g., 'frontal', 'scalp')

    All specifications are resolved to a single flat list of channel indices.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_channels, ...).
    ch_names : list of str
        Channel names corresponding to first dimension of data.
    rois : list of str or int
        Flat list of channel specifications. Can mix types:
        - Integers: channel indices (e.g., [0, 1, 2])
        - Strings (channel names): (e.g., ['E1', 'E2', 'E224'])
        - Strings (ROI names): (e.g., ['frontal', 'scalp'])
        Example: [0, 'E5', 'frontal', 10] - all valid, resolved to channel indices
    equipment : str
        Equipment configuration for ICM-specific ROIs: 'standard', 'egi256', or 'egi128'.

    Returns
    -------
    dict
        Dictionary with single key 'selected_channels' mapping to filtered data array.
    """
    if not rois:
        return {}

    # Get ROI mappings for resolving semantic ROI names from equipment knowledge base
    roi_mapping = get_icm_roi_mapping(equipment)

    # Collect all channel indices
    channel_indices = []

    for item in rois:
        if isinstance(item, int):
            # Direct channel index
            if 0 <= item < len(ch_names):
                channel_indices.append(item)
            else:
                raise ValueError(
                    f"Channel index {item} out of range (0-{len(ch_names) - 1})"
                )
        elif isinstance(item, str):
            # Check if it's a semantic ROI name first
            if item in roi_mapping:
                # Semantic ROI - expand to all channels in that ROI
                roi_channels = roi_mapping[item]
                for ch_name in roi_channels:
                    if ch_name in ch_names:
                        channel_indices.append(ch_names.index(ch_name))
            # Check if it's a direct channel name
            elif item in ch_names:
                channel_indices.append(ch_names.index(item))
            else:
                raise ValueError(
                    f"'{item}' not found as channel name or ROI in equipment '{equipment}'"
                )
        else:
            raise TypeError(f"ROI item must be int or str, got {type(item)}")

    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in channel_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # Get channel names for selected indices
    selected_ch_names = [ch_names[idx] for idx in unique_indices]

    # Return single ROI with all selected channels and their names
    return {
        "selected_channels": data[unique_indices],
        "selected_channels_names": selected_ch_names,
    }


def aggregate_data(
    data: np.ndarray,
    method: str,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Aggregate data using specified method.

    Supports NICE-compatible aggregation methods including trimmed means.

    Parameters
    ----------
    data : np.ndarray
        Data to aggregate.
    method : str
        Aggregation method: 'mean', 'std', 'median', 'min', 'max',
        'trim_mean80', 'trim_mean90', or 'sum'.
    axis : int, optional
        Axis along which to aggregate. If None, aggregate over all.

    Returns
    -------
    np.ndarray
        Aggregated data.

    Notes
    -----
    NICE aggregation functions:
    - 'trim_mean80': Removes top/bottom 10% (uses 80% of center data)
    - 'trim_mean90': Removes top/bottom 5% (uses 90% of center data)
    - 'median': Used for connectivity matrix channels_y dimension in WSMI
    """
    if method == "mean":
        return np.mean(data, axis=axis)
    if method == "std":
        return np.std(data, axis=axis)
    if method == "median":
        return np.median(data, axis=axis)
    if method == "min":
        return np.min(data, axis=axis)
    if method == "max":
        return np.max(data, axis=axis)
    if method == "sum":
        return np.sum(data, axis=axis)
    if method == "trim_mean80":
        # NICE-compatible trimmed mean: remove top/bottom 10% (use 80% of data)
        return stats.trim_mean(data, proportiontocut=0.1, axis=axis)
    if method == "trim_mean90":
        # NICE-compatible trimmed mean: remove top/bottom 5% (use 90% of data)
        return stats.trim_mean(data, proportiontocut=0.05, axis=axis)
    raise ValueError(f"Unknown aggregation method: {method}")


def apply_aggregation_preserve_dims(
    data: np.ndarray,
    channel_method: Optional[str] = None,
    trial_method: Optional[str] = None,
    connectivity_method: Optional[str] = None,
    aggregation_order: str = "channel_trial",
) -> np.ndarray:
    """Apply aggregation while preserving tensor dimensions.

    This unified helper reduces code duplication across EEG markers.
    After aggregation, dimensions are preserved using expand_dims,
    resulting in size-1 dimensions where aggregation was applied.

    Handles data shapes:
    - 2D: (n_epochs, n_channels)
    - 3D: (n_bands/n_taus, n_epochs, n_channels)
    - 4D: (n_taus, n_epochs, n_channels_x, n_channels_y) for connectivity

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    channel_method : str or None
        Method to aggregate across channels (last axis for 2D/3D, axis 2 for 4D).
        Options: 'mean', 'std', 'median', 'trim_mean80', 'trim_mean90', etc.
    trial_method : str or None
        Method to aggregate across trials/epochs (axis 0 for 2D, axis 1 for 3D/4D).
        Options: 'mean', 'std', 'median', 'trim_mean80', 'trim_mean90', etc.
    connectivity_method : str or None
        Method to aggregate across connectivity dimension (axis 3 for 4D data).
        Only applicable to 4D connectivity data.
        Options: 'mean', 'median', etc.
    aggregation_order : str, default='channel_trial'
        Order of aggregation operations. Options:
        - 'channel_trial': Apply channel aggregation first, then trial (default)
        - 'trial_channel': Apply trial aggregation first, then channel
        Note: Order matters for methods like trim_mean80 that remove outliers.

    Returns
    -------
    np.ndarray
        Aggregated data with preserved dimensions. Aggregated axes have size 1.

    Examples
    --------
    >>> # 2D: (100 epochs, 64 channels) -> (100, 1) after channel aggregation
    >>> data_2d = np.random.randn(100, 64)
    >>> result = apply_aggregation_preserve_dims(data_2d, channel_method='mean')
    >>> result.shape
    (100, 1)

    >>> # 3D: (5 bands, 100 epochs, 64 channels) -> (5, 1, 1) after both
    >>> data_3d = np.random.randn(5, 100, 64)
    >>> result = apply_aggregation_preserve_dims(
    ...     data_3d, channel_method='mean', trial_method='mean'
    ... )
    >>> result.shape
    (5, 1, 1)

    Notes
    -----
    This helper implements the common aggregation pattern used by:
    - KolmogorovComplexity
    - PermutationEntropy
    - SpectralPowerBands
    - TimeLockedTopography
    - TimeLockedContrast
    - ContingentNegativeVariation
    - PowerSpectralDensitySummary
    - SlowWavesDetection
    - SpindlesDetection
    - SymbolicMutualInformationROIs
    """
    result = data.copy()
    ndim = result.ndim

    if ndim == 2:
        # Shape: (n_epochs, n_channels)
        # Channel aggregation: axis=1
        if channel_method is not None:
            result = aggregate_data(result, channel_method, axis=1)
            result = np.expand_dims(
                result, axis=1
            )  # (n_epochs,) -> (n_epochs, 1)

        # Trial aggregation: axis=0
        if trial_method is not None:
            result = aggregate_data(result, trial_method, axis=0)
            result = np.expand_dims(result, axis=0)  # -> (1, ...)

    elif ndim == 3:
        # Shape: (n_bands/n_taus, n_epochs, n_channels)
        if aggregation_order == "trial_channel":
            # Trial aggregation first (axis=1)
            if trial_method is not None:
                result = aggregate_data(result, trial_method, axis=1)
                result = np.expand_dims(result, axis=1)  # -> (n_bands, 1, ...)

            # Channel aggregation second (axis=2)
            if channel_method is not None:
                result = aggregate_data(result, channel_method, axis=2)
                result = np.expand_dims(result, axis=2)  # -> (..., 1)
        else:
            # Channel aggregation first (axis=2) - default
            if channel_method is not None:
                result = aggregate_data(result, channel_method, axis=2)
                result = np.expand_dims(result, axis=2)  # -> (..., 1)

            # Trial aggregation second (axis=1)
            if trial_method is not None:
                result = aggregate_data(result, trial_method, axis=1)
                result = np.expand_dims(result, axis=1)  # -> (n_bands, 1, ...)

    elif ndim == 4:
        # Shape: (n_taus, n_epochs, n_channels_x, n_channels_y)
        # Connectivity aggregation: axis=3 (across channels_y)
        if connectivity_method is not None:
            result = aggregate_data(result, connectivity_method, axis=3)
            result = np.expand_dims(result, axis=3)  # -> (..., 1)

        # Channel aggregation: axis=2 (across channels_x)
        if channel_method is not None:
            result = aggregate_data(result, channel_method, axis=2)
            result = np.expand_dims(
                result, axis=2
            )  # -> (n_taus, n_epochs, 1, ...)

        # Trial aggregation: axis=1
        if trial_method is not None:
            result = aggregate_data(result, trial_method, axis=1)
            result = np.expand_dims(result, axis=1)  # -> (n_taus, 1, ...)

    return result


def apply_roi_filtering_to_epochs(
    epochs,
    rois: List[str | int],
    equipment: str = "egi256",
    marker_name: str = "Marker",
):
    """Apply ROI filtering to an MNE Epochs object.

    This is a common utility to reduce code duplication across markers that
    need to filter epochs by ROI before computation.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs object
    rois : list of str or int
        ROI specification for channel filtering
    equipment : str, default="egi256"
        Equipment type for electrode mapping
    marker_name : str, default="Marker"
        Name of the calling marker for error messages

    Returns
    -------
    epochs_filtered : mne.Epochs
        New Epochs object with only the ROI-filtered channels
    ch_names_filtered : list of str
        Updated channel names after filtering

    Notes
    -----
    This helper implements the common transpose/filter/transpose-back pattern
    used by SpectralPowerBands, PermutationEntropy, SymbolicMutualInformation,
    and other markers that need ROI filtering before computation.
    """
    import mne

    ch_names = list(epochs.ch_names)
    sfreq = epochs.info["sfreq"]

    # Get equipment from data metadata or fallback
    description = epochs.info.get("description") or ""
    if "equipment=" in description:
        equipment = description.replace("equipment=", "")

    # Get data as (n_epochs, n_channels, n_times)
    data_array = epochs.get_data()

    # Transpose to (n_channels, n_epochs, n_times) for ROI filtering
    data_transposed = data_array.transpose(1, 0, 2)

    # Get ROI-filtered data
    roi_data_dict = get_data_for_rois(
        data_transposed,
        ch_names,
        rois,
        equipment,
    )

    # Extract the filtered data
    if "selected_channels" not in roi_data_dict:
        raise ValueError(f"ROI filtering failed for rois: {rois}")

    data_filtered = roi_data_dict["selected_channels"]
    # Transpose back to (n_epochs, n_channels, n_times)
    data_filtered = data_filtered.transpose(1, 0, 2)

    # Validate that ROI count matches filtered data shape
    n_channels_filtered = data_filtered.shape[1]
    if len(rois) != n_channels_filtered:
        raise ValueError(
            f"ROI count mismatch: {len(rois)} != {n_channels_filtered} in {marker_name}"
        )

    # Create minimal info for filtered channels
    # Use actual ROI channel names to preserve channel identity
    info = mne.create_info(
        ch_names=list(rois)
        if not all(isinstance(r, str) for r in rois)
        else rois,
        sfreq=sfreq,
        ch_types="eeg",
    )

    # Create new Epochs object with filtered channels
    epochs_filtered = mne.EpochsArray(
        data_filtered,
        info,
        events=epochs.events,
        tmin=epochs.tmin,
        verbose=False,
    )

    return epochs_filtered, list(epochs_filtered.ch_names)


def create_connectivity_pair_column_names(
    channel_names: List[str],
) -> list[str]:
    """Create connectivity pair column names for upper triangular matrix.

    Generates column names in the format: channel1-channel2
    Used for connectivity markers (SMI, coherence, PLV, etc.).

    Used by: SymbolicMutualInformation, and future connectivity markers

    Parameters
    ----------
    channel_names : list of str
        Names of EEG channels.

    Returns
    -------
    col_names : list of str
        Column names in format: channel1-channel2 for upper triangular pairs.

    Examples
    --------
    >>> channels = ['Fp1', 'Fz', 'Cz']
    >>> names = create_connectivity_pair_column_names(channels)
    >>> names
    ['Fp1-Fz', 'Fp1-Cz', 'Fz-Cz']
    """
    col_names = []
    n_channels = len(channel_names)

    # Generate column names for upper triangular pairs only (excluding diagonal)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):  # Only upper triangular (j > i)
            col_names.append(f"{channel_names[i]}-{channel_names[j]}")

    return col_names
