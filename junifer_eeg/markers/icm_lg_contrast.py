"""ICM Local-Global Contrast Analysis Marker for EEG data.

This marker computes contrast measures specific to the ICM Local-Global
paradigm, including local vs global processing differences and hierarchical
processing effects.
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class ICMLGContrast(BaseMarker):
    """ICM Local-Global Contrast Analysis Marker.

    This marker computes contrast measures specific to the ICM LG paradigm:
    - Local vs Global processing contrasts
    - Standard vs Deviant contrasts within each level
    - Hierarchical processing difference waves
    - Time window-specific contrasts

    Parameters
    ----------
    time_windows : dict, optional
        Time windows for contrast analysis.
        Default: {'local': (0.08, 0.20), 'global': (0.20, 0.40)}
    baseline : tuple, optional
        Baseline period for contrast computation (tmin, tmax) in seconds.
        Default: (-0.2, 0.0)
    electrode_groups : dict, optional
        Electrode groups for region-specific analysis.
        Default: {'frontal': ['Fz', 'F3', 'F4'], 'central': ['Cz', 'C3', 'C4']}
    rois : list of str, optional
        List of ROI names for aggregation.
    roi_aggregation_method : list of str, optional
        Methods to aggregate across ROI electrodes: ['mean', 'std'].
    trial_aggregation_method : list of str, optional
        Methods to aggregate across trials/epochs: ['mean', 'std'].
    equipment : str, optional
        Equipment type for electrode mapping. Default: 'standard'
    on : str or list of str, optional
        The data type to apply the marker to. Default: 'EEG'
    name : str, optional
        The name of the marker. Default: None
    """

    _DEPENDENCIES: ClassVar[set[str]] = {"numpy", "mne"}
    _MARKER_INOUT_MAPPINGS: ClassVar[dict[str, dict[str, str]]] = {
        "EEG": {"local_global_contrast": "vector"},
    }

    def __init__(
        self,
        time_windows: Optional[Dict[str, Tuple[float, float]]] = None,
        baseline: Optional[Tuple[float, float]] = (-0.2, 0.0),
        electrode_groups: Optional[Dict[str, List[str]]] = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        equipment: str = "standard",
        on: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the ICMLGContrast marker."""
        self.time_windows = time_windows or {
            "local": (0.08, 0.20),
            "global": (0.20, 0.40),
        }
        self.baseline = baseline
        self.electrode_groups = electrode_groups or {
            "frontal": ["Fz", "F3", "F4"],
            "central": ["Cz", "C3", "C4"],
        }
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        super().__init__(on=on, name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute ICM LG contrast measures.

        Parameters
        ----------
        input : dict
            The input data dictionary containing EEG epochs data
        extra_input : dict, optional
            Additional input data (not used)

        Returns
        -------
        dict
            Dictionary containing computed contrast measures
        """
        # Get epochs data
        epochs = input["data"]

        # Get event information
        events = epochs.events
        event_id = epochs.event_id

        # ICM LG event IDs
        icm_events = {
            "HSTD": 10,
            "HDVT": 20,
            "LSGS": 30,
            "LSGD": 40,
            "LDGS": 60,
            "LDGD": 50,
        }

        # Get data matrix (n_epochs, n_channels, n_times)
        data = epochs.get_data()

        # Get time vector
        times = epochs.times

        # Apply baseline correction if specified
        if self.baseline is not None:
            baseline_mask = (times >= self.baseline[0]) & (
                times <= self.baseline[1]
            )
            baseline_data = data[:, :, baseline_mask].mean(
                axis=2,
                keepdims=True,
            )
            data = data - baseline_data

        # Group epochs by condition
        condition_data = {}
        for condition, event_code in icm_events.items():
            if event_code in event_id.values():
                condition_mask = events[:, 2] == event_code
                if condition_mask.any():
                    condition_data[condition] = data[condition_mask]

        # Compute contrast values for different time windows
        contrast_values = {}

        # Local vs Global processing contrast
        local_conditions = []
        global_conditions = []

        # Collect local and global conditions
        for condition, cond_data in condition_data.items():
            if condition.startswith("LS"):  # Local Standard/Deviant
                local_conditions.append(cond_data.mean(axis=0))
            elif condition.startswith(
                "LD"
            ):  # Global conditions (Local Deviant)
                global_conditions.append(cond_data.mean(axis=0))

        # Compute contrasts for each time window
        for window_name, (tmin, tmax) in self.time_windows.items():
            window_mask = (times >= tmin) & (times <= tmax)
            if window_mask.any():
                if local_conditions and global_conditions:
                    # Average across conditions
                    local_avg = np.mean(local_conditions, axis=0)
                    global_avg = np.mean(global_conditions, axis=0)

                    # Compute contrast
                    contrast = global_avg - local_avg

                    # Extract time window mean
                    if contrast.ndim == 2:  # (n_channels, n_times)
                        window_contrast = contrast[:, window_mask].mean(axis=1)
                    else:  # 1D array
                        window_contrast = contrast
                    contrast_values[f"{window_name}_contrast"] = (
                        window_contrast
                    )

        # Convert to format suitable for aggregation
        n_channels = len(epochs.ch_names)
        contrast_data = np.zeros(
            (1, n_channels)
        )  # Single "trial" with contrast per channel

        if contrast_values:
            # Use the first contrast as the main result
            main_contrast = next(iter(contrast_values.values()))
            contrast_data[0, :] = main_contrast

        # Apply ROI selection and aggregation
        if self.rois is not None:
            roi_data = get_data_for_rois(
                contrast_data.T,  # Transpose to (n_channels, 1)
                list(epochs.ch_names),
                self.rois,
                equipment=self.equipment,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: contrast_data[:, i : i + 1].T
                for i, ch in enumerate(epochs.ch_names)
            }

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="icmlgcontrast",
        )

        return results
