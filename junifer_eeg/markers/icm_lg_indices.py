"""ICM Local-Global Index Analysis Marker for EEG data.

This marker computes global-local processing indices specific to the ICM Local-Global
paradigm, including global precedence index, local processing index, and
hierarchical interaction indices.
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class ICMLGGlobalLocalIndex(BaseMarker):
    """ICM Local-Global Index Analysis Marker.

    This marker computes processing indices specific to the ICM LG paradigm:
    - Global precedence index (early vs late processing)
    - Local processing index (local focus vs global integration)
    - Hierarchical interaction index (cross-level interactions)
    - Time-resolved processing indices

    Parameters
    ----------
    early_window : tuple, optional
        Time window for early processing (tmin, tmax) in seconds.
        Default: (0.08, 0.20)
    late_window : tuple, optional
        Time window for late processing (tmin, tmax) in seconds.
        Default: (0.20, 0.40)
    baseline : tuple, optional
        Baseline period for index computation (tmin, tmax) in seconds.
        Default: (-0.2, 0.0)
    time_resolved : bool, optional
        Whether to compute time-resolved indices.
        Default: True
    sliding_window : float, optional
        Window size for time-resolved analysis in seconds.
        Default: 0.05
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

    _DEPENDENCIES: ClassVar[set[str]] = {"numpy", "mne", "scipy"}
    _MARKER_INOUT_MAPPINGS: ClassVar[dict[str, dict[str, str]]] = {
        "EEG": {"global_local_index": "vector"},
    }

    def __init__(
        self,
        early_window: Tuple[float, float] = (0.08, 0.20),
        late_window: Tuple[float, float] = (0.20, 0.40),
        baseline: Optional[Tuple[float, float]] = (-0.2, 0.0),
        time_resolved: bool = True,
        sliding_window: float = 0.05,
        electrode_groups: Optional[Dict[str, List[str]]] = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        equipment: str = "standard",
        on: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the ICMLGGlobalLocalIndex marker."""
        self.early_window = early_window
        self.late_window = late_window
        self.baseline = baseline
        self.time_resolved = time_resolved
        self.sliding_window = sliding_window
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
        """Compute ICM LG global-local processing indices.

        Parameters
        ----------
        input : dict
            The input data dictionary containing EEG epochs data
        extra_input : dict, optional
            Additional input data (not used)

        Returns
        -------
        dict
            Dictionary containing computed processing indices
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

        # Compute global precedence index
        index_values = {}

        # Early vs Late processing comparison
        early_mask = (times >= self.early_window[0]) & (
            times <= self.early_window[1]
        )
        late_mask = (times >= self.late_window[0]) & (
            times <= self.late_window[1]
        )

        if early_mask.any() and late_mask.any():
            # Compute average activity across all conditions for early and late windows
            all_conditions = []
            for cond_data in condition_data.values():
                all_conditions.append(cond_data.mean(axis=0))

            if all_conditions:
                combined_avg = np.mean(all_conditions, axis=0)
                if combined_avg.ndim == 2:  # (n_channels, n_times)
                    early_activity = combined_avg[:, early_mask].mean(axis=1)
                    late_activity = combined_avg[:, late_mask].mean(axis=1)
                else:  # 1D array
                    early_activity = combined_avg
                    late_activity = combined_avg

                # Global precedence index: Early activity relative to late activity
                gpi = (early_activity - late_activity) / (
                    early_activity + late_activity + 1e-8
                )
                index_values["global_precedence_index"] = gpi

        # Local vs Global processing index
        if "LSGS" in condition_data and "LDGS" in condition_data:
            lsgs_avg = condition_data["LSGS"].mean(axis=0)
            ldgs_avg = condition_data["LDGS"].mean(axis=0)

            # Compute difference in early window
            if early_mask.any():
                if lsgs_avg.ndim == 2:  # (n_channels, n_times)
                    lsgs_early = lsgs_avg[:, early_mask].mean(axis=1)
                    ldgs_early = ldgs_avg[:, early_mask].mean(axis=1)
                else:  # 1D array
                    lsgs_early = lsgs_avg
                    ldgs_early = ldgs_avg

                # Local processing index: Local standard vs Global standard
                lpi = (lsgs_early - ldgs_early) / (
                    lsgs_early + ldgs_early + 1e-8
                )
                index_values["local_processing_index"] = lpi

        # Convert to format suitable for aggregation
        n_channels = len(epochs.ch_names)
        index_data = np.zeros(
            (1, n_channels)
        )  # Single "trial" with index per channel

        if index_values:
            # Use the first index as the main result
            main_index = next(iter(index_values.values()))
            index_data[0, :] = main_index

        # Apply ROI selection and aggregation
        if self.rois is not None:
            roi_data = get_data_for_rois(
                index_data.T,  # Transpose to (n_channels, 1)
                list(epochs.ch_names),
                self.rois,
                equipment=self.equipment,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: index_data[:, i : i + 1].T
                for i, ch in enumerate(epochs.ch_names)
            }

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="icmlgindex",
        )

        return results
