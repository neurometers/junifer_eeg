"""ICM Local-Global Mismatch Negativity Analysis Marker for EEG data.

This marker computes MMN-related measures specific to the ICM Local-Global
paradigm, including hierarchical MMN effects and N1 component analysis.
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class ICMLGMismatchNegativity(BaseMarker):
    """ICM Local-Global Mismatch Negativity Analysis Marker.

    This marker computes MMN measures specific to the ICM LG paradigm:
    - Local MMN (LSGD vs LSGS)
    - Global MMN (LDGD vs LDGS)
    - Hierarchical MMN effects
    - N1 component analysis

    Parameters
    ----------
    mmn_window : tuple, optional
        Time window for MMN analysis (tmin, tmax) in seconds.
        Default: (0.15, 0.25)
    n1_window : tuple, optional
        Time window for N1 analysis (tmin, tmax) in seconds.
        Default: (0.08, 0.14)
    baseline : tuple, optional
        Baseline period for MMN computation (tmin, tmax) in seconds.
        Default: (-0.2, 0.0)
    peak_detection : bool, optional
        Whether to perform peak detection for MMN amplitude.
        Default: True
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
        "EEG": {"mmn_amplitude": "vector"},
    }

    def __init__(
        self,
        mmn_window: Tuple[float, float] = (0.15, 0.25),
        n1_window: Tuple[float, float] = (0.08, 0.14),
        baseline: Optional[Tuple[float, float]] = (-0.2, 0.0),
        peak_detection: bool = True,
        electrode_groups: Optional[Dict[str, List[str]]] = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        equipment: str = "standard",
        on: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the ICMLGMismatchNegativity marker."""
        self.mmn_window = mmn_window
        self.n1_window = n1_window
        self.baseline = baseline
        self.peak_detection = peak_detection
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
        """Compute ICM LG MMN measures.

        Parameters
        ----------
        input : dict
            The input data dictionary containing EEG epochs data
        extra_input : dict, optional
            Additional input data (not used)

        Returns
        -------
        dict
            Dictionary containing computed MMN measures
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

        # Compute MMN measures
        mmn_values = {}

        # Local MMN: LSGD vs LSGS
        if "LSGD" in condition_data and "LSGS" in condition_data:
            lsgd_avg = condition_data["LSGD"].mean(axis=0)
            lsgs_avg = condition_data["LSGS"].mean(axis=0)

            mmn_mask = (times >= self.mmn_window[0]) & (
                times <= self.mmn_window[1]
            )
            if mmn_mask.any():
                mmn_diff = lsgd_avg - lsgs_avg
                if mmn_diff.ndim == 2:  # (n_channels, n_times)
                    local_mmn = mmn_diff[:, mmn_mask].mean(axis=1)
                else:  # 1D array
                    local_mmn = mmn_diff
                mmn_values["local_mmn"] = local_mmn

        # Global MMN: LDGD vs LDGS
        if "LDGD" in condition_data and "LDGS" in condition_data:
            ldgd_avg = condition_data["LDGD"].mean(axis=0)
            ldgs_avg = condition_data["LDGS"].mean(axis=0)

            mmn_mask = (times >= self.mmn_window[0]) & (
                times <= self.mmn_window[1]
            )
            if mmn_mask.any():
                mmn_diff = ldgd_avg - ldgs_avg
                if mmn_diff.ndim == 2:  # (n_channels, n_times)
                    global_mmn = mmn_diff[:, mmn_mask].mean(axis=1)
                else:  # 1D array
                    global_mmn = mmn_diff
                mmn_values["global_mmn"] = global_mmn

        # N1 component analysis
        n1_mask = (times >= self.n1_window[0]) & (times <= self.n1_window[1])
        if n1_mask.any():
            for condition, cond_data in condition_data.items():
                condition_avg = cond_data.mean(axis=0)
                if condition_avg.ndim == 2:  # (n_channels, n_times)
                    n1_amplitude = condition_avg[:, n1_mask].mean(axis=1)
                else:  # 1D array
                    n1_amplitude = condition_avg
                mmn_values[f"n1_{condition.lower()}"] = n1_amplitude

        # Convert to format suitable for aggregation
        n_channels = len(epochs.ch_names)
        mmn_data = np.zeros(
            (1, n_channels)
        )  # Single "trial" with MMN per channel

        if mmn_values:
            # Use the first MMN value as the main result
            main_mmn = next(iter(mmn_values.values()))
            mmn_data[0, :] = main_mmn

        # Apply ROI selection and aggregation
        if self.rois is not None:
            roi_data = get_data_for_rois(
                mmn_data.T,  # Transpose to (n_channels, 1)
                list(epochs.ch_names),
                self.rois,
                equipment=self.equipment,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: mmn_data[:, i : i + 1].T
                for i, ch in enumerate(epochs.ch_names)
            }

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="icmlgmmn",
        )

        return results
