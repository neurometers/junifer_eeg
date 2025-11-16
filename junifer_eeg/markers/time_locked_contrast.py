"""Time-locked contrast marker for EEG analysis."""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import get_data_for_rois


@register_marker
class TimeLockedContrast(BaseMarker):
    """Time-locked contrast marker for condition comparisons.

    This marker computes contrasts between different experimental conditions
    in specific time windows, following the NICE pattern:

    1. Process condition_a through TimeLockedTopography pipeline
    2. Process condition_b through TimeLockedTopography pipeline
    3. Compute contrast: A - B

    Both conditions use the SAME ROI filtering and aggregation parameters.

    Essential for analyzing event-related potentials (ERPs) and condition effects.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "timelockedcontrast": "vector",
        },
    }

    def __init__(
        self,
        condition_a: str | List[str],
        condition_b: str | List[str],
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
        comment: Optional[str] = None,
        rois: Union[List[str], List[int], None] = None,
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
        equipment: str = "egi256",
        epoch_length: Optional[float] = None,
        overlap: float = 0.0,
        on: Optional[str | List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize TimeLockedContrast marker.

        Parameters
        ----------
        condition_a : str or list of str
            Condition(s) for group A.
        condition_b : str or list of str
            Condition(s) for group B.
        tmin : float, optional
            Start time for analysis window.
        tmax : float, optional
            End time for analysis window.
        baseline : tuple of float, optional
            Baseline correction period (start, end).
        comment : str, optional
            Label for this contrast.
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp', 'mmn_roi', 'p3a_roi')

            Examples:
            - list(range(224)) - NICE scalp ROI
            - ['mmn_roi'] - Task-specific MMN ROI
            - ['p3a_roi'] - Task-specific P3a ROI

            **NOTE:** Use task-specific ROIs for ERP contrasts (mmn, p3a, p3b).
        channel_aggregation_method : str or None, optional
            Methods to aggregate across channels: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        trial_aggregation_method : str or None, optional
            Methods to aggregate across epochs: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        equipment : str, optional
            Equipment type for electrode mapping.
        epoch_length : float, optional
            Length of epochs if creating from continuous data.
        overlap : float, default=0.0
            Overlap between epochs.
        """
        self.condition_a = (
            condition_a if isinstance(condition_a, list) else [condition_a]
        )
        self.condition_b = (
            condition_b if isinstance(condition_b, list) else [condition_b]
        )
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.comment = (
            comment
            or f"{'-'.join(map(str, self.condition_a))}_vs_{'-'.join(map(str, self.condition_b))}"
        )
        self.rois = rois
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        self.epoch_length = epoch_length
        self.overlap = overlap

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute time-locked contrast between conditions.

        Following NICE pattern:
        1. Process condition_a through TimeLockedTopography pipeline
        2. Process condition_b through TimeLockedTopography pipeline
        3. Compute contrast: A - B

        Both conditions use SAME ROI filtering and aggregation parameters.
        """
        from .utils import filter_to_eeg_channels

        epochs = input["data"]

        # Filter to EEG channels
        epochs, eeg_ch_names, eeg_indices = filter_to_eeg_channels(epochs)

        # Apply baseline correction if specified
        if self.baseline is not None:
            epochs = epochs.copy().apply_baseline(self.baseline)

        # Filter epochs by conditions
        # Handle both integer and string condition IDs
        def get_epochs_for_condition(epochs, condition_list):
            """Get epochs for condition(s), handling int/str conversion and multiple conditions."""
            # Convert all conditions to strings that exist in epochs.event_id
            valid_conditions = []
            for cond in condition_list:
                # Try the condition as-is
                if cond in epochs.event_id:
                    valid_conditions.append(cond)
                # Try as string
                elif str(cond) in epochs.event_id:
                    valid_conditions.append(str(cond))
                # Try as int
                else:
                    try:
                        if int(cond) in epochs.event_id:
                            valid_conditions.append(int(cond))
                    except (ValueError, TypeError):
                        pass

            if not valid_conditions:
                return None

            # Return epochs for all valid conditions
            return epochs[valid_conditions]

        epochs_a = get_epochs_for_condition(epochs, self.condition_a)
        epochs_b = get_epochs_for_condition(epochs, self.condition_b)

        # Check for missing conditions
        if (
            epochs_a is None
            or epochs_b is None
            or len(epochs_a) == 0
            or len(epochs_b) == 0
        ):
            # Return empty results
            return {
                "timelockedcontrast": {
                    "data": np.array([[]]),
                    "col_names": [],
                }
            }

        # Store original epochs for raw data return
        epochs_a_full = epochs_a
        epochs_b_full = epochs_b

        # Crop to time window if specified (only for aggregation)
        if self.tmin is not None or self.tmax is not None:
            epochs_a = epochs_a.copy().crop(tmin=self.tmin, tmax=self.tmax)
            epochs_b = epochs_b.copy().crop(tmin=self.tmin, tmax=self.tmax)

        # Helper function to process one condition through TimeLockedTopography pipeline
        def process_condition(epochs_cond):
            """Process one condition: ROI filter → time avg → create roi_data.

            This replicates TimeLockedTopography pipeline for one condition.
            """
            # Get data: (n_epochs, n_channels, n_times)
            data = epochs_cond.get_data()
            ch_names = list(epochs_cond.ch_names)
            n_epochs, n_channels, n_times = data.shape

            # Apply ROI filtering BEFORE time averaging if specified
            if self.rois is not None:
                # Transpose to (n_channels, n_epochs, n_times)
                data_transposed = data.transpose(1, 0, 2)

                # Get ROI-filtered data
                roi_data_dict = get_data_for_rois(
                    data_transposed,
                    ch_names,
                    self.rois,
                    self.equipment,
                )

                # Extract the filtered data (returns {"selected_channels": data})
                if "selected_channels" in roi_data_dict:
                    data_filtered = roi_data_dict["selected_channels"]
                    # Transpose back to (n_epochs, n_channels, n_times)
                    data = data_filtered.transpose(1, 0, 2)
                    n_epochs, n_channels, n_times = data.shape
                    # Update channel names
                    ch_names = [f"ch_{i}" for i in range(n_channels)]

            # Step 1: Average across time dimension (like TimeLockedTopography)
            # Shape: (n_epochs, n_channels)
            time_averaged = np.mean(data, axis=2)

            return time_averaged, ch_names

        # Check if we should return raw temporal data without aggregation
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return raw data from both conditions combined (matching NICE)
            # NICE stores all epochs from both conditions with FULL time range (not cropped)
            # Get raw data for both conditions (use full epochs, not cropped)
            data_a = (
                epochs_a_full.get_data()
            )  # Shape: (n_epochs_a, n_channels, n_times)
            data_b = (
                epochs_b_full.get_data()
            )  # Shape: (n_epochs_b, n_channels, n_times)

            # Concatenate both conditions to match NICE behavior
            all_data = np.concatenate([data_a, data_b], axis=0)
            # Shape: (n_epochs_a + n_epochs_b, n_channels, n_times)

            return {
                "timelockedcontrast": {
                    "data": all_data,  # Keep full temporal dimension (not cropped)
                }
            }

        # For aggregation case, process conditions separately
        from .utils import aggregate_data

        data_a, ch_names_a = process_condition(epochs_a)
        data_b, ch_names_b = process_condition(epochs_b)

        # Helper function to aggregate condition data
        def aggregate_condition(data, ch_names):
            # Apply aggregation
            result_data = data

            # Channel aggregation
            if self.channel_aggregation_method is not None:
                result_data = aggregate_data(
                    result_data, self.channel_aggregation_method, axis=1
                )

            # Trial aggregation
            if self.trial_aggregation_method is not None:
                if result_data.ndim == 1:
                    result_data = aggregate_data(
                        result_data, self.trial_aggregation_method, axis=None
                    )
                else:
                    result_data = aggregate_data(
                        result_data, self.trial_aggregation_method, axis=0
                    )

            # Generate column names based on aggregation and result shape
            if (
                self.channel_aggregation_method is not None
                and self.trial_aggregation_method is not None
            ):
                col_names = ["all_channels_all_trials"]
            elif self.channel_aggregation_method is not None:
                # result_data shape: (n_trials,) after channel aggregation
                n_trials = result_data.shape[0] if result_data.ndim >= 1 else 1
                col_names = [f"trial_{i}" for i in range(n_trials)]
            elif self.trial_aggregation_method is not None:
                col_names = [f"{ch}" for ch in ch_names]
            else:
                col_names = [f"{ch}" for ch in ch_names]

            return result_data, col_names

        # Apply aggregation to both conditions
        agg_data_a, col_names_a = aggregate_condition(data_a, ch_names_a)
        agg_data_b, col_names_b = aggregate_condition(data_b, ch_names_b)

        # Compute contrast: A - B
        contrast_data = agg_data_a - agg_data_b

        return {
            "timelockedcontrast": {
                "data": contrast_data,
                "col_names": col_names_a,
            }
        }
