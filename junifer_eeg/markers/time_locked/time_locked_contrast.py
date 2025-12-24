"""Time-locked contrast marker for EEG analysis - refactored version.

This marker computes contrasts between different experimental conditions
in specific time windows, following the NICE pattern:

1. Process condition_a through TimeLockedTopography pipeline
2. Process condition_b through TimeLockedTopography pipeline
3. Compute contrast: A - B

Both conditions use the SAME ROI filtering and aggregation parameters.

Refactored to use TimeLockedBase for common data preparation operations.
"""

from typing import Any, ClassVar, Dict, List, Optional, Union

from junifer.api.decorators import register_marker

from ..base import format_marker_result
from ..utils import apply_aggregation_preserve_dims
from ._time_locked_base import TimeLockedBase


@register_marker
class TimeLockedContrast(TimeLockedBase):
    """Time-locked contrast marker for condition comparisons.

    This marker computes contrasts between different experimental conditions
    in specific time windows, following the NICE pattern:

    1. Process condition_a through TimeLockedTopography pipeline
    2. Process condition_b through TimeLockedTopography pipeline
    3. Compute contrast: A - B

    Both conditions use the SAME ROI filtering and aggregation parameters.

    Essential for analyzing event-related potentials (ERPs) and condition effects.

    Refactored to use shared base class helpers for ROI filtering and data preparation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "timelockedcontrast": "timeseries",  # 3D: (epochs, channels, times) when not aggregated
        },
    }

    def __init__(
        self,
        condition_a: str | int | List[str | int],
        condition_b: str | int | List[str | int],
        tmin: float,
        tmax: float,
        baseline: tuple[float, float] | None = None,
        reference: str | list[str] | None = None,
        comment: str | None = None,
        rois: Union[List[str], List[int], None] = None,
        channel_method: str | None = None,
        trial_method: str | None = None,
        equipment: str = "egi256",
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the TimeLockedContrast marker.

        Parameters
        ----------
        condition_a : str, int, or list
            First condition(s) for contrast analysis.
        condition_b : str, int, or list
            Second condition(s) for contrast analysis.
        tmin : float
            Start time for analysis in seconds.
        tmax : float
            End time for analysis in seconds.
        baseline : tuple of float, optional
            Baseline period as (tmin, tmax) for baseline correction.
        reference : str or list of str, optional
            EEG reference to apply before analysis.
        comment : str, optional
            Comment for the analysis (not used in computation).
        rois : list of str or int, optional
            ROI specification for channel filtering.
        channel_method : str, optional
            Method to aggregate across channels.
        trial_method : str, optional
            Method to aggregate across epochs.
        equipment : str, default="egi256"
            Equipment type for electrode mapping.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.baseline = baseline
        self.reference = reference
        self.comment = comment
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method
        super().__init__(
            tmin=tmin,
            tmax=tmax,
            equipment=equipment,
            on=on,
            name=name,
        )

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

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed time-locked contrast features.

        Raises
        ------
        ValueError
            If input data is not Epochs, if epochs are empty, or if conditions are missing.
        """
        from ..utils import filter_to_eeg_channels

        epochs = input["data"]

        if not hasattr(epochs, "events"):
            raise ValueError(
                "TimeLockedContrast requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(epochs) == 0:
            raise ValueError(
                "Cannot compute time-locked contrast on empty epochs."
            )

        # Filter to EEG channels
        epochs, eeg_ch_names, eeg_indices = filter_to_eeg_channels(epochs)

        # Apply re-referencing FIRST (if specified)
        if self.reference is not None:
            epochs = self._apply_reference(epochs, self.reference)

        # Apply baseline correction SECOND (if specified)
        if self.baseline is not None:
            epochs = epochs.copy().apply_baseline(self.baseline)

        # Store full epochs for non-aggregated case
        _ = epochs.copy()

        # Filter epochs by conditions using original implementation approach
        def get_epochs_for_condition(epochs, condition):
            """Get epochs for condition using MNE's built-in filtering."""
            # First try MNE's native indexing which supports hierarchical matching
            # epochs["go"] will match all events starting with "go/"
            try:
                return epochs[condition]
            except (KeyError, ValueError):
                pass

            # Handle list of conditions (e.g., ['60', '50'])
            if isinstance(condition, list):
                # MNE supports list indexing directly
                valid_conditions = []
                for cond in condition:
                    if cond in epochs.event_id:
                        valid_conditions.append(cond)
                    elif str(cond) in epochs.event_id:
                        valid_conditions.append(str(cond))
                    else:
                        try:
                            int_cond = int(cond)
                            if int_cond in epochs.event_id.values():
                                # Find key for this int value
                                for k, v in epochs.event_id.items():
                                    if v == int_cond:
                                        valid_conditions.append(k)
                                        break
                        except (ValueError, TypeError):
                            pass
                if not valid_conditions:
                    raise ValueError(
                        f"None of conditions {condition} found in epochs. "
                        f"Available event IDs: {list(epochs.event_id.keys())}"
                    )
                return epochs[valid_conditions]

            # Handle single condition (string or integer)
            if condition in epochs.event_id:
                # Condition exists as-is in event_id
                return epochs[condition]
            elif str(condition) in epochs.event_id:
                # Try string conversion
                return epochs[str(condition)]
            else:
                # Try integer conversion
                try:
                    int_condition = int(condition)
                    if int_condition in epochs.event_id:
                        return epochs[int_condition]
                except (ValueError, TypeError):
                    pass

                # Condition not found
                raise ValueError(
                    f"Condition {condition} not found or empty in epochs. "
                    f"Available event IDs: {list(epochs.event_id.keys())}"
                )

        epochs_a = get_epochs_for_condition(epochs, self.condition_a)
        epochs_b = get_epochs_for_condition(epochs, self.condition_b)

        # Helper function to process one condition using base class helpers
        def process_condition(epochs_cond):
            """Process one condition using base class helpers.

            This replicates TimeLockedTopography pipeline for one condition
            using the shared base class functionality.
            """
            # Prepare epochs data using base class helper
            data = self._prepare_epochs_data(epochs_cond, self.tmin, self.tmax)
            ch_names = list(epochs_cond.ch_names)

            # Apply ROI filtering using base class helper
            data, ch_names = self._apply_roi_filtering(
                data, ch_names, self.rois, self.equipment
            )

            # Average across time using base class helper
            time_averaged = self._average_across_time(data)

            return time_averaged, ch_names

        # Check if we should return raw temporal data without aggregation
        # Always process conditions separately with time averaging
        data_a, ch_names_a = process_condition(epochs_a)
        data_b, ch_names_b = process_condition(epochs_b)

        # Check if both conditions have the same number of epochs
        if data_a.shape[0] != data_b.shape[0]:
            raise ValueError(
                f"Conditions must have the same number of epochs. "
                f"Condition A has {data_a.shape[0]} epochs, "
                f"Condition B has {data_b.shape[0]} epochs."
            )

        # Now compute contrast: A - B (per epoch, per channel)
        contrast_result = data_a - data_b  # Shape: (n_epochs, n_channels)

        # Apply aggregation while preserving 2D structure
        contrast_result = apply_aggregation_preserve_dims(
            contrast_result,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
        )

        # Use centralized format_marker_result to ensure consistent col_names storage
        return format_marker_result(
            feature_name="timelockedcontrast",
            data=contrast_result,
            col_names=ch_names_a,
            channel_aggregated=(self.channel_method is not None),
        )
