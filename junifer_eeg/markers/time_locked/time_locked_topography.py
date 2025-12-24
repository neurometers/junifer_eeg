"""Time-locked topography marker for junifer_eeg - refactored version.

This marker extracts time-locked topographies (ERPs) by averaging
across specified time windows, using shared base class helpers.

Refactored to use TimeLockedBase for common data preparation operations.
"""

from typing import Any, ClassVar, List, Union

from junifer.api.decorators import register_marker

from ..base import format_marker_result
from ..utils import apply_aggregation_preserve_dims, filter_to_eeg_channels
from ._time_locked_base import TimeLockedBase


@register_marker
class TimeLockedTopography(TimeLockedBase):
    """Time-locked topography marker.

    This marker extracts time-locked topographies (ERPs) by averaging
    across specified time windows.

    Follows next_icm aggregation pattern:
    1. Average across time points (tmin to tmax)
    2. Aggregate across electrodes (using channel_method)
    3. Aggregate across trials (using trial_method)

    Refactored to use shared base class helpers for ROI filtering and data preparation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"timelockedtopo": "timeseries"}
    }  # 2D: (epochs, channels)

    def __init__(
        self,
        tmin: float,
        tmax: float,
        baseline: tuple[float, float] | None = None,
        reference: str | list[str] | None = None,
        rois: Union[List[str], List[int], None] = None,
        channel_method: str | None = None,
        trial_method: str | None = None,
        equipment: str = "standard",
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the TimeLockedTopography marker.

        Parameters
        ----------
        tmin : float
            Start time relative to epoch center in seconds.
        tmax : float
            End time relative to epoch center in seconds.
        baseline : tuple of float, optional
            Baseline period as (tmin, tmax) for baseline correction.
            Example: (-0.2, 0) for 200ms pre-stimulus baseline.
        reference : str or list of str, optional
            EEG reference to apply before analysis. Validates channels
            against actual data (like ROI system). Options:
            - 'average': Average reference across all channels
            - Channel name string: Single channel (e.g., 'Cz', 'TP9')
            - List of channel names: Multiple channels (e.g., ['TP9', 'TP10'])
            Examples: 'average', 'Cz', ['TP9', 'TP10']
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp')

            Examples:
            - list(range(224)) - NICE scalp ROI via indices
            - ['scalp'] - Semantic ROI (expands to all scalp channels)

            If None, uses all channels.
        channel_method : str or None, optional
            Methods to aggregate across channels: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        trial_method : str or None, optional
            Methods to aggregate across epochs: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        equipment : str, default="standard"
            Equipment type for electrode mapping.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.baseline = baseline
        self.reference = reference
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

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type - always returns timeseries for 2D tensor data.

        Always returns 'timeseries' since we now always return
        2D tensors with shape (n_epochs, n_channels)
        where dimensions can be size 1 when aggregated.
        """
        return "timeseries"

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute time-locked topography.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed time-locked topography features.

        Raises
        ------
        ValueError
            If input data is not Epochs or if epochs are empty.
        """
        # Get the MNE data object - must be Epochs
        data_obj = input["data"]

        if not hasattr(data_obj, "events"):
            raise ValueError(
                "TimeLockedTopography requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(data_obj) == 0:
            raise ValueError(
                "Cannot compute time-locked topography on empty epochs."
            )

        # Filter to only EEG channels (exclude EOG, stim, etc.)
        epochs, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Apply re-referencing FIRST (if specified)
        if self.reference is not None:
            epochs = self._apply_reference(epochs, self.reference)

        # Apply baseline correction SECOND (before cropping)
        # This ensures baseline period is available for correction
        if self.baseline is not None:
            epochs = epochs.copy().apply_baseline(self.baseline)

        # Prepare epochs data using base class helper
        data = self._prepare_epochs_data(epochs, self.tmin, self.tmax)
        ch_names = list(epochs.ch_names)

        # Apply ROI filtering using base class helper
        data, ch_names = self._apply_roi_filtering(
            data, ch_names, self.rois, self.equipment
        )

        # Get data dimensions
        n_epochs, n_channels, n_times = data.shape

        # Always average across time first using base class helper
        # This is the semantic meaning of time-locked topography (ERP)
        time_averaged = self._average_across_time(
            data
        )  # Shape: (n_epochs, n_channels)

        # Apply aggregation while preserving 2D structure
        # Use trial→channel order (matches original NICE behavior)
        result_data = apply_aggregation_preserve_dims(
            time_averaged,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
            aggregation_order="trial_channel",
        )

        # Use centralized format_marker_result to ensure consistent col_names storage
        return format_marker_result(
            feature_name="timelockedtopo",
            data=result_data,  # Shape: (n_epochs, n_channels) with size-1 for aggregated dims
            col_names=ch_names,
            channel_aggregated=(self.channel_method is not None),
        )
