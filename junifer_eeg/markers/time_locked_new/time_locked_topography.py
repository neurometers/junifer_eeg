"""Time-locked topography marker for junifer_eeg - refactored version.

This marker extracts time-locked topographies (ERPs) by averaging
across specified time windows, using shared base class helpers.

Refactored to use TimeLockedBase for common data preparation operations.
"""

from typing import Any, ClassVar, List, Union

from junifer.api.decorators import register_marker

from ..utils import aggregate_data
from ._time_locked_base import TimeLockedBase


@register_marker
class TimeLockedTopography(TimeLockedBase):
    """Time-locked topography marker.

    This marker extracts time-locked topographies (ERPs) by averaging
    across specified time windows.

    Follows next_icm aggregation pattern:
    1. Average across time points (tmin to tmax)
    2. Aggregate across electrodes (using channel_aggregation_method)
    3. Aggregate across trials (using trial_aggregation_method)

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
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
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
        channel_aggregation_method : str or None, optional
            Methods to aggregate across channels: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        trial_aggregation_method : str or None, optional
            Methods to aggregate across epochs: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        equipment : str, default="standard"
            Equipment type for electrode mapping.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.reference = reference
        self.rois = rois
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 3D tensor data (no aggregation)
        - 'vector': 1D array (one aggregation applied)
        - 'scalar_table': scalar value (both aggregations applied)

        Dimensionality:
        - No aggregation: (epochs, channels, times) → 3D → timeseries
        - One aggregation: time-averaged → then aggregated → 1D → vector
        - Both aggregations: time-averaged → fully aggregated → scalar → scalar_table
        """
        # No aggregation → 3D tensor (epochs, channels, times) → use timeseries
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            return "timeseries"

        # Both aggregations → scalar → use scalar_table
        if (
            self.channel_aggregation_method is not None
            and self.trial_aggregation_method is not None
        ):
            return "scalar_table"

        # One aggregation → 1D array → use vector
        return "vector"

    def _apply_reference(self, epochs):
        """Apply EEG re-referencing using flexible channel specification.

        Parameters
        ----------
        epochs : mne.Epochs
            Epochs to re-reference

        Returns
        -------
        epochs : mne.Epochs
            Re-referenced epochs

        Notes
        -----
        Reference can be specified as:
        - 'average': Average reference across all channels
        - String channel name: Single channel (e.g., 'Cz', 'TP9')
        - List of channel names: Multiple channels (e.g., ['TP9', 'TP10'])
        - Reference channels are resolved from actual data, like ROIs
        """
        reference = self.reference
        ch_names = epochs.ch_names

        # Handle special case: 'average' reference
        if reference == "average":
            epochs_reref = epochs.copy().set_eeg_reference(
                ref_channels="average"
            )
            return epochs_reref

        # Handle single string channel name
        if isinstance(reference, str):
            if reference in ch_names:
                # Valid channel name
                epochs_reref = epochs.copy().set_eeg_reference(
                    ref_channels=reference
                )
                return epochs_reref
            else:
                raise ValueError(
                    f"Reference channel '{reference}' not found in data. "
                    f"Available channels: {ch_names[:20]}... "
                    f"(showing first 20 of {len(ch_names)})"
                )

        # Handle list of channel names
        if isinstance(reference, list):
            # Validate all channels exist in data
            invalid_channels = [ch for ch in reference if ch not in ch_names]
            if invalid_channels:
                raise ValueError(
                    f"Reference channel(s) {invalid_channels} not found in data. "
                    f"Available channels: {ch_names[:20]}... "
                    f"(showing first 20 of {len(ch_names)})"
                )

            # All channels valid
            epochs_reref = epochs.copy().set_eeg_reference(
                ref_channels=reference
            )
            return epochs_reref

        # Should not reach here, but handle unexpected types
        raise TypeError(
            f"Reference must be 'average', string channel name, or list of channel names. "
            f"Got: {type(reference)}"
        )

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
        from ..utils import filter_to_eeg_channels

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
            epochs = self._apply_reference(epochs)

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

        # Check if we should return raw temporal data without aggregation
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return raw per-epoch, per-channel, per-time results (matching NICE)
            # Shape: (n_epochs, n_channels, n_times)
            return {
                "timelockedtopo": {
                    "data": data,  # Keep full temporal dimension
                }
            }

        # For aggregation, average across time first using base class helper
        time_averaged = self._average_across_time(
            data
        )  # Shape: (n_epochs, n_channels)

        # Apply aggregation
        result_data = time_averaged

        # Step 1: Channel aggregation (aggregate across axis=1)
        if self.channel_aggregation_method is not None:
            result_data = aggregate_data(
                result_data,
                self.channel_aggregation_method,
                axis=1,
            )
            # After channel agg: (n_epochs,)

        # Step 2: Trial aggregation
        if self.trial_aggregation_method is not None:
            if result_data.ndim == 1:
                # Already reduced by channel agg: (n_epochs,)
                result_data = aggregate_data(
                    result_data,
                    self.trial_aggregation_method,
                    axis=None,
                )
                # Result: scalar
            else:
                # No channel agg yet: (n_epochs, n_channels)
                result_data = aggregate_data(
                    result_data,
                    self.trial_aggregation_method,
                    axis=0,
                )
                # Result: (n_channels,)

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

        return {
            "timelockedtopo": {
                "data": result_data,
                "col_names": col_names,
            }
        }
