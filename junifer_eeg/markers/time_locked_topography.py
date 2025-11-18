"""Time-locked topography marker for junifer_eeg."""

from typing import Any, ClassVar, List, Union

import mne
import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import aggregate_data, get_data_for_rois


@register_marker
class TimeLockedTopography(BaseMarker):
    """Time-locked topography marker.

    This marker extracts time-locked topographies (ERPs) by creating
    epochs from continuous data and averaging across specified time windows.

    Follows next_icm aggregation pattern:
    1. Average across time points (tmin to tmax)
    2. Aggregate across electrodes (using channel_aggregation_method)
    3. Aggregate across trials (using trial_aggregation_method)

    Note: This is adapted from NICE for continuous data. The original NICE
    implementation worked with pre-epoched data.
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
        epoch_length: float = 2.0,
        overlap: float = 0.0,
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
        equipment : str, optional
            Equipment type for electrode mapping. Default: 'standard'
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.epoch_length = epoch_length
        self.overlap = overlap
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
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed time-locked topography features.
        """
        from .utils import filter_to_eeg_channels

        # Get the MNE data object (can be Raw or Epochs)
        data_obj = input["data"]

        # Filter to only EEG channels (exclude EOG, stim, etc.)
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Check if we have Epochs or Raw data
        if hasattr(data_obj, "get_data") and hasattr(data_obj, "events"):
            # This is already Epochs data - use it directly
            epochs = data_obj
        else:
            # This is Raw data - create epochs from continuous data
            raw = data_obj
            duration = self.epoch_length
            overlap_samples = int(self.overlap * duration * raw.info["sfreq"])

            # Create events at regular intervals
            sfreq = raw.info["sfreq"]
            duration_samples = int(duration * sfreq)
            step_samples = duration_samples - overlap_samples

            # Calculate the number of epochs we can create
            n_samples = raw.n_times
            n_epochs = max(
                1, (n_samples - duration_samples) // step_samples + 1
            )

            # Create event array
            events = np.zeros((n_epochs, 3), dtype=int)
            for i in range(n_epochs):
                events[i, 0] = (
                    i * step_samples + duration_samples // 2
                )  # Event at epoch center
                events[i, 2] = 1  # Event ID

            # Make sure events don't exceed data length
            valid_events = events[
                events[:, 0] < n_samples - duration_samples // 2
            ]

            if len(valid_events) == 0:
                raise ValueError("Data too short to create any epochs")

            # Create epochs
            epochs = mne.Epochs(
                raw,
                valid_events,
                event_id={"epoch": 1},
                tmin=-duration / 2,
                tmax=duration / 2,
                baseline=None,  # We'll apply baseline later if needed
                preload=True,
                verbose=False,
            )

        # Check for empty epochs first
        if len(epochs) == 0:
            # Return empty results for empty epochs
            return {
                "timelockedtopo": {
                    "data": np.array([[]]),
                    "col_names": [],
                }
            }

        # Apply re-referencing FIRST (if specified)
        if self.reference is not None:
            epochs = self._apply_reference(epochs)

        # Apply baseline correction SECOND (before cropping)
        # This ensures baseline period is available for correction
        if self.baseline is not None:
            epochs = epochs.copy().apply_baseline(self.baseline)

        # Clamp to the available epoch time range to avoid MNE errors
        epoch_min = epochs.tmin
        epoch_max = epochs.tmax
        crop_tmin = self.tmin
        crop_tmax = self.tmax
        if crop_tmin is None or crop_tmin < epoch_min:
            crop_tmin = epoch_min
        if crop_tmax is None or crop_tmax > epoch_max:
            crop_tmax = epoch_max

        # Crop epochs to the specified time window (after baseline correction)
        epochs_cropped = epochs.crop(tmin=crop_tmin, tmax=crop_tmax)

        # Get the raw time-series data
        data = (
            epochs_cropped.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)
        ch_names = list(epochs_cropped.ch_names)
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
                # Preserve the actual ROI channel names
                ch_names = self.rois

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

        # For aggregation, average across time first
        time_averaged = np.mean(data, axis=2)  # Shape: (n_epochs, n_channels)

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
