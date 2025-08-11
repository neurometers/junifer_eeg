"""Time-locked topography marker for junifer_eeg."""

from typing import Any, ClassVar, List, Optional

import mne
import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class TimeLockedTopography(BaseMarker):
    """Time-locked topography marker.

    This marker extracts time-locked topographies (ERPs) by creating
    epochs from continuous data and averaging across specified time windows.

    Follows next_icm aggregation pattern:
    1. Average across time points (tmin to tmax)
    2. Aggregate across electrodes (using roi_aggregation_method)
    3. Aggregate across trials (using trial_aggregation_method)

    Note: This is adapted from NICE for continuous data. The original NICE
    implementation worked with pre-epoched data.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"timelockedtopo": "vector"}}

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 0.8,
        epoch_length: float = 2.0,
        overlap: float = 0.5,
        baseline: tuple[float, float] | None = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        equipment: str = "standard",
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the TimeLockedTopography marker.

        Parameters
        ----------
        tmin : float, default=-0.2
            Start time relative to epoch center in seconds.
        tmax : float, default=0.8
            End time relative to epoch center in seconds.
        epoch_length : float, default=2.0
            Length of epochs to create from continuous data.
        overlap : float, default=0.5
            Overlap between epochs (0.0 = no overlap, 0.9 = 90% overlap).
        baseline : tuple of float, optional
            Baseline correction period (start, end) in seconds.
        rois : list of str, optional
            List of ROI names for aggregation.
        roi_aggregation_method : list of str, optional
            Methods to aggregate across ROI electrodes: ['mean', 'std'].
        trial_aggregation_method : list of str, optional
            Methods to aggregate across trials/epochs: ['mean', 'std'].
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
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        super().__init__(on=on, name=name)

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
        # Get the MNE data object (can be Raw or Epochs)
        data_obj = input["data"]

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
            ch_names = epochs.ch_names
            if self.rois is not None:
                roi_data = {
                    roi: np.array([]).reshape(0, 0) for roi in self.rois
                }
            else:
                roi_data = {ch: np.array([]).reshape(0, 0) for ch in ch_names}

            # Apply aggregation to empty data
            results = apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="timelockedtopo",
            )
            return results

        # Clamp to the available epoch time range to avoid MNE errors
        epoch_min = epochs.tmin
        epoch_max = epochs.tmax
        crop_tmin = self.tmin
        crop_tmax = self.tmax
        if crop_tmin is None or crop_tmin < epoch_min:
            crop_tmin = epoch_min
        if crop_tmax is None or crop_tmax > epoch_max:
            crop_tmax = epoch_max

        # Crop epochs to the specified time window
        epochs_cropped = epochs.copy().crop(tmin=crop_tmin, tmax=crop_tmax)

        # Apply baseline correction if specified
        if self.baseline is not None:
            epochs_cropped.apply_baseline(self.baseline)

        # Get the raw time-series data (like NICE implementation)
        data = (
            epochs_cropped.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)

        # Follow NICE pattern: Apply time window selection first, then average
        # NICE uses time_mask to select time points, then reduction functions handle averaging
        # Since we already cropped to the time window, now we average across time
        # This matches NICE's approach where time averaging happens after time selection
        time_averaged = np.mean(data, axis=2)  # Shape: (n_epochs, n_channels)

        # Reshape to (n_channels, n_epochs) for aggregation framework
        time_averaged = time_averaged.T  # Shape: (n_channels, n_epochs)

        # Apply ROI selection
        if self.rois is not None:
            roi_data = get_data_for_rois(
                time_averaged,  # (n_channels, n_epochs)
                list(epochs.ch_names),
                self.rois,
                equipment=self.equipment,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: time_averaged[i : i + 1, :]  # (1, n_epochs)
                for i, ch in enumerate(epochs.ch_names)
            }

        # Apply aggregation to get final clinical values
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="timelockedtopo",
        )

        return results
