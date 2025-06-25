"""Time-locked topography marker for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


@register_marker
class TimeLockedTopography(BaseMarker):
    """Time-locked topography marker.

    This marker extracts time-locked topographies (ERPs) by creating
    epochs from continuous data and averaging across specified time windows.

    Note: This is adapted from NICE for continuous data. The original NICE
    implementation worked with pre-epoched data.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"time_locked_topo": "matrix"}}

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 0.8,
        epoch_length: float = 2.0,
        overlap: float = 0.5,
        baseline: tuple[float, float] | None = None,
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
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
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
        import mne

        # Get the MNE Raw object
        raw = input["data"]

        # Create epochs from continuous data
        # This creates non-overlapping or overlapping epochs from continuous data
        duration = self.epoch_length
        overlap_samples = int(self.overlap * duration * raw.info["sfreq"])

        # Create events at regular intervals
        sfreq = raw.info["sfreq"]
        duration_samples = int(duration * sfreq)
        step_samples = duration_samples - overlap_samples

        # Calculate the number of epochs we can create
        n_samples = raw.n_times
        n_epochs = max(1, (n_samples - duration_samples) // step_samples + 1)

        # Create event array
        events = np.zeros((n_epochs, 3), dtype=int)
        for i in range(n_epochs):
            events[i, 0] = (
                i * step_samples + duration_samples // 2
            )  # Event at epoch center
            events[i, 2] = 1  # Event ID

        # Make sure events don't exceed data length
        valid_events = events[events[:, 0] < n_samples - duration_samples // 2]

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

        # Apply baseline correction if specified
        if self.baseline is not None:
            epochs.apply_baseline(self.baseline)

        # Crop to requested time window
        epochs_cropped = epochs.copy().crop(tmin=self.tmin, tmax=self.tmax)

        # Get the data
        data = (
            epochs_cropped.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)

        # Average across epochs to get ERP
        erp_data = np.mean(data, axis=0)  # Shape: (n_channels, n_times)

        # Create time labels
        times = epochs_cropped.times
        time_labels = [f"t_{t:.3f}s" for t in times]

        # Return data in junifer format
        return {
            "time_locked_topo": {
                "data": erp_data,  # Shape: (n_channels, n_times)
                "col_names": time_labels,
                "row_names": list(raw.ch_names),
            }
        }
