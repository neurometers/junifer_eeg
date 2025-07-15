"""Kolmogorov Complexity marker for junifer_eeg."""

from typing import Any, ClassVar, List, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class KolmogorovComplexity(BaseMarker):
    """Kolmogorov Complexity marker with flexible ROI and trial aggregation.

    This marker computes the Kolmogorov complexity of EEG signals using
    compression-based approximation. The signal is first symbolized into
    discrete bins, then compressed using zlib, and the complexity is
    measured as the compression ratio.

    Supports channel-wise computation with flexible ROI and trial aggregation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"kolmogorovcomplexity": "vector"},
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        nbins: int = 16,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the KolmogorovComplexity marker.

        Parameters
        ----------
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        nbins : int, default=16
            Number of bins for signal symbolization.
        epoch_length : float, default=2.0
            Length of epochs to create from continuous data in seconds.
        overlap : float, default=0.0
            Overlap between epochs (0.0 = no overlap, 0.9 = 90% overlap).
        rois : list of str, optional
            List of ROI names. If None, use all channels.
        roi_aggregation_method : list of str, optional
            Methods to aggregate across ROI electrodes: ['mean', 'std'].
        trial_aggregation_method : list of str, optional
            Methods to aggregate across trials/epochs: ['mean', 'std'].
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.nbins = nbins
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        super().__init__(on=on, name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Kolmogorov complexity with flexible aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed Kolmogorov complexity features with aggregation.
        """

        from mne.utils import _time_mask

        # Get the MNE object (can be Raw or Epochs)
        data_obj = input["data"]

        # Handle both Raw and Epochs objects
        if hasattr(data_obj, "get_data") and hasattr(data_obj, "events"):
            # This is an Epochs object
            epochs_data = (
                data_obj.get_data()
            )  # Shape: (n_epochs, n_channels, n_times)

            # Apply time cropping if specified
            if self.tmin is not None or self.tmax is not None:
                data_obj = data_obj.copy().crop(tmin=self.tmin, tmax=self.tmax)
                epochs_data = data_obj.get_data()
        else:
            # This is a Raw object
            # Create epochs from continuous data if needed
            if self.trial_aggregation_method is not None:
                epochs_data = self._create_epochs_from_continuous(data_obj)
                # epochs_data shape: (n_epochs, n_channels, n_samples)
            else:
                # Single "epoch" from continuous data
                data = data_obj.get_data()  # Shape: (n_channels, n_times)
                if self.tmin is not None or self.tmax is not None:
                    time_mask = _time_mask(
                        data_obj.times, self.tmin, self.tmax
                    )
                    data = data[:, time_mask]
                epochs_data = data[
                    np.newaxis,
                    :,
                    :,
                ]  # Shape: (1, n_channels, n_samples)

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Compute Kolmogorov complexity for each channel and epoch
        k_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            for channel_idx in range(n_channels):
                signal = epochs_data[epoch_idx, channel_idx, :]
                k_values[epoch_idx, channel_idx] = (
                    self._compute_kolmogorov_for_signal(signal)
                )

        # Handle ROI selection
        if self.rois is not None:
            # Extract data for specified ROIs
            roi_data = get_data_for_rois(
                k_values.T,  # Transpose to (n_channels, n_epochs)
                list(data_obj.ch_names),
                self.rois,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: k_values[:, i : i + 1].T
                for i, ch in enumerate(data_obj.ch_names)
            }

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="KolmogorovComplexity",
        )

        return results

    def _create_epochs_from_continuous(self, raw):
        """Create epochs from continuous data."""
        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_times)

        # Apply time mask if specified
        if self.tmin is not None or self.tmax is not None:
            from mne.utils import _time_mask

            time_mask = _time_mask(raw.times, self.tmin, self.tmax)
            data = data[:, time_mask]

        n_channels, n_samples = data.shape
        sfreq = raw.info["sfreq"]

        # Calculate epoch parameters
        epoch_samples = int(self.epoch_length * sfreq)
        overlap_samples = int(self.overlap * epoch_samples)
        step_samples = epoch_samples - overlap_samples

        # Calculate number of epochs
        n_epochs = max(1, (n_samples - epoch_samples) // step_samples + 1)

        # Create epochs
        epochs_data = np.zeros((n_epochs, n_channels, epoch_samples))

        for epoch_idx in range(n_epochs):
            start_sample = epoch_idx * step_samples
            end_sample = start_sample + epoch_samples

            if end_sample <= n_samples:
                epochs_data[epoch_idx] = data[:, start_sample:end_sample]
            else:
                # Pad with last available samples if needed
                available_samples = n_samples - start_sample
                epochs_data[epoch_idx, :, :available_samples] = data[
                    :,
                    start_sample:,
                ]
                # Pad with zeros or repeat last sample
                epochs_data[epoch_idx, :, available_samples:] = data[:, -1:]

        return epochs_data

    def _compute_kolmogorov_for_signal(self, signal: np.ndarray) -> float:
        """Compute Kolmogorov complexity for a single signal."""
        import zlib

        # Symbolic transformation - match original algorithm exactly
        ssignal = np.sort(signal)
        items = signal.shape[0]
        first = int(items / 10)
        last = items - first if first > 1 else items - 1
        lower = ssignal[first]
        upper = ssignal[last]
        bsize = (upper - lower) / self.nbins

        # Create symbolic representation
        osignal = np.zeros(signal.shape, dtype=np.uint8)
        maxbin = self.nbins - 1

        for i in range(items):
            tbin = int((signal[i] - lower) / bsize) if bsize > 0 else 0
            osignal[i] = (0 if tbin < 0 else min(tbin, maxbin)) + ord("A")

        # Use zlib for compression
        string = osignal.tobytes()
        cstring = zlib.compress(string)
        return float(len(cstring)) / float(len(string))
