"""Kolmogorov Complexity marker for junifer_eeg."""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import aggregate_data, get_data_for_rois


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
        "EEG": {
            "kolmogorovcomplexity": "timeseries"
        },  # 2D: (epochs, channels)
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        nbins: int = 16,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        rois: Union[List[str], List[int], None] = None,
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
        equipment: str = "egi256",
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
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1', 'E224') OR semantic ROI (e.g., 'frontal', 'scalp')

            Examples:
            - [0, 1, 2, ..., 223] - NICE scalp ROI via indices
            - ['E1', 'E2', ..., 'E224'] - NICE scalp ROI via names
            - ['scalp'] - Semantic ROI (expands to all scalp channels)
            - [0, 'E5', 10, 'frontal'] - Mix of types

            If None, uses all channels.
        channel_aggregation_method : str, optional
            Methods to aggregate across ROI electrodes/channels: 'mean', 'std',
            'median', 'trim_mean80', 'trim_mean90', etc.
        trial_aggregation_method : str, optional
            Methods to aggregate across trials/epochs: 'mean', 'std',
            'median', 'trim_mean80', 'trim_mean90', etc.
        equipment : str, optional
            Equipment name for named ROIs.
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
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns 'timeseries' for 2D tensor data (no aggregation) and 'vector'
        for aggregated 1D/scalar results.
        """
        # No aggregation → 2D tensor (epochs, channels) → use timeseries
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            return "timeseries"
        # Aggregation applied → 1D or scalar → use vector
        return "vector"

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

        from .utils import filter_to_eeg_channels

        # Get the MNE object (can be Raw or Epochs)
        data_obj = input["data"]

        # Filter to only EEG channels (exclude EOG, stim, etc.)
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

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
        ch_names = list(data_obj.ch_names)

        # Apply ROI filtering BEFORE computation if specified
        if self.rois is not None:
            # Transpose to (n_channels, n_epochs, n_samples)
            data_transposed = epochs_data.transpose(1, 0, 2)

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
                # Transpose back to (n_epochs, n_channels, n_samples)
                epochs_data = data_filtered.transpose(1, 0, 2)
                n_epochs, n_channels, n_samples = epochs_data.shape
                # Preserve the actual ROI channel names
                ch_names = self.rois

        # Compute Kolmogorov complexity for each channel and epoch
        k_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            for channel_idx in range(n_channels):
                signal = epochs_data[epoch_idx, channel_idx, :]
                k_values[epoch_idx, channel_idx] = (
                    self._compute_kolmogorov_for_signal(signal)
                )

        # k_values shape: (n_epochs, n_channels)

        # Check if we should return raw data without aggregation
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return raw per-epoch, per-channel data
            col_names = [f"{ch}" for ch in ch_names]
            return {
                "kolmogorovcomplexity": {
                    "data": k_values,
                    "col_names": col_names,
                }
            }

        # Apply aggregation
        result_data = k_values

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

        # Return result data without unnecessary reshaping - preserve tensor structure
        # Scalar: keep as scalar
        # 1D array: keep as 1D (n_trials) or (n_channels)
        # 2D array: keep as 2D (n_trials, n_channels)

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
            "kolmogorovcomplexity": {
                "data": result_data,
                "col_names": col_names,
            }
        }

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
