"""Permutation Entropy marker for junifer_eeg."""

from itertools import permutations
from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class PermutationEntropy(BaseMarker):
    """Permutation Entropy marker.

    This marker computes the permutation entropy of EEG signals using the
    ordinal pattern method. Permutation entropy measures the complexity
    of a time series by quantifying the relative frequencies of ordinal
    patterns.

    Adapted from the NICE package implementation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scipy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"permutationentropy": "vector"},
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        kernel: int = 3,
        tau: int = 8,
        filter_freq: float | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        filter_order: int = 6,
        rois: list[str] | None = None,
        roi_aggregation_method: list[str] | None = None,
        trial_aggregation_method: list[str] | None = None,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the PermutationEntropy marker.

        Parameters
        ----------
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        kernel : int, default=3
            Length of ordinal patterns.
        tau : int, default=8
            Time delay for ordinal patterns.
        filter_freq : float, optional
            Low-pass filter frequency. If None, automatically computed.
        fmin : float, optional
            Lower frequency bound for band-pass filtering. If None, no filtering is applied.
        fmax : float, optional
            Upper frequency bound for band-pass filtering. If None, no filtering is applied.
        filter_order : int, optional
            Order of the Butterworth filter. Default: 6 (matches NICE implementation).
        rois : list of str, optional
            List of ROI names or electrode names to aggregate.
        roi_aggregation_method : list of str, optional
            List of aggregation methods for ROIs ('mean', 'std', 'median', 'min', 'max').
        trial_aggregation_method : list of str, optional
            List of aggregation methods for trials ('mean', 'std', 'median', 'min', 'max').
        epoch_length : float, default=2.0
            Length of epochs in seconds for trial aggregation.
        overlap : float, default=0.0
            Overlap between epochs (0.0 to 0.9).
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.kernel = kernel
        self.tau = tau
        self.filter_freq = filter_freq
        self.fmin = fmin
        self.fmax = fmax
        self.filter_order = filter_order
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.epoch_length = epoch_length
        self.overlap = overlap
        super().__init__(on=on, name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute permutation entropy.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed permutation entropy features.
        """
        from scipy.signal import butter, filtfilt

        # Get the MNE data object (can be Raw or Epochs)
        data_obj = input["data"]

        # Handle both Raw and Epochs objects
        if hasattr(data_obj, "get_data") and hasattr(data_obj, "events"):
            # This is an Epochs object
            # Check if epochs object is empty
            if len(data_obj) == 0:
                # Return empty results for empty epochs
                ch_names = data_obj.ch_names
                if self.rois is not None:
                    roi_data = {
                        roi: np.array([]).reshape(0, 0) for roi in self.rois
                    }
                else:
                    roi_data = {
                        ch: np.array([]).reshape(0, 0) for ch in ch_names
                    }

                return apply_roi_trial_aggregation(
                    roi_data,
                    roi_aggregation_methods=self.roi_aggregation_method,
                    trial_aggregation_methods=self.trial_aggregation_method,
                    marker_name="permutationentropy",
                )
            epochs_data = (
                data_obj.get_data()
            )  # Shape: (n_epochs, n_channels, n_times)
        else:
            # This is a Raw object, get data directly and reshape to 3D
            raw_data = data_obj.get_data()  # Shape: (n_channels, n_times)
            # Reshape to (1, n_channels, n_times) to treat as single epoch
            epochs_data = raw_data[np.newaxis, :, :]

        ch_names = data_obj.ch_names
        sfreq = data_obj.info["sfreq"]

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Apply frequency filtering based on parameters (matching WSMI approach)
        if self.fmin is not None and self.fmax is not None:
            # Bandpass filtering using fmin/fmax parameters (like WSMI)
            nyquist = sfreq / 2.0
            low = self.fmin / nyquist
            high = self.fmax / nyquist
            b, a = butter(self.filter_order, [low, high], btype="band")

            # NICE-style filtering: concatenate epochs horizontally, filter, then split back
            data_concat = np.hstack(
                epochs_data
            )  # Shape: (n_channels, total_time)

            # Filter concatenated data
            for ch_idx in range(n_channels):
                data_concat[ch_idx, :] = filtfilt(b, a, data_concat[ch_idx, :])

            # Split back and transpose exactly like NICE: [1, 2, 0]
            fdata = np.transpose(
                np.array(np.split(data_concat, n_epochs, axis=1)), [1, 2, 0]
            )
        elif self.filter_freq is not None:
            # Low-pass filtering using filter_freq parameter
            nyquist = sfreq / 2.0
            b, a = butter(4, self.filter_freq / nyquist, btype="low")

            # NICE-style filtering: concatenate epochs horizontally, filter, then split back
            data_concat = np.hstack(
                epochs_data
            )  # Shape: (n_channels, total_time)

            # Filter concatenated data
            for ch_idx in range(n_channels):
                data_concat[ch_idx, :] = filtfilt(b, a, data_concat[ch_idx, :])

            # Split back and transpose exactly like NICE: [1, 2, 0]
            fdata = np.transpose(
                np.array(np.split(data_concat, n_epochs, axis=1)), [1, 2, 0]
            )
        else:
            # Apply default filtering (following NICE approach exactly)
            filter_freq = np.double(sfreq) / self.kernel / self.tau
            b, a = butter(6, 2.0 * filter_freq / np.double(sfreq), "lowpass")

            # NICE-style filtering: concatenate epochs horizontally, filter, then split back
            data_concat = np.hstack(
                epochs_data
            )  # Shape: (n_channels, total_time)

            # Filter concatenated data
            for ch_idx in range(n_channels):
                data_concat[ch_idx, :] = filtfilt(b, a, data_concat[ch_idx, :])

            # Split back and transpose exactly like NICE: [1, 2, 0]
            fdata = np.transpose(
                np.array(np.split(data_concat, n_epochs, axis=1)), [1, 2, 0]
            )

        # Apply time mask AFTER filtering (like NICE)
        from mne.utils import _time_mask

        time_mask = _time_mask(data_obj.times, self.tmin, self.tmax)
        fdata = fdata[:, time_mask, :]

        # Convert back to (n_epochs, n_channels, n_samples) for our processing
        epochs_data = np.transpose(fdata, [2, 0, 1])

        # Compute permutation entropy for each epoch and channel
        pe_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal = epochs_data[epoch_idx, ch_idx, :]
                pe_values[epoch_idx, ch_idx] = (
                    self._compute_permutation_entropy(
                        signal,
                        self.kernel,
                        self.tau,
                    )
                )

        # Handle ROI selection
        if self.rois is not None:
            # Extract data for specified ROIs
            roi_data = get_data_for_rois(
                pe_values.T,  # Transpose to (n_channels, n_epochs)
                list(ch_names),
                self.rois,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: pe_values[:, i : i + 1].T for i, ch in enumerate(ch_names)
            }

        # Check if we should return per-epoch data without aggregation
        if (
            self.roi_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return per-epoch results without any aggregation
            col_names = []

            # Collect column names from all ROIs/channels
            for roi_name, roi_data_array in roi_data.items():
                # roi_data_array is (n_channels_in_roi, n_epochs)
                n_channels_in_roi = roi_data_array.shape[0]
                for ch_idx in range(n_channels_in_roi):
                    col_names.append(f"{roi_name}_ch{ch_idx}")

            # Stack data: each row is an epoch, each column is a channel
            epoch_data = []
            for epoch_idx in range(n_epochs):
                epoch_values = []
                for _, roi_data_array in roi_data.items():
                    # Extract values for this epoch across all channels in this ROI
                    for ch_idx in range(roi_data_array.shape[0]):
                        epoch_values.append(roi_data_array[ch_idx, epoch_idx])
                epoch_data.append(epoch_values)

            # Convert to numpy array: (n_epochs, n_channels)
            epoch_data_array = np.array(epoch_data)

            results = {
                "permutationentropy": {
                    "data": epoch_data_array,
                    "col_names": col_names,
                }
            }
        else:
            # Apply aggregation
            results = apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="permutationentropy",
            )

        return results

    def _compute_permutation_entropy(self, signal, kernel, tau):
        """Compute permutation entropy for a single filtered signal."""
        # Symbolic transformation
        symbols = self._define_symbols(kernel)

        # Calculate signal_sym shape
        signal_sym_length = len(signal) - tau * (kernel - 1)
        if signal_sym_length <= 0:
            # Return NaN for too short signals
            return np.nan

        signal_sym = np.zeros(signal_sym_length, dtype=np.int32)

        # Create ordinal patterns
        for k in range(signal_sym_length):
            subsamples = range(k, k + kernel * tau, tau)
            subsample_vals = signal[subsamples]
            ind = np.argsort(subsample_vals)
            pattern_str = "".join(map(str, ind))
            signal_sym[k] = symbols.index(pattern_str)

        # Count ordinal patterns
        n_symbols = len(symbols)
        count = np.bincount(signal_sym, minlength=n_symbols)
        count = count.astype(np.float64) / signal_sym_length

        # Compute permutation entropy
        with np.errstate(divide="ignore", invalid="ignore"):
            log_count = np.where(count > 0, np.log(count), 0)
            pe = -np.sum(count * log_count)

        # Normalize by maximum possible entropy
        return pe / np.log(n_symbols)

    def _define_symbols(self, kernel: int) -> list[str]:
        """Define symbols for permutation entropy.

        Parameters
        ----------
        kernel : int
            Length of ordinal patterns.

        Returns
        -------
        list of str
            List of ordinal pattern symbols.
        """
        symbols = []
        for perm in permutations(range(kernel)):
            symbols.append("".join(map(str, perm)))
        return symbols
