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
        "EEG": {"permutationentropy": "vector"}
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        kernel: int = 3,
        tau: int = 8,
        filter_freq: float | None = None,
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
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.epoch_length = epoch_length
        self.overlap = overlap
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
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
        from mne.utils import _time_mask
        from scipy.signal import butter, filtfilt

        # Get the MNE Epochs object
        epochs = input["data"]

        # Get epochs data: Shape (n_epochs, n_channels, n_times)
        epochs_data = epochs.get_data()
        ch_names = epochs.ch_names
        sfreq = epochs.info["sfreq"]

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Apply time window if specified
        if self.tmin is not None or self.tmax is not None:
            time_mask = _time_mask(epochs.times, self.tmin, self.tmax)
            epochs_data = epochs_data[:, :, time_mask]

        # Apply frequency filtering if specified
        if self.filter_freq is not None:
            # Design low-pass filter
            nyquist = sfreq / 2.0
            if self.filter_freq >= nyquist:
                raise ValueError(
                    f"Filter frequency ({self.filter_freq}) must be less than "
                    f"Nyquist frequency ({nyquist})"
                )

            b, a = butter(4, self.filter_freq / nyquist, btype="low")

            # Apply filter to each epoch and channel
            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    epochs_data[epoch_idx, ch_idx, :] = filtfilt(
                        b, a, epochs_data[epoch_idx, ch_idx, :]
                    )
        else:
            # Apply default filtering (following NICE approach)
            filter_freq = np.double(sfreq) / self.kernel / self.tau
            b, a = butter(6, 2.0 * filter_freq / np.double(sfreq), "lowpass")

            # Apply filter to each epoch and channel
            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    epochs_data[epoch_idx, ch_idx, :] = filtfilt(
                        b, a, epochs_data[epoch_idx, ch_idx, :]
                    )

        # Compute permutation entropy for each epoch and channel
        pe_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal = epochs_data[epoch_idx, ch_idx, :]
                pe_values[epoch_idx, ch_idx] = (
                    self._compute_permutation_entropy(
                        signal, self.kernel, self.tau
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
