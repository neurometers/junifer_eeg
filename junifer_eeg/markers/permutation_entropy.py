"""Permutation Entropy marker for junifer_eeg."""

import math
from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois

# Try to import numba for acceleration
try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


def _pe_numpy(
    signal: np.ndarray, kernel: int, tau: int, fact: np.ndarray
) -> float:
    """Vectorized NumPy implementation of permutation entropy.

    Uses Lehmer code ranking instead of string operations for significant speedup
    while producing identical results.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    kernel : int
        Length of ordinal patterns.
    tau : int
        Time delay for ordinal patterns.
    fact : np.ndarray
        Precomputed factorials for kernel.

    Returns
    -------
    float
        Normalized permutation entropy.
    """
    # Length of ordinal windows
    L = signal.size - tau * (kernel - 1)
    if L <= 0:
        return np.nan

    # Build all window indices at once: shape (L, kernel)
    base = np.arange(L)[:, None]
    offs = (np.arange(kernel) * tau)[None, :]
    idx = base + offs  # (L, kernel)
    X = signal[idx]  # (L, kernel)

    # Get permutations for each row: argsort gives positions of ascending order
    # Use default quicksort (unstable) to match NICE's behavior
    P = np.argsort(X, axis=1)  # (L, kernel)

    # Compute Lehmer code row-wise
    # Lehmer code c[j] = number of elements to the right of position j
    # that are smaller than P[j]
    Lc = np.zeros((L, kernel), dtype=np.int64)
    for j in range(kernel - 1):
        pj = P[:, j][:, None]  # (L, 1)
        right = P[:, j + 1 :]  # (L, kernel-1-j)
        Lc[:, j] = np.sum(right < pj, axis=1)

    # Ranks from Lehmer code: sum c[j] * (k-1-j)!
    # factorials precomputed in `fact`
    weights = fact[kernel - 1 : 0 : -1]  # [(k-1)!, (k-2)!, ..., 1!]
    ranks = (Lc[:, :-1] * weights).sum(axis=1)

    # Histogram of pattern ranks
    n_symbols = fact[kernel]
    count = np.bincount(ranks, minlength=n_symbols).astype(np.float64)
    count /= L

    # Compute entropy (natural log) and normalize by log(n_symbols)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(count > 0, np.log(count), 0.0)
        pe = -np.sum(count * logp)

    return pe / np.log(n_symbols)


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=False)
    def _pe_numba(
        signal: np.ndarray, kernel: int, tau: int, fact: np.ndarray
    ) -> float:
        """Numba-accelerated implementation of permutation entropy.

        Uses JIT compilation for maximum performance while maintaining
        identical results to the NumPy implementation.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D array).
        kernel : int
            Length of ordinal patterns.
        tau : int
            Time delay for ordinal patterns.
        fact : np.ndarray
            Precomputed factorials for kernel.

        Returns
        -------
        float
            Normalized permutation entropy.
        """
        L = signal.size - tau * (kernel - 1)
        if L <= 0:
            return np.nan

        # Initialize histogram
        n_symbols = fact[kernel]
        counts = np.zeros(n_symbols, dtype=np.int64)

        # Process each window
        for start in range(L):
            # Collect window values
            vals = np.empty(kernel, dtype=np.float64)
            for j in range(kernel):
                vals[j] = signal[start + j * tau]

            # Compute permutation via stable sort (insertion sort)
            idxs = np.empty(kernel, dtype=np.int64)
            for j in range(kernel):
                idxs[j] = j

            # Insertion sort to get stable ordering
            for i in range(1, kernel):
                key_idx = idxs[i]
                key_val = vals[key_idx]
                j = i - 1
                while j >= 0 and vals[idxs[j]] > key_val:
                    idxs[j + 1] = idxs[j]
                    j -= 1
                idxs[j + 1] = key_idx

            # Compute Lehmer code and convert to rank
            rank = 0
            for j in range(kernel - 1):
                cj = 0
                pj = idxs[j]
                for i in range(j + 1, kernel):
                    if idxs[i] < pj:
                        cj += 1
                rank += cj * fact[kernel - 1 - j]

            counts[rank] += 1

        # Compute entropy
        total = float(L)
        pe = 0.0
        for i in range(counts.size):
            if counts[i] > 0:
                p = counts[i] / total
                pe -= p * math.log(p)

        return pe / math.log(n_symbols)
else:
    # Dummy function if numba is not available
    def _pe_numba(signal, kernel, tau, fact):
        """Placeholder function when Numba is not available."""
        raise RuntimeError("Numba not available")


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

        # Cache factorials for Lehmer code ranking
        self._fact = np.array(
            [math.factorial(i) for i in range(self.kernel + 1)],
            dtype=np.int64,
        )

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

        # Store original times for later time masking (like NICE does)
        # NICE applies time_mask AFTER filtering, not before
        original_times = data_obj.times

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Apply frequency filtering based on parameters
        # CRITICAL FIX: For frequency-specific PE, use ONLY bandpass (no additional lowpass)
        # The bandpass itself limits frequency content appropriately for PE computation
        if self.fmin is not None and self.fmax is not None:
            nyquist = sfreq / 2.0

            # Bandpass filter to isolate the frequency band
            low = self.fmin / nyquist
            high = self.fmax / nyquist
            b, a = butter(self.filter_order, [low, high], btype="band")

            # NICE-style filtering: concatenate epochs horizontally, filter, then split back
            data_concat = np.hstack(
                epochs_data
            )  # Shape: (n_channels, total_time)

            # Apply bandpass filter
            for ch_idx in range(n_channels):
                data_concat[ch_idx, :] = filtfilt(b, a, data_concat[ch_idx, :])

            # Split back and transpose exactly like NICE: [1, 2, 0]
            fdata = np.transpose(
                np.array(np.split(data_concat, n_epochs, axis=1)), [1, 2, 0]
            )

            # CRITICAL FIX: Apply time mask AFTER filtering (like NICE does)
            # NICE: time_mask = _time_mask(epochs.times, tmin, tmax); fdata = fdata[:, time_mask, :]
            from mne.utils import _time_mask

            time_mask = _time_mask(original_times, self.tmin, self.tmax)
            fdata = fdata[:, time_mask, :]

        elif self.filter_freq is not None:
            # Low-pass filtering using filter_freq parameter (match NICE's order=6)
            nyquist = sfreq / 2.0
            b, a = butter(
                self.filter_order, self.filter_freq / nyquist, btype="low"
            )

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

            # CRITICAL: Filter EXACTLY like NICE - apply filtfilt to entire array at once
            # filtfilt filters along last axis by default, which is what we want
            filtered_data = filtfilt(b, a, data_concat)

            # Split back and transpose exactly like NICE: [1, 2, 0]
            fdata = np.transpose(
                np.array(np.split(filtered_data, n_epochs, axis=1)), [1, 2, 0]
            )

            # CRITICAL FIX: Apply time mask AFTER filtering (like NICE does)
            from mne.utils import _time_mask

            time_mask = _time_mask(original_times, self.tmin, self.tmax)
            fdata = fdata[:, time_mask, :]

        # NICE approach: Compute PE on concatenated signal
        # fdata shape after time mask: (n_channels, n_times_cropped, n_epochs)
        # We need to concatenate along time dimension: (n_channels, total_time)
        # Optimize concatenation: transpose and reshape instead of Python loops
        # This produces identical layout to np.hstack([fdata[:,:,epoch] for epoch in range(n_epochs)])
        concatenated_data = (
            np.ascontiguousarray(fdata)
            .transpose(0, 2, 1)
            .reshape(n_channels, -1)
        )
        # Result shape: (n_channels, total_time) where total_time = n_times_cropped * n_epochs

        # Compute permutation entropy on concatenated signal (like NICE)
        pe_values = np.zeros(n_channels, dtype=np.float64)

        for ch_idx in range(n_channels):
            signal = concatenated_data[ch_idx, :]
            pe_values[ch_idx] = self._compute_permutation_entropy(
                signal,
                self.kernel,
                self.tau,
            )

        # Convert to expected output format: (n_epochs, n_channels) with same value repeated
        # Since we computed on concatenated signal, all epochs get the same PE value per channel
        pe_values_expanded = np.tile(pe_values, (n_epochs, 1))

        # Handle ROI selection
        if self.rois is not None:
            # Extract data for specified ROIs
            roi_data = get_data_for_rois(
                pe_values_expanded.T,  # Transpose to (n_channels, n_epochs)
                list(ch_names),
                self.rois,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: pe_values_expanded[:, i : i + 1].T
                for i, ch in enumerate(ch_names)
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
        """Compute permutation entropy for a single filtered signal.

        Uses optimized Numba implementation if available, otherwise falls back
        to vectorized NumPy implementation. Both produce identical results to
        the original string-based implementation but are significantly faster.
        """
        signal = np.asarray(signal, dtype=np.float64)

        if _HAVE_NUMBA:
            return _pe_numba(signal, kernel, tau, self._fact)
        else:
            return _pe_numpy(signal, kernel, tau, self._fact)
