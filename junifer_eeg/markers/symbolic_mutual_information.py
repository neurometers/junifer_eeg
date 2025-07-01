"""Symbolic mutual information marker for EEG analysis."""

import math
from itertools import permutations
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker


@register_marker
class SymbolicMutualInformation(BaseMarker):
    """Symbolic mutual information marker for connectivity analysis.

    This marker computes symbolic mutual information between EEG channels,
    providing a measure of nonlinear coupling between brain regions.
    Based on the NICE package implementation with ordinal pattern transformation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scipy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "symbolic_mutual_information": "matrix",
        },
    }

    def __init__(
        self,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        kernel: int = 3,
        tau: int = 8,
        weighted: bool = True,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[str | List[str]] = None,
        trial_aggregation_method: Optional[str | List[str]] = None,
        equipment: str = "standard",
        epoch_length: Optional[float] = None,
        overlap: float = 0.0,
        on: Optional[str | List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SymbolicMutualInformation marker."""
        self.tmin = tmin
        self.tmax = tmax
        self.kernel = kernel
        self.tau = tau
        self.weighted = weighted
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        self.epoch_length = epoch_length
        self.overlap = overlap

        super().__init__(on=on, name=name)

    def _define_symbols(self, kernel: int) -> list[str]:
        """Define all possible symbols for a given kernel size (NICE implementation)."""
        result_dict = {}
        total_symbols = math.factorial(kernel)
        cursymbol = 0
        for perm in permutations(range(kernel)):
            order = "".join(map(str, perm))
            if order not in result_dict:
                result_dict[order] = cursymbol
                cursymbol = cursymbol + 1
                result_dict[order[::-1]] = total_symbols - cursymbol
        result = []
        for v in range(total_symbols):
            for symbol, value in result_dict.items():
                if value == v:
                    result += [symbol]
        return result

    def _symb_python_optimized(
        self, data: np.ndarray, kernel: int, tau: int
    ) -> tuple:
        """Compute symbolic transform following NICE logic but optimized."""
        symbols = self._define_symbols(kernel)
        dims = data.shape

        signal_sym_shape = list(dims)
        signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)

        if signal_sym_shape[1] <= 0:
            # Return empty arrays for too short signals
            signal_sym = (
                np.array([]).reshape((dims[0], 0, dims[2])).astype(np.int32)
            )
            count = np.zeros((dims[0], len(symbols), dims[2]))
            return signal_sym, count

        signal_sym = np.zeros(signal_sym_shape, np.int32)

        count_shape = list(dims)
        count_shape[1] = len(symbols)
        count = np.zeros(count_shape, np.int32)

        # Create a dict for fast lookup
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}

        for k in range(signal_sym_shape[1]):
            subsamples = range(k, k + kernel * tau, tau)
            ind = np.argsort(data[:, subsamples], 1)

            # Process each channel and epoch
            for ch in range(data.shape[0]):
                for ep in range(data.shape[2]):
                    symbol_str = "".join(map(str, ind[ch, :, ep]))
                    signal_sym[ch, k, ep] = symbol_to_idx[symbol_str]

        count = np.double(
            np.apply_along_axis(
                lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym
            )
        )

        return signal_sym, (count / signal_sym_shape[1])

    def _get_weights_matrix(self, nsym: int) -> np.ndarray:
        """Get weights matrix (NICE implementation)."""
        wts = np.ones((nsym, nsym))
        np.fill_diagonal(wts, 0)
        wts = np.fliplr(wts)
        np.fill_diagonal(wts, 0)
        wts = np.fliplr(wts)
        return wts

    def _wsmi_computation(
        self, data_sym: np.ndarray, counts: np.ndarray, wts_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute wSMI or SMI from symbolic data (following NICE logic)."""
        nchannels, nsamples_after_symb, ntrials = data_sym.shape
        n_unique_symbols = counts.shape[1]

        result = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)

        epsilon = 1e-15
        log_counts = np.log(counts + epsilon)

        for trial_idx in range(ntrials):
            for ch1_idx in range(nchannels):
                for ch2_idx in range(ch1_idx + 1, nchannels):
                    pxy = np.zeros(
                        (n_unique_symbols, n_unique_symbols), dtype=np.double
                    )
                    for sample_idx in range(nsamples_after_symb):
                        sym1 = data_sym[ch1_idx, sample_idx, trial_idx]
                        sym2 = data_sym[ch2_idx, sample_idx, trial_idx]
                        pxy[sym1, sym2] += 1

                    if nsamples_after_symb > 0:
                        pxy /= nsamples_after_symb

                    current_result_val = 0.0

                    # Compute MI terms manually
                    for r_idx in range(n_unique_symbols):
                        for c_idx in range(n_unique_symbols):
                            if pxy[r_idx, c_idx] > epsilon:
                                log_pxy_val = np.log(pxy[r_idx, c_idx])
                                log_px_val = log_counts[
                                    ch1_idx, r_idx, trial_idx
                                ]
                                log_py_val = log_counts[
                                    ch2_idx, c_idx, trial_idx
                                ]

                                mi_term = pxy[r_idx, c_idx] * (
                                    log_pxy_val - log_px_val - log_py_val
                                )

                                if self.weighted:
                                    current_result_val += (
                                        wts_matrix[r_idx, c_idx] * mi_term
                                    )
                                else:
                                    current_result_val += mi_term

                    result[ch1_idx, ch2_idx, trial_idx] = current_result_val

        # Normalize
        if n_unique_symbols > 1:
            norm_factor = np.log(n_unique_symbols)
            if norm_factor > epsilon:
                result /= norm_factor

        return result

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute symbolic mutual information between channels."""
        from mne.utils import _time_mask
        from scipy.signal import butter, filtfilt

        # Get the MNE Raw or Epochs object
        raw_or_epochs = input["data"]

        # Handle both Raw and Epochs objects
        if hasattr(raw_or_epochs, "get_data") and hasattr(
            raw_or_epochs, "events"
        ):
            # It's Epochs
            epochs = raw_or_epochs
            # Crop to time window if specified
            if self.tmin is not None or self.tmax is not None:
                epochs = epochs.crop(tmin=self.tmin, tmax=self.tmax)

            # Get data: (n_epochs, n_channels, n_times)
            data = epochs.get_data()
            sfreq = epochs.info["sfreq"]
            ch_names = epochs.ch_names

            # Reshape for processing: (n_channels, n_times, n_epochs)
            fdata = data.transpose(1, 2, 0)

        else:
            # It's Raw - create epochs from continuous data
            raw = raw_or_epochs
            sfreq = raw.info["sfreq"]
            ch_names = raw.ch_names

            # Create epochs from continuous data if needed
            if self.trial_aggregation_method is not None:
                epochs_data = self._create_epochs_from_continuous(raw)
                # epochs_data shape: (n_epochs, n_channels, n_samples)
                fdata = epochs_data.transpose(
                    1, 2, 0
                )  # (n_channels, n_times, n_epochs)
            else:
                # Single "epoch" from continuous data
                data = raw.get_data()  # Shape: (n_channels, n_times)
                if self.tmin is not None or self.tmax is not None:
                    time_mask = _time_mask(raw.times, self.tmin, self.tmax)
                    data = data[:, time_mask]
                fdata = data[
                    :, :, np.newaxis
                ]  # Shape: (n_channels, n_times, 1)

        # Apply filtering (following NICE approach)
        filter_freq = np.double(sfreq) / self.kernel / self.tau
        b, a = butter(6, 2.0 * filter_freq / np.double(sfreq), "lowpass")

        # Filter each channel
        fdata_filtered = np.zeros_like(fdata)
        for ch in range(fdata.shape[0]):
            for ep in range(fdata.shape[2]):
                fdata_filtered[ch, :, ep] = filtfilt(b, a, fdata[ch, :, ep])

        # Symbolic transformation
        sym, count = self._symb_python_optimized(
            fdata_filtered, self.kernel, self.tau
        )

        if sym.shape[1] == 0:
            # Handle case where time window is too short
            n_channels = len(ch_names)
            smi_matrix = np.zeros((n_channels, n_channels))

            # Set diagonal to 1.0
            np.fill_diagonal(smi_matrix, 1.0)

            # Create column names for matrix storage
            col_names = []
            for i in range(n_channels):
                for j in range(n_channels):
                    col_names.append(f"SMI_{ch_names[i]}_{ch_names[j]}")

            # Flatten matrix for storage
            smi_flat = smi_matrix.flatten().reshape(1, -1)

            return {
                "symbolic_mutual_information": {
                    "data": smi_flat,
                    "col_names": col_names,
                }
            }

        n_unique_symbols = count.shape[1]
        wts = self._get_weights_matrix(n_unique_symbols)

        # Compute wSMI/SMI
        result = self._wsmi_computation(sym, count, wts)
        # result is (n_channels, n_channels, n_epochs)

        # Average across epochs (trials)
        result_averaged = np.mean(result, axis=2)  # (n_channels, n_channels)

        # Fill diagonal with 1.0 (self-connectivity)
        np.fill_diagonal(result_averaged, 1.0)

        # Mirror upper triangle to lower triangle
        n_channels = result_averaged.shape[0]
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                result_averaged[j, i] = result_averaged[i, j]

        # Create column names for matrix storage
        col_names = []
        for i in range(n_channels):
            for j in range(n_channels):
                col_names.append(f"SMI_{ch_names[i]}_{ch_names[j]}")

        # Flatten matrix for storage
        smi_flat = result_averaged.flatten().reshape(1, -1)

        return {
            "symbolic_mutual_information": {
                "data": smi_flat,
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
                    :, start_sample:
                ]
                # Pad with zeros or repeat last sample
                epochs_data[epoch_idx, :, available_samples:] = data[:, -1:]

        return epochs_data
