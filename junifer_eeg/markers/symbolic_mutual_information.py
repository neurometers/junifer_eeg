"""Symbolic mutual information marker for EEG analysis."""

import math
from itertools import permutations
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import apply_roi_trial_aggregation


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
            "symbolicmutualinformation": "matrix",
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
        self,
        data: np.ndarray,
        kernel: int,
        tau: int,
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
                lambda x: np.bincount(x, minlength=len(symbols)),
                1,
                signal_sym,
            ),
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
        self,
        data_sym: np.ndarray,
        counts: np.ndarray,
        wts_matrix: np.ndarray,
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
                        (n_unique_symbols, n_unique_symbols),
                        dtype=np.double,
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
                                    ch1_idx,
                                    r_idx,
                                    trial_idx,
                                ]
                                log_py_val = log_counts[
                                    ch2_idx,
                                    c_idx,
                                    trial_idx,
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

    def _compute_smi_matrix(
        self,
        data_sym: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """Compute SMI matrix from symbolic data."""
        nchannels, nsamples_after_symb, ntrials = data_sym.shape
        n_unique_symbols = counts.shape[1]

        result = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)

        epsilon = 1e-15
        log_counts = np.log(counts + epsilon)

        for trial_idx in range(ntrials):
            for ch1_idx in range(nchannels):
                for ch2_idx in range(ch1_idx + 1, nchannels):
                    pxy = np.zeros(
                        (n_unique_symbols, n_unique_symbols),
                        dtype=np.double,
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
                                    ch1_idx,
                                    r_idx,
                                    trial_idx,
                                ]
                                log_py_val = log_counts[
                                    ch2_idx,
                                    c_idx,
                                    trial_idx,
                                ]

                                mi_term = pxy[r_idx, c_idx] * (
                                    log_pxy_val - log_px_val - log_py_val
                                )

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
        from scipy.signal import butter, filtfilt

        # Get the MNE Epochs object
        epochs = input["data"]

        # Crop to time window if specified
        if self.tmin is not None or self.tmax is not None:
            epochs = epochs.copy().crop(tmin=self.tmin, tmax=self.tmax)

        # Get data: (n_epochs, n_channels, n_times)
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]

        # Reshape for processing: (n_channels, n_times, n_epochs)
        fdata = data.transpose(1, 2, 0)

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
            fdata_filtered,
            self.kernel,
            self.tau,
        )

        # Compute Symbolic Mutual Information
        smi_result = self._compute_smi_matrix(sym, count)

        # Average across trials (epochs): (n_channels, n_channels, n_trials) -> (n_channels, n_channels)
        smi_matrix = np.mean(smi_result, axis=2)

        # Fill diagonal with 1.0 (self-connectivity)
        np.fill_diagonal(smi_matrix, 1.0)

        # Mirror upper triangle to lower triangle (make symmetric)
        n_channels = smi_matrix.shape[0]
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                smi_matrix[j, i] = smi_matrix[i, j]

        # For connectivity matrices, we use all channel pairs
        # ROI aggregation for connectivity is complex and not implemented properly yet
        roi_data = {"all_pairs": smi_matrix.flatten().reshape(1, -1).T}

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="symbolic_mutual_information",
        )

        return results
