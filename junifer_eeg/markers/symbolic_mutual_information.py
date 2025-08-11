"""Symbolic mutual information marker for EEG analysis."""

import math
from itertools import permutations
from typing import Any, ClassVar, Dict, List, Optional

import numba
import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker


def _define_symbols(kernel):
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


def _symb_python_optimized(data, kernel, tau):
    """Compute symbolic transform using original NICE logic."""
    symbols = _define_symbols(kernel)
    dims = data.shape

    signal_sym_shape = list(dims)
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)

    if signal_sym_shape[1] <= 0:
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


def _get_weights_matrix(nsym):
    """Get weights matrix (NICE implementation)."""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


@numba.njit(parallel=True)
def _wsmi_python_jitted(data_sym, counts, wts_matrix, weighted=True):
    """Compute raw wSMI or SMI from symbolic data (Numba-jitted) - EXACT NICE implementation."""
    nchannels, nsamples_after_symb, ntrials = data_sym.shape
    n_unique_symbols = counts.shape[1]

    result = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)

    epsilon = 1e-15
    log_counts = np.log(counts + epsilon)

    for trial_idx in numba.prange(ntrials):
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

                # Compute MI terms manually to avoid broadcasting issues in Numba
                for r_idx in range(n_unique_symbols):
                    for c_idx in range(n_unique_symbols):
                        if pxy[r_idx, c_idx] > epsilon:
                            log_pxy_val = np.log(pxy[r_idx, c_idx])
                            log_px_val = log_counts[ch1_idx, r_idx, trial_idx]
                            log_py_val = log_counts[ch2_idx, c_idx, trial_idx]

                            mi_term = pxy[r_idx, c_idx] * (
                                log_pxy_val - log_px_val - log_py_val
                            )

                            if weighted:
                                current_result_val += (
                                    wts_matrix[r_idx, c_idx] * mi_term
                                )
                            else:
                                current_result_val += mi_term

                result[ch1_idx, ch2_idx, trial_idx] = current_result_val

    # Normalize exactly like NICE
    if n_unique_symbols > 1:
        norm_factor = np.log(n_unique_symbols)
        if norm_factor > epsilon:
            result /= norm_factor

    return result


def check_indices(indices):
    """Simple implementation of check_indices."""
    if indices is None:
        return None
    return (np.array(indices[0]), np.array(indices[1]))


@register_marker
class SymbolicMutualInformation(BaseMarker):
    """Symbolic mutual information marker for connectivity analysis.

    This marker computes symbolic mutual information between EEG channels,
    providing a measure of nonlinear coupling between brain regions.
    Based on the NICE package implementation with ordinal pattern transformation.

    This implementation uses the exact same computational approach as the original
    NICE wsmi function to ensure numerical equivalence.
    """

    _DEPENDENCIES: ClassVar = {
        "mne",
        "numpy",
        "scipy",
        "numba",
        "mne-connectivity",
    }

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "symbolicmutualinformation": "vector",  # Can be vector (per-channel) or matrix (full connectivity)
        },
    }

    def __init__(
        self,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        kernel: int = 3,
        tau: int = 8,
        weighted: bool = True,
        anti_aliasing: bool = True,
        average: bool = True,
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
        self.anti_aliasing = anti_aliasing
        self.average = average
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        self.epoch_length = epoch_length
        self.overlap = overlap

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute symbolic mutual information using EXACT NICE implementation."""
        import warnings

        from mne.utils import _time_mask
        from scipy.signal import butter, filtfilt

        # Get the MNE Epochs object
        epochs = input["data"]

        # Extract metadata from input to preserve element information
        meta = input.get("meta", None)

        # Check if epochs object is empty
        if len(epochs) == 0:
            # Return empty results for empty epochs
            ch_names = epochs.ch_names
            if self.rois is not None:
                roi_data = {
                    roi: np.array([]).reshape(0, 0) for roi in self.rois
                }
            else:
                roi_data = {ch: np.array([]).reshape(0, 0) for ch in ch_names}

            from .utils import apply_roi_trial_aggregation

            return apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="symbolicmutualinformation",
                meta=meta,
            )

        # Validate parameters (like NICE)
        if self.kernel <= 1:
            raise ValueError(
                f"kernel (pattern length) must be > 1, got {self.kernel}"
            )
        if self.tau <= 0:
            raise ValueError(f"tau (delay) must be > 0, got {self.tau}")

        sfreq = epochs.info["sfreq"]

        # Handle bad channels manually (consistent with NICE)
        data_for_comp = epochs.get_data()
        picked_ch_names = [
            ch for ch in epochs.ch_names if ch not in epochs.info["bads"]
        ]

        # Check if we have any good channels
        if len(picked_ch_names) == 0:
            raise ValueError(
                "No good channels found. Check that channels are not all marked as bad."
            )

        # Remove bad channel data if any bad channels exist
        if epochs.info["bads"]:
            bad_indices = [
                i
                for i, ch in enumerate(epochs.ch_names)
                if ch in epochs.info["bads"]
            ]
            good_indices = [
                i for i in range(len(epochs.ch_names)) if i not in bad_indices
            ]
            data_for_comp = data_for_comp[:, good_indices, :]

        n_epochs, n_channels_picked, n_times_epoch = data_for_comp.shape

        # Check for insufficient channels for connectivity computation
        if n_channels_picked < 2:
            raise ValueError(
                f"At least 2 channels are required for connectivity computation, "
                f"but only {n_channels_picked} channels available after excluding "
                f"bad channels."
            )

        # Use all connections (lower triangular matrix) like NICE
        indices_use = np.tril_indices(n_channels_picked, k=-1)

        # Anti-aliasing filtering (EXACT NICE implementation)
        if self.anti_aliasing:
            anti_alias_freq = np.double(sfreq) / self.kernel / self.tau
            nyquist_freq = sfreq / 2.0

            if anti_alias_freq >= nyquist_freq * 0.99:
                # Anti-aliasing frequency too close to Nyquist - skip filtering
                fdata = data_for_comp.transpose(1, 2, 0)
            else:
                # Design and apply low-pass filter
                normalized_freq = 2.0 * anti_alias_freq / np.double(sfreq)
                b, a = butter(6, normalized_freq, "lowpass")
                data_concatenated = np.hstack(
                    data_for_comp
                )  # Concatenate epochs horizontally

                # Filter the concatenated data
                fdata_concatenated = filtfilt(b, a, data_concatenated)

                # Split back into epochs and transpose to match original format
                fdata = np.transpose(
                    np.array(np.split(fdata_concatenated, n_epochs, axis=1)),
                    [1, 2, 0],
                )
        else:
            # Skip filtering - use data as-is but warn about potential issues
            effective_sfreq = sfreq / self.tau
            warnings.warn(
                f"Anti-aliasing disabled. Effective sampling rate for symbolic "
                f"transformation is {effective_sfreq:.1f} Hz (sfreq/tau = {sfreq}/{self.tau}). "
                f"Ensure your data is appropriately filtered to prevent aliasing artifacts",
                UserWarning,
                stacklevel=2,
            )
            # Transpose to match filtered data format: (n_channels, n_times, n_epochs)
            fdata = data_for_comp.transpose(1, 2, 0)

        # Time masking (EXACT NICE implementation)
        time_mask = _time_mask(epochs.times, self.tmin, self.tmax)
        fdata_masked = fdata[:, time_mask, :]

        # Check if time masking resulted in too few samples for symbolization
        min_samples_needed_for_one_symbol = self.tau * (self.kernel - 1) + 1
        if fdata_masked.shape[1] < min_samples_needed_for_one_symbol:
            raise ValueError(
                f"After time masking ({self.tmin}-{self.tmax}s), data has "
                f"{fdata_masked.shape[1]} samples per epoch, but at least "
                f"{min_samples_needed_for_one_symbol} are needed for kernel={self.kernel}, "
                f"tau={self.tau}. Adjust tmin/tmax or check epoch length."
            )

        # Symbolic Transformation
        sym, count = _symb_python_optimized(
            fdata_masked, self.kernel, self.tau
        )
        n_unique_symbols = count.shape[1]

        if (
            sym.shape[0] != n_channels_picked
            or sym.shape[2] != n_epochs
            or count.shape[0] != n_channels_picked
            or count.shape[2] != n_epochs
        ):
            raise ValueError(
                f"Symbolic transformation output has unexpected shape. "
                f"Got sym: {sym.shape}, count: {count.shape}. "
                f"Expected channels: {n_channels_picked}, epochs: {n_epochs}."
            )

        wts = _get_weights_matrix(n_unique_symbols)

        # wSMI/SMI Computation (Jitted) - EXACT NICE implementation
        result = _wsmi_python_jitted(sym, count, wts, self.weighted)
        # result is (n_channels_picked, n_channels_picked, n_epochs)

        if self.average:
            # Average across epochs - EXACT NICE behavior
            result_epoched = result.transpose(
                2, 0, 1
            )  # (n_epochs, n_channels, n_channels)

            # Extract connectivity for specified connections only - EXACT NICE behavior
            n_cons = len(indices_use[0])
            result_conn_data = np.zeros(n_cons)
            indices_list = list(zip(indices_use[0], indices_use[1]))

            # Average across epochs first, then extract values
            result_avg = np.mean(
                result_epoched, axis=0
            )  # (n_channels, n_channels)

            for conn_idx, (i, j) in enumerate(indices_list):
                # The computation fills upper triangle, but indices_list has lower triangle indices
                # So we need to swap i,j to get the computed values
                result_conn_data[conn_idx] = result_avg[j, i]

            # Check if we should use ROI aggregation for topographic visualization
            if (
                self.roi_aggregation_method is not None
                or self.trial_aggregation_method is not None
            ):
                # Create full connectivity matrix for aggregation
                full_matrix = np.zeros((n_channels_picked, n_channels_picked))
                # Keep diagonal at 0.0 to match NICE implementation

                # Fill the lower triangle with computed values
                for conn_idx, (i, j) in enumerate(indices_list):
                    full_matrix[i, j] = result_conn_data[conn_idx]
                    full_matrix[j, i] = result_conn_data[
                        conn_idx
                    ]  # Make symmetric

                # Use ROI aggregation to get per-channel values for topographic plotting
                results = self._aggregate_connectivity_for_rois(
                    full_matrix, picked_ch_names
                )
            else:
                # Return full connectivity matrix for compatibility
                full_matrix = np.zeros((n_channels_picked, n_channels_picked))
                # Keep diagonal at 0.0 to match NICE implementation

                # Fill the lower triangle with computed values
                for conn_idx, (i, j) in enumerate(indices_list):
                    full_matrix[i, j] = result_conn_data[conn_idx]
                    full_matrix[j, i] = result_conn_data[
                        conn_idx
                    ]  # Make symmetric

                # Return results with proper structure
                results = {
                    "symbolicmutualinformation": {
                        "data": full_matrix.flatten().reshape(1, -1),
                        "col_names": [
                            f"{picked_ch_names[i]}-{picked_ch_names[j]}"
                            for i in range(n_channels_picked)
                            for j in range(n_channels_picked)
                        ],
                    }
                }

        else:
            # Return epoch-wise connectivity
            result_epoched = result.transpose(2, 0, 1)
            n_cons = len(indices_use[0])
            result_conn_data = np.zeros((n_epochs, n_cons))
            indices_list = list(zip(indices_use[0], indices_use[1]))

            for epoch_idx in range(n_epochs):
                for conn_idx, (i, j) in enumerate(indices_list):
                    # Same fix for epoch-wise case - swap i,j
                    result_conn_data[epoch_idx, conn_idx] = result_epoched[
                        epoch_idx, j, i
                    ]

            # For non-averaged case, return trial-averaged matrix for junifer compatibility
            result_avg = np.mean(result_conn_data, axis=0)

            # Check if we should use ROI aggregation for topographic visualization
            if (
                self.roi_aggregation_method is not None
                or self.trial_aggregation_method is not None
            ):
                # Create full connectivity matrix for aggregation
                full_matrix = np.zeros((n_channels_picked, n_channels_picked))
                # Keep diagonal at 0.0 to match NICE implementation

                for conn_idx, (i, j) in enumerate(indices_list):
                    full_matrix[i, j] = result_avg[conn_idx]
                    full_matrix[j, i] = result_avg[conn_idx]

                # Use ROI aggregation to get per-channel values for topographic plotting
                results = self._aggregate_connectivity_for_rois(
                    full_matrix, picked_ch_names
                )
            else:
                full_matrix = np.zeros((n_channels_picked, n_channels_picked))
                # Keep diagonal at 0.0 to match NICE implementation

                for conn_idx, (i, j) in enumerate(indices_list):
                    # indices_list has (i,j) where i>j, but we want to fill matrix[i,j]
                    full_matrix[i, j] = result_avg[conn_idx]
                    full_matrix[j, i] = result_avg[conn_idx]

                results = {
                    "symbolicmutualinformation": {
                        "data": full_matrix.flatten().reshape(1, -1),
                        "col_names": [
                            f"{picked_ch_names[i]}-{picked_ch_names[j]}"
                            for i in range(n_channels_picked)
                            for j in range(n_channels_picked)
                        ],
                    }
                }

        return results

    def _aggregate_connectivity_for_rois(
        self, connectivity_matrix, picked_ch_names
    ):
        """Aggregate connectivity matrix into per-channel values for topographic plotting."""
        from .utils import apply_roi_trial_aggregation, get_data_for_rois

        # For connectivity, we need to aggregate each channel's connections
        # This creates a per-channel summary suitable for topographic plotting
        n_channels = connectivity_matrix.shape[0]

        # Calculate per-channel connectivity strength (excluding diagonal)
        per_channel_values = np.zeros(n_channels)

        for i in range(n_channels):
            # Get all connections for channel i (excluding self-connection)
            connections = np.concatenate(
                [
                    connectivity_matrix[
                        i, :i
                    ],  # connections to channels 0 to i-1
                    connectivity_matrix[
                        i, i + 1 :
                    ],  # connections to channels i+1 to end
                ]
            )
            # Use mean absolute connectivity as the per-channel measure
            per_channel_values[i] = (
                np.mean(np.abs(connections)) if len(connections) > 0 else 0.0
            )

        # Now use standard ROI aggregation
        if self.rois is not None:
            # Get data for specified ROIs
            roi_data = get_data_for_rois(
                per_channel_values.reshape(1, -1),  # Shape: (1, n_channels)
                picked_ch_names,
                self.rois,
            )
        else:
            # Use all channels as individual ROIs
            roi_data = {
                ch: per_channel_values[i : i + 1].reshape(1, -1)
                for i, ch in enumerate(picked_ch_names)
            }

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="symbolicmutualinformation",
        )

        return results
