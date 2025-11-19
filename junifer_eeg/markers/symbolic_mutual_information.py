"""Symbolic mutual information marker for EEG analysis."""

import math
from itertools import permutations
from typing import Any, ClassVar, Dict, List, Optional, Union

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
            "symbolicmutualinformation": "timeseries",  # 2D: (epochs, channel_pairs)
        },
    }

    def __init__(
        self,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        kernel: int = 3,
        tau: int = 8,
        weighted: bool = True,
        csd: bool = True,
        anti_aliasing: bool = True,
        rois: Union[List[str], List[int], None] = None,
        connectivity_aggregation_method: Optional[str | List[str]] = None,
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
        epoch_length: Optional[float] = None,
        overlap: float = 0.0,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        filter_order: int = 6,
        on: Optional[str | List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SymbolicMutualInformation marker.

        Parameters
        ----------
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1', 'E224') OR semantic ROI (e.g., 'scalp')

            Examples:
            - list(range(224)) - NICE scalp ROI via indices (EGI/256)
            - ['E1', 'E2', ..., 'E224'] - NICE scalp ROI via names
            - ['scalp'] - Semantic ROI (expands to all scalp channels)

            **CRITICAL FOR NICE:** Applied BEFORE connectivity computation.
            Filters BOTH dimensions of the connectivity matrix.
        connectivity_aggregation_method : str, optional
            Method(s) to aggregate across the second connectivity dimension (channels_y).
            Options: 'median', 'mean', 'std', 'trim_mean80', etc.
            For NICE compatibility, use 'median'. If None, no aggregation on channels_y.
        channel_aggregation_method : str, optional
            Method(s) to aggregate across channels dimension.
            For NICE compatibility, use 'mean'. If None, no aggregation on channels.
        trial_aggregation_method : str, optional
            Method(s) to aggregate across epochs dimension.
            For NICE compatibility, use 'trim_mean80'. If None, no aggregation on epochs.
        csd : bool, optional
            Whether to apply Current Source Density (CSD) preprocessing for weighted SMI.
            Only applied when weighted=True. Default: True (matches NICE behavior).
        fmin : float, optional
            Lower frequency bound for band-pass filtering. If None, no filtering is applied.
        fmax : float, optional
            Upper frequency bound for band-pass filtering. If None, no filtering is applied.
        filter_order : int, optional
            Order of the Butterworth filter. Default: 6 (matches NICE implementation).
        """
        self.tmin = tmin
        self.tmax = tmax
        self.kernel = kernel
        self.tau = tau
        self.weighted = weighted
        self.csd = csd
        self.anti_aliasing = anti_aliasing
        self.rois = rois
        self.connectivity_aggregation_method = connectivity_aggregation_method
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.filter_order = filter_order

        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 2D/3D tensor data (0-1 aggregations)
        - 'vector': 1D array (2 aggregations applied)
        - 'scalar_table': scalar value (all 3 aggregations applied)

        WSMI dimensionality:
        - No agg: (epochs, channels, channels) → 3D → timeseries
        - Only connectivity_agg: (epochs, channels) → 2D → timeseries
        - Only trial_agg: (channels, channels) → 2D → timeseries
        - Only channel_agg: (epochs, channels) → 2D → timeseries
        - Two aggs: 1D → vector
        - All three aggs: scalar → scalar_table
        """
        # Count how many aggregation methods are applied
        agg_count = sum(
            [
                self.connectivity_aggregation_method is not None,
                self.channel_aggregation_method is not None,
                self.trial_aggregation_method is not None,
            ]
        )

        # 0 or 1 aggregation → 2D or 3D tensor → use timeseries
        if agg_count <= 1:
            return "timeseries"

        # All 3 aggregations → scalar → use scalar_table
        if agg_count == 3:
            return "scalar_table"

        # 2 aggregations → 1D array → use vector
        return "vector"

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Symbolic Mutual Information.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed SMI connectivity matrix.
        """
        from .utils import filter_to_eeg_channels

        # Get the MNE data object
        data_obj = input["data"]

        # CRITICAL FIX: Filter to only EEG channels (E1-E256), excluding D/DI auxiliary channels
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Import required modules
        from mne.utils import _time_mask
        from scipy.signal import butter, filtfilt

        # Get the filtered MNE Epochs object
        epochs = data_obj

        # Check if epochs object is empty
        if len(data_obj) == 0:
            # Return empty results for empty epochs
            return {
                "symbolicmutualinformation": {
                    "data": np.array([[]]),
                    "col_names": [],
                }
            }

        # Validate parameters (like NICE)
        if self.kernel <= 1:
            raise ValueError(
                f"kernel (pattern length) must be > 1, got {self.kernel}"
            )
        if self.tau <= 0:
            raise ValueError(f"tau (delay) must be > 0, got {self.tau}")

        sfreq = epochs.info["sfreq"]

        # Apply CSD preprocessing BEFORE ROI filtering (needs all channels with montage)
        if self.csd:
            from mne import pick_types
            from mne.preprocessing import compute_current_source_density

            # Apply CSD to EEG channels if available (matching NICE behavior)
            if (
                "eeg" in epochs
                and pick_types(epochs.info, meg=False, eeg=True).size > 0
            ):
                epochs_temp = epochs.copy()
                if epochs_temp.info["bads"]:
                    # Interpolate bad channels for CSD computation (like NICE)
                    epochs_temp.interpolate_bads(reset_bads=True)

                # Compute CSD with correct parameters
                epochs_csd = compute_current_source_density(
                    epochs_temp, lambda2=1e-5
                )

                # Check if CSD actually produced CSD channels
                csd_picks = pick_types(epochs_csd.info, csd=True)
                if len(csd_picks) > 0:
                    epochs = epochs_csd
                else:
                    # CSD didn't work, use original EEG data
                    pass  # epochs remains unchanged

        # Pick data channels for connectivity computation (matching NICE exactly)
        # MEG, EEG, CSD, SEEG, ECoG are typical data channels. Exclude bads.
        from mne import pick_types

        picks = pick_types(
            epochs.info,
            meg=True,
            eeg=True,
            csd=True,
            seeg=True,
            ecog=True,
            ref_meg=False,
            exclude="bads",
        )

        if len(picks) == 0:
            raise ValueError(
                "No suitable channels (MEG, EEG, CSD, SEEG, ECoG) "
                "found after picking logic. Check channel types and 'bads'."
            )

        data_for_comp = epochs.get_data(picks=picks)
        picked_ch_names = [epochs.ch_names[i] for i in picks]
        n_epochs, n_channels_picked, n_times_epoch = data_for_comp.shape

        # Apply ROI filtering AFTER CSD (equipment from data metadata)
        if self.rois is not None:
            from .utils import get_data_for_rois

            # Get equipment from data (set by data reader)
            description = epochs.info.get("description") or ""
            if "equipment=" in description:
                equipment = description.replace("equipment=", "")
            else:
                raise ValueError(
                    f"Equipment metadata not found in data. "
                    f"Data reader should set it. Got: {description}"
                )

            # Transpose to (n_channels, n_epochs, n_times)
            data_transposed = data_for_comp.transpose(1, 0, 2)

            # Get ROI-filtered data
            roi_data_dict = get_data_for_rois(
                data_transposed,
                picked_ch_names,
                self.rois,
                equipment,
            )

            # Extract the filtered data (returns {"selected_channels": data})
            if "selected_channels" in roi_data_dict:
                data_filtered = roi_data_dict["selected_channels"]
                # Transpose back to (n_epochs, n_channels, n_times)
                data_for_comp = data_filtered.transpose(1, 0, 2)
                n_channels_picked = data_for_comp.shape[1]
                # Update channel names
                picked_ch_names = [f"ch_{i}" for i in range(n_channels_picked)]

        # Check for insufficient channels for connectivity computation
        if n_channels_picked < 2:
            raise ValueError(
                f"At least 2 channels are required for connectivity computation, "
                f"but only {n_channels_picked} channels available after excluding "
                f"bad channels and applying ROI filtering."
            )

        # Apply filtering using NICE's EXACT approach
        # Calculate filter frequency using NICE's formula: sfreq / kernel / tau
        filter_freq = (
            np.double(sfreq) / self.kernel / self.tau
        )  # Use np.double like NICE

        # Apply filtering if not disabled by custom fmin/fmax
        if self.fmin is None and self.fmax is None:
            # Concatenate epochs, filter, then split back
            b, a = butter(
                self.filter_order,
                2.0 * filter_freq / np.double(sfreq),
                "lowpass",
            )
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
        elif self.fmin is not None and self.fmax is not None:
            # Custom frequency band filtering - apply to already picked data
            filter_freq = (self.fmin + self.fmax) / 2.0
            # Concatenate epochs, filter, then split back
            b, a = butter(
                self.filter_order,
                2.0 * filter_freq / np.double(sfreq),
                "lowpass",
            )
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
            # No filtering - just transpose to match NICE format
            fdata = np.transpose(
                data_for_comp, [1, 2, 0]
            )  # (channels, times, epochs)

        # Use all connections (upper triangular matrix) like NICE
        indices_use = np.triu_indices(n_channels_picked, k=1)

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
        # Note: only fills upper triangle, result[i,j] where i < j
        # This ensures connectivity matrix is symmetric: result[i,j] = result[j,i]
        result = result + result.transpose(1, 0, 2)

        # Check if we should apply configurable aggregation pipeline
        if (
            self.connectivity_aggregation_method is not None
            or self.channel_aggregation_method is not None
            or self.trial_aggregation_method is not None
        ):
            # Apply configurable reduction pipeline on symmetrized data
            # result shape: (n_channels, n_channels, n_epochs)
            results = self._apply_configurable_aggregation(
                result, picked_ch_names
            )
        else:
            # Return per-epoch connectivity matrices without any aggregation
            result_epoched = result.transpose(
                2, 0, 1
            )  # (n_epochs, n_channels, n_channels)

            # Return per-epoch results - each epoch gets its own row
            # Use lower triangular matrix like(excluding diagonal)
            epoch_data = []

            # Generate column names for upper triangular pairs using utility function
            from .utils import create_connectivity_pair_column_names

            col_names = create_connectivity_pair_column_names(picked_ch_names)

            # Extract upper triangular values for each epoch
            for epoch_idx in range(n_epochs):
                epoch_matrix = result_epoched[
                    epoch_idx
                ]  # (n_channels, n_channels)
                # Extract upper triangular values (excluding diagonal) - only computes these
                upper_tri_values = epoch_matrix[indices_use]
                epoch_data.append(upper_tri_values)

            # Stack all epochs: (n_epochs, n_channel_pairs)
            epoch_data_array = np.array(epoch_data)

            # Return results with proper structure - per-epoch data
            results = {
                "symbolicmutualinformation": {
                    "data": epoch_data_array,
                    "col_names": col_names,
                }
            }

        return results

    def _apply_configurable_aggregation(
        self, connectivity_tensor, picked_ch_names
    ):
        """Apply configurable aggregation pipeline to connectivity tensor.

        Applies aggregation in the order specified by NICE:
        1. connectivity_aggregation_method across channels_y dimension (axis=1)
        2. channel_aggregation_method across channels dimension (axis=0)
        3. trial_aggregation_method across epochs dimension

        Parameters
        ----------
        connectivity_tensor : np.ndarray
            Shape (n_channels, n_channels, n_epochs)
        picked_ch_names : list
            Channel names

        Returns
        -------
        dict
            Results dictionary with aggregated data
        """
        from .utils import aggregate_data

        # Start with raw connectivity tensor: (n_channels, n_channels, n_epochs)
        current_data = connectivity_tensor

        # Step 1: Aggregate across channels_y (second connectivity dimension, axis=1)
        if self.connectivity_aggregation_method is not None:
            current_data = aggregate_data(
                current_data, self.connectivity_aggregation_method, axis=1
            )
            # After first aggregation: (n_channels, n_epochs)

        # Step 2: Aggregate across channels (first dimension, axis=0)
        if self.channel_aggregation_method is not None:
            current_data = aggregate_data(
                current_data, self.channel_aggregation_method, axis=0
            )

        else:
            # CRITICAL FIX: If roi_aggregation is skipped, after connectivity aggregation
            # current_data is (n_channels, n_epochs). We need to transpose so epochs
            # are on axis=0 for trial aggregation to work correctly on epochs (not channels)
            if current_data.ndim == 2:
                current_data = current_data.T  # Now (n_epochs, n_channels)

        # Step 3: Aggregate across epochs (axis depends on current shape)
        if self.trial_aggregation_method is not None:
            if current_data.ndim > 0:
                # Determine correct axis based on shape:
                # - 3D (n_channels, n_channels, n_epochs): aggregate axis=2 (epochs)
                # - 2D (n_epochs, n_channels): aggregate axis=0 (epochs)
                # - 1D (n_epochs,): aggregate axis=0 (epochs)
                if current_data.ndim == 3:
                    # Still have 3D connectivity tensor - aggregate across epochs (axis=2)
                    epoch_axis = 2
                else:
                    # Already reduced to 2D or 1D - epochs are on axis=0
                    epoch_axis = 0

                current_data = aggregate_data(
                    current_data,
                    self.trial_aggregation_method,
                    axis=epoch_axis,
                )
                # After aggregation:
                # - Was 3D: now (n_channels, n_channels)
                # - Was 2D: now (n_channels,)
                # - Was 1D: now scalar

        # Format output based on result shape - preserve tensor structure
        if np.isscalar(current_data):
            # Scalar result - fully aggregated
            results = {
                "symbolicmutualinformation": {
                    "data": float(current_data),  # Return scalar directly
                }
            }
        elif hasattr(current_data, "size") and current_data.size == 1:
            # Single element array - convert to scalar
            results = {
                "symbolicmutualinformation": {
                    "data": float(current_data.item()),  # Convert to scalar
                }
            }
        else:
            # Tensor result - return as is without flattening
            # Preserve the natural tensor structure based on aggregation state
            results = {
                "symbolicmutualinformation": {
                    "data": current_data,  # Preserve tensor dimensions
                }
            }

        return results
