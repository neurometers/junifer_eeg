"""Base symbolic mutual information computation with caching."""

import math
from itertools import permutations
from typing import ClassVar

import numba
import numpy as np
from junifer.utils import logger

from ...utils.singleton import Singleton

__all__ = [
    "SymbolicMutualInformationBase",
    "_define_symbols",
    "_get_weights_matrix",
    "_symb_python_optimized",
    "_wsmi_python_jitted",
]


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


class SymbolicMutualInformationBase(metaclass=Singleton):
    """Base symbolic mutual information computation with caching.

    Singleton class that computes symbolic mutual information with
    internal caching for efficient reuse across multiple markers.

    The caching uses (data_id, params) as key - valid while the
    data array exists in memory. This follows Fede's pattern but
    adapted for in-memory EEG data instead of file paths.

    This class handles the core SMI/WSMI algorithm including:
    - Symbolic transformation of filtered data
    - Mutual information computation between channel pairs
    - Optional weighting for weighted SMI

    """

    _DEPENDENCIES: ClassVar = {"numpy", "numba"}

    # Internal cache: {(data_id, kernel, tau, weighted): connectivity_matrix}
    _cache: ClassVar[dict] = {}

    def __del__(self) -> None:  # pragma: no cover
        """Terminate and clear cache."""
        logger.debug("Clearing cache for SMI computation")
        SymbolicMutualInformationBase._cache.clear()

    def compute(
        self,
        filtered_data: np.ndarray,
        kernel: int,
        tau: int,
        weighted: bool,
    ) -> np.ndarray:
        """Compute symbolic mutual information with caching.

        This method follows Fede's pattern (like AFNIReHo) but adapted for
        in-memory EEG data. The cache key is (data_id, params) - valid
        while the data array exists in memory.

        Parameters
        ----------
        filtered_data : np.ndarray
            Filtered and time-masked data with shape (n_channels, n_times, n_epochs).
        kernel : int
            Length of ordinal patterns.
        tau : int
            Time delay for ordinal patterns.
        weighted : bool
            Whether to compute weighted SMI (True) or unweighted SMI (False).

        Returns
        -------
        connectivity_matrix : np.ndarray
            Connectivity matrix with shape (n_channels, n_channels, n_epochs).
            Values represent SMI/WSMI between channel pairs.

        """
        # Build cache key from hashable parameters
        cache_key = (id(filtered_data), kernel, tau, weighted)

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"SMI cache hit: data_id={id(filtered_data)}")
            return self._cache[cache_key]

        logger.debug(
            f"SMI cache miss: computing kernel={kernel}, tau={tau}, weighted={weighted}"
        )

        # Symbolic transformation
        sym, count = _symb_python_optimized(filtered_data, kernel, tau)
        n_unique_symbols = count.shape[1]

        # Get weights matrix
        wts = _get_weights_matrix(n_unique_symbols)

        # Compute SMI/WSMI
        result = _wsmi_python_jitted(sym, count, wts, weighted)

        # Symmetrize the upper triangular result
        result = result + result.transpose(1, 0, 2)

        # Store in cache
        self._cache[cache_key] = result

        return result
