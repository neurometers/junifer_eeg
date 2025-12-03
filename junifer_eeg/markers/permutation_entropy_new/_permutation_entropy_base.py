"""Base permutation entropy computation with caching."""

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


# Try to import numba for acceleration
try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


__all__ = ["PermutationEntropyBase", "_pe_numpy"]


def _pe_numpy(
    signal: np.ndarray, kernel: int, tau: int, fact: np.ndarray
) -> float:
    """Vectorized NumPy implementation of permutation entropy."""
    L = signal.size - tau * (kernel - 1)
    if L <= 0:
        return np.nan

    base = np.arange(L)[:, None]
    offs = (np.arange(kernel) * tau)[None, :]
    idx = base + offs
    X = signal[idx]

    P = np.argsort(X, axis=1)

    Lc = np.zeros((L, kernel), dtype=np.int64)
    for j in range(kernel - 1):
        pj = P[:, j][:, None]
        right = P[:, j + 1 :]
        Lc[:, j] = np.sum(right < pj, axis=1)

    weights = fact[kernel - 1 : 0 : -1]
    ranks = (Lc[:, :-1] * weights).sum(axis=1)

    n_symbols = fact[kernel]
    count = np.bincount(ranks, minlength=n_symbols).astype(np.float64)
    count /= L

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(count > 0, np.log(count), 0.0)
        pe = -np.sum(count * logp)

    return pe / np.log(n_symbols)


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=False)
    def _pe_numba(
        signal: np.ndarray, kernel: int, tau: int, fact: np.ndarray
    ) -> float:
        """Numba-accelerated permutation entropy."""
        L = signal.size - tau * (kernel - 1)
        if L <= 0:
            return np.nan

        n_symbols = fact[kernel]
        counts = np.zeros(n_symbols, dtype=np.int64)

        for start in range(L):
            vals = np.empty(kernel, dtype=np.float64)
            for j in range(kernel):
                vals[j] = signal[start + j * tau]

            idxs = np.empty(kernel, dtype=np.int64)
            for j in range(kernel):
                idxs[j] = j

            for i in range(1, kernel):
                key_idx = idxs[i]
                key_val = vals[key_idx]
                j = i - 1
                while j >= 0 and vals[idxs[j]] > key_val:
                    idxs[j + 1] = idxs[j]
                    j -= 1
                idxs[j + 1] = key_idx

            rank = 0
            for j in range(kernel - 1):
                cj = 0
                pj = idxs[j]
                for i in range(j + 1, kernel):
                    if idxs[i] < pj:
                        cj += 1
                rank += cj * fact[kernel - 1 - j]

            counts[rank] += 1

        probs = counts.astype(np.float64) / L
        pe = 0.0
        for p in probs:
            if p > 0.0:
                pe -= p * np.log(p)

        return pe / np.log(n_symbols)


class PermutationEntropyBase(metaclass=Singleton):
    """Base permutation entropy computation with caching.

    Singleton class that computes permutation entropy with LRU caching
    for efficient reuse across multiple markers.

    """

    _DEPENDENCIES: ClassVar = {"numpy"}

    def __init__(self) -> None:
        """Initialize PE base."""
        self._use_numba = _HAVE_NUMBA

    def __del__(self) -> None:  # pragma: no cover
        """Terminate and clear cache."""
        logger.debug("Clearing cache for PE computation")
        if hasattr(self, "compute"):
            self.compute.cache_clear()

    def _compute_pe(
        self,
        epochs: "mne.Epochs",
        kernel: int,
        tau: int,
        fmin: float | None,
        fmax: float | None,
        filter_order: int,
        tmin: float,
        tmax: float,
    ) -> np.ndarray:
        """Compute permutation entropy for epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object.
        kernel : int
            Length of ordinal patterns.
        tau : int
            Time delay for patterns.
        fmin : float or None
            Minimum frequency for bandpass filter. If None, uses adaptive lowpass.
        fmax : float or None
            Maximum frequency for bandpass filter. If None, uses adaptive lowpass.
        filter_order : int
            Filter order.
        tmin : float
            Start time for analysis.
        tmax : float
            End time for analysis.

        Returns
        -------
        pe_values : np.ndarray
            PE values with shape (n_epochs, n_channels).

        """
        from mne.utils import _time_mask
        from scipy.signal import butter, filtfilt

        epochs_data = epochs.get_data()
        n_epochs, n_channels, n_samples = epochs_data.shape
        sfreq = epochs.info["sfreq"]

        # Cache factorials
        fact = np.array(
            [math.factorial(i) for i in range(kernel + 1)],
            dtype=np.int64,
        )

        # Filtering: bandpass if fmin/fmax provided, else adaptive lowpass
        data_concat = np.hstack(epochs_data)

        if fmin is not None and fmax is not None:
            # Bandpass filter
            nyquist = sfreq / 2.0
            low = fmin / nyquist
            high = fmax / nyquist
            b, a = butter(filter_order, [low, high], btype="band")

            for ch_idx in range(n_channels):
                data_concat[ch_idx, :] = filtfilt(b, a, data_concat[ch_idx, :])
        else:
            # Adaptive lowpass filter (matches old marker default)
            filter_freq = np.double(sfreq) / kernel / tau
            b, a = butter(6, 2.0 * filter_freq / np.double(sfreq), "lowpass")

            # Filter entire concatenated array at once
            data_concat = filtfilt(b, a, data_concat)

        # Split back: (n_channels, n_epochs*n_times) -> list of (n_channels, n_times)
        # np.array gives (n_epochs, n_channels, n_times), transpose to (n_epochs, n_times, n_channels)
        fdata = np.transpose(
            np.array(np.split(data_concat, n_epochs, axis=1)), [0, 2, 1]
        )  # Shape: (n_epochs, n_times, n_channels)

        # Apply time mask after filtering
        time_mask = _time_mask(epochs.times, tmin, tmax)
        fdata = fdata[
            :, time_mask, :
        ]  # Shape: (n_epochs, n_masked_times, n_channels)

        # Compute PE for each epoch and channel
        pe_values = np.zeros((n_epochs, n_channels))
        pe_func = _pe_numba if self._use_numba else _pe_numpy

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal = fdata[epoch_idx, :, ch_idx]
                pe_values[epoch_idx, ch_idx] = pe_func(
                    signal, kernel, tau, fact
                )

        return pe_values
