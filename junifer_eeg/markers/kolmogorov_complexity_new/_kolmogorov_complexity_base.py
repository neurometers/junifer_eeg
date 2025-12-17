"""Base Kolmogorov Complexity computation with caching (singleton pattern).

Follows the same pattern as PermutationEntropyBase for consistency.

Structure (2-file, matching permutation_entropy_new):
- _kolmogorov_complexity_base.py: Singleton with caching + compute logic (this file)
- kolmogorov_complexity.py: Main marker extending EEGEpochsMarker
"""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


__all__ = ["KolmogorovComplexityBase"]


class KolmogorovComplexityBase(metaclass=Singleton):
    """Kolmogorov Complexity computation with caching.

    Singleton class that computes Kolmogorov complexity with
    internal caching for efficient reuse across multiple markers.

    The caching uses (epochs_id, params) as key - valid while the
    epochs object exists in memory. This follows the same pattern
    as PermutationEntropyBase.
    """

    _DEPENDENCIES: ClassVar = {"numpy"}

    # Internal cache: {(epochs_id, params_tuple): kc_values}
    _cache: ClassVar[dict] = {}

    def __init__(self) -> None:
        """Initialize KC base."""
        pass

    def __del__(self) -> None:  # pragma: no cover
        """Terminate and clear cache."""
        logger.debug("Clearing cache for KC computation")
        KolmogorovComplexityBase._cache.clear()

    def compute(
        self,
        epochs: "mne.Epochs",
        nbins: int,
        tmin: float | None,
        tmax: float | None,
    ) -> np.ndarray:
        """Compute Kolmogorov complexity with caching.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object (already filtered to desired channels).
        nbins : int
            Number of bins for signal symbolization.
        tmin : float or None
            Start time for analysis.
        tmax : float or None
            End time for analysis.

        Returns
        -------
        kc_values : np.ndarray
            KC values with shape (n_epochs, n_channels).
        """
        # Build cache key from hashable parameters
        cache_key = (
            id(epochs),
            nbins,
            tmin,
            tmax,
        )

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"KC cache hit: epochs_id={id(epochs)}")
            return self._cache[cache_key]

        logger.debug(f"KC cache miss: computing nbins={nbins}")

        # Apply time cropping if specified
        if tmin is not None or tmax is not None:
            epochs = epochs.copy().crop(
                tmin=tmin if tmin is not None else epochs.tmin,
                tmax=tmax if tmax is not None else epochs.tmax,
            )

        epochs_data = epochs.get_data()
        n_epochs, n_channels, _ = epochs_data.shape

        # Compute KC for each epoch and channel
        kc_values = np.zeros((n_epochs, n_channels))

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal = epochs_data[epoch_idx, ch_idx, :]
                kc_values[epoch_idx, ch_idx] = self._compute_kc_for_signal(
                    signal, nbins
                )

        # Store in cache
        self._cache[cache_key] = kc_values

        return kc_values

    def _compute_kc_for_signal(self, signal: np.ndarray, nbins: int) -> float:
        """Compute Kolmogorov complexity for a single signal.

        Uses compression-based approximation with zlib. The signal is first
        symbolized into discrete bins, then compressed, and the complexity
        is measured as the compression ratio.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D array)
        nbins : int
            Number of bins for symbolization

        Returns
        -------
        float
            Kolmogorov complexity value (compression ratio)
        """
        import zlib

        # Symbolic transformation - match original algorithm exactly
        ssignal = np.sort(signal)
        items = signal.shape[0]
        first = int(items / 10)
        last = items - first if first > 1 else items - 1
        lower = ssignal[first]
        upper = ssignal[last]
        bsize = (upper - lower) / nbins

        # Create symbolic representation
        osignal = np.zeros(signal.shape, dtype=np.uint8)
        maxbin = nbins - 1

        for i in range(items):
            tbin = int((signal[i] - lower) / bsize) if bsize > 0 else 0
            osignal[i] = (0 if tbin < 0 else min(tbin, maxbin)) + ord("A")

        # Use zlib for compression
        string = osignal.tobytes()
        cstring = zlib.compress(string)
        return float(len(cstring)) / float(len(string))
