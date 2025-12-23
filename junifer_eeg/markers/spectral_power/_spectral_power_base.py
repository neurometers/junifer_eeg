"""Base PSD computation with caching for spectral power markers."""

from typing import TYPE_CHECKING, ClassVar, Optional, Tuple

import numpy as np
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


__all__ = ["SpectralPowerBase"]


class SpectralPowerBase(metaclass=Singleton):
    """Base PSD computation with caching.

    Singleton class that computes Power Spectral Density (PSD) with
    internal caching for efficient reuse across multiple markers.

    The caching uses (epochs_id, params) as key - valid while the
    epochs object exists in memory. This follows Fede's pattern but
    adapted for in-memory EEG data instead of file paths.

    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

    # Internal cache: {(epochs_id, params_tuple): (psds, freqs)}
    _cache: ClassVar[dict] = {}

    def __del__(self) -> None:  # pragma: no cover
        """Terminate the class and clear cache."""
        logger.debug("Clearing cache for PSD computation")
        SpectralPowerBase._cache.clear()

    def compute(
        self,
        epochs: "mne.Epochs",
        n_fft: Optional[int],
        n_per_seg: Optional[int],
        n_overlap: Optional[int],
        tmin: Optional[float],
        tmax: Optional[float],
        fmin: float,
        fmax: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PSD with caching for real epochs data.

        This method follows Fede's pattern (like AFNIReHo) but adapted for
        in-memory EEG data. The cache key is (epochs_id, params) - valid
        while the epochs object exists in memory.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object.
        n_fft : int, optional
            Length of the FFT.
        n_per_seg : int, optional
            Length of each segment.
        n_overlap : int, optional
            Number of overlap points.
        tmin : float, optional
            Start time for crop.
        tmax : float, optional
            End time for crop.
        fmin : float
            Minimum frequency.
        fmax : float
            Maximum frequency.

        Returns
        -------
        psds : np.ndarray
            Power spectral densities (n_epochs, n_channels, n_freqs).
        freqs : np.ndarray
            Frequencies (n_freqs,).

        """
        # Build cache key from hashable parameters (like Fede's input_path)
        cache_key = (
            id(epochs),
            n_fft,
            n_per_seg,
            n_overlap,
            tmin,
            tmax,
            fmin,
            fmax,
        )

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"PSD cache hit: epochs_id={id(epochs)}")
            return self._cache[cache_key]

        logger.debug(f"PSD cache miss: epochs_id={id(epochs)}, computing...")

        epochs_data = epochs.get_data()
        n_samples = epochs_data.shape[2]

        # Adaptive parameters
        effective_n_per_seg = n_per_seg
        effective_n_overlap = n_overlap
        if effective_n_per_seg is None:
            effective_n_per_seg = min(64, n_samples // 2)
        if effective_n_overlap is None:
            effective_n_overlap = min(32, effective_n_per_seg // 2)

        # Build PSD parameters
        psd_params = {
            "method": "welch",
            "fmin": fmin,
            "fmax": fmax,
            "n_per_seg": effective_n_per_seg,
            "n_overlap": effective_n_overlap,
            "verbose": False,
        }

        if n_fft is not None:
            psd_params["n_fft"] = n_fft

        # Crop if needed
        if tmin is not None or tmax is not None:
            epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)

        # Compute PSD
        psd = epochs.compute_psd(**psd_params)
        psds, freqs = psd.get_data(return_freqs=True)

        # Store in cache
        self._cache[cache_key] = (psds, freqs)

        return psds, freqs
