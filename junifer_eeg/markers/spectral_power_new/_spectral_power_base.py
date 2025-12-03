"""Base PSD computation with caching for spectral power markers."""

from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple

import numpy as np
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


__all__ = ["SpectralPowerBase"]


class SpectralPowerBase(metaclass=Singleton):
    """Base PSD computation with caching.

    Singleton class that computes Power Spectral Density (PSD) with LRU
    caching for efficient reuse across multiple markers.

    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

    def __del__(self) -> None:  # pragma: no cover
        """Terminate the class and clear cache."""
        logger.debug("Clearing cache for PSD computation")
        SpectralPowerBase.compute.cache_clear()

    @staticmethod
    @lru_cache(maxsize=None, typed=True)
    def compute(
        epochs_id: int,
        n_fft: Optional[int],
        n_per_seg: Optional[int],
        n_overlap: Optional[int],
        tmin: Optional[float],
        tmax: Optional[float],
        fmin: float,
        fmax: float,
        sfreq: float,
        n_epochs: int,
        n_channels: int,
        n_times: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PSD with caching - delegates to actual computation.

        This method is cached based on parameters only. The actual epochs
        data is passed separately to the _compute_psd method.

        """
        # This is just a cache wrapper - actual computation happens
        # in the calling marker with real epochs data
        logger.debug(f"PSD cache key: epochs_id={epochs_id}")
        return None, None  # Placeholder

    def _compute_psd(
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
        """Compute PSD for real epochs data.

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
        epochs_data = epochs.get_data()
        n_samples = epochs_data.shape[2]

        # Adaptive parameters
        if n_per_seg is None:
            n_per_seg = min(64, n_samples // 2)
        if n_overlap is None:
            n_overlap = min(32, n_per_seg // 2)

        # Build PSD parameters
        psd_params = {
            "method": "welch",
            "fmin": fmin,
            "fmax": fmax,
            "n_per_seg": n_per_seg,
            "n_overlap": n_overlap,
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

        return psds, freqs
