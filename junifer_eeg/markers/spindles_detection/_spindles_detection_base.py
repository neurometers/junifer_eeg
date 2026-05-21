"""Base spindles detection computation with caching (singleton pattern).

Structure (2-file, matching kolmogorov_complexity_new):
- _spindles_detection_base.py: Singleton with caching + compute logic (this file)
- spindles_detection.py: Main marker extending EEGEpochsMarker
"""

from typing import TYPE_CHECKING, ClassVar, Dict, Tuple

import numpy as np
import pandas as pd
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


__all__ = ["SpindlesDetectionBase"]


# Valid features that can be computed
SPINDLE_FEATURES = ("Duration", "Amplitude", "Frequency", "Density")


class SpindlesDetectionBase(metaclass=Singleton):
    """Spindles detection computation with caching.

    Singleton class that detects spindles and computes ALL features with
    internal caching for efficient reuse across multiple markers.

    The caching uses (epochs_id, params) as key - valid while the
    epochs object exists in memory. This follows the same pattern
    as KolmogorovComplexityBase.

    Features computed:
    - Duration: Mean spindle duration per epoch/channel
    - Amplitude: Mean spindle amplitude per epoch/channel
    - Frequency: Mean spindle frequency per epoch/channel
    - Density: Count of spindles per epoch/channel
    """

    _DEPENDENCIES: ClassVar = {"numpy", "yasa", "pandas", "mne"}

    # Internal cache: {(epochs_id, params_tuple): {feature_name: array}}
    _cache: ClassVar[dict] = {}

    def __init__(self) -> None:
        """Initialize spindles detection base."""
        pass

    def __del__(self) -> None:  # pragma: no cover
        """Terminate and clear cache."""
        logger.debug("Clearing cache for spindles detection computation")
        SpindlesDetectionBase._cache.clear()

    def compute(
        self,
        epochs: "mne.Epochs",
        freq_sp: Tuple[float, float],
        freq_broad: Tuple[float, float],
        duration: Tuple[float, float],
        min_distance: int,
        thresh_rms: float,
        thresh_corr: float,
        reference_channels: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """Compute spindle detection with caching.

        Returns ALL features in a dictionary. The caller selects which
        feature to return based on user request.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object (already filtered to desired channels).
        freq_sp : tuple of float
            Spindles frequency range in Hz (fmin, fmax).
        freq_broad : tuple of float
            Broad band frequency range in Hz (fmin, fmax).
        duration : tuple of float
            Min/max spindle duration in seconds.
        min_distance : int
            Minimum distance between spindles in ms.
        thresh_rms : float
            RMS threshold (SD above mean).
        thresh_corr : float
            Correlation threshold.
        reference_channels : tuple of str
            Reference channel names for re-referencing.

        Returns
        -------
        features : dict of np.ndarray
            Dictionary with keys 'Duration', 'Amplitude', 'Frequency', 'Density'.
            Each value is an array of shape (n_epochs, n_channels).
        """
        # Build cache key from hashable parameters
        cache_key = (
            id(epochs),
            freq_sp,
            freq_broad,
            duration,
            min_distance,
            thresh_rms,
            thresh_corr,
            reference_channels,
        )

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"Spindles cache hit: epochs_id={id(epochs)}")
            return self._cache[cache_key]

        logger.debug(
            f"Spindles cache miss: computing freq_sp={freq_sp}, "
            f"duration={duration}"
        )

        # Perform actual computation
        features = self._detect_spindles(
            epochs,
            freq_sp,
            freq_broad,
            duration,
            min_distance,
            thresh_rms,
            thresh_corr,
            reference_channels,
        )

        # Store in cache
        self._cache[cache_key] = features

        return features

    def _detect_spindles(
        self,
        epochs: "mne.Epochs",
        freq_sp: Tuple[float, float],
        freq_broad: Tuple[float, float],
        duration: Tuple[float, float],
        min_distance: int,
        thresh_rms: float,
        thresh_corr: float,
        reference_channels: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """Perform spindle detection on epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object.
        freq_sp : tuple of float
            Spindles frequency range in Hz.
        freq_broad : tuple of float
            Broad band frequency range in Hz.
        duration : tuple of float
            Min/max spindle duration in seconds.
        min_distance : int
            Minimum distance between spindles in ms.
        thresh_rms : float
            RMS threshold.
        thresh_corr : float
            Correlation threshold.
        reference_channels : tuple of str
            Reference channel names.

        Returns
        -------
        features : dict of np.ndarray
            Dictionary with all computed features.
        """
        import yasa

        # Make a copy to avoid modifying original data
        epochs_copy = epochs.copy()

        # Re-reference to specified reference channels
        try:
            # Check if reference channels exist
            missing_channels = [
                ch
                for ch in reference_channels
                if ch not in epochs_copy.ch_names
            ]
            if missing_channels:
                logger.warning(
                    f"Reference channels {missing_channels} not found. "
                    "Using original data."
                )
            else:
                # Check if reference channels have valid data
                valid_ref = True
                for ch in reference_channels:
                    ch_data = epochs_copy.get_data(picks=ch)
                    if np.allclose(ch_data, 0):
                        logger.warning(
                            f"Reference channel {ch} contains all zeros. "
                            "Using original data."
                        )
                        valid_ref = False
                        break

                if valid_ref:
                    # Apply custom referencing
                    ref_data = epochs_copy.get_data(
                        picks=list(reference_channels)
                    ).mean(axis=1)
                    ref_data = ref_data[:, np.newaxis, :]
                    epochs_copy._data -= ref_data

        except Exception as e:
            logger.warning(f"Re-referencing failed ({e}), using original data")

        # Get sampling frequency and channel info
        sf = epochs_copy.info["sfreq"]
        ch_names = epochs_copy.ch_names
        chan2idx = {ch: i for i, ch in enumerate(ch_names)}

        # Collect all events from all epochs
        all_events = []

        # Process each epoch individually
        for epoch_idx in range(len(epochs_copy)):
            epoch_data = epochs_copy[epoch_idx].get_data()[0]

            # Convert from volts to microvolts (YASA expects µV)
            epoch_data_uV = epoch_data * 1e6

            # Run YASA spindles detection
            res = yasa.spindles_detect(
                data=epoch_data_uV,
                sf=sf,
                ch_names=ch_names,
                freq_sp=freq_sp,
                freq_broad=freq_broad,
                duration=duration,
                min_distance=min_distance,
                thresh={"rms": thresh_rms, "corr": thresh_corr},
                multi_only=False,
                remove_outliers=False,
            )

            # Collect events from this epoch
            df = res.summary() if res is not None else None
            if df is not None and len(df):
                df = df.copy()
                df["Epoch"] = epoch_idx
                df["ChanIdx"] = df["Channel"].map(chan2idx)
                all_events.append(df)

        # Combine all events from all epochs
        events_df = (
            pd.concat(all_events, ignore_index=True)
            if all_events
            else pd.DataFrame()
        )

        # Get dimensions
        n_epochs = len(epochs_copy)
        n_channels = len(ch_names)

        # Means stay NaN when no spindles exist (mean of nothing is undefined).
        # Density is a count: zero detections is 0, not missing. Upstream
        # bad-channel/epoch rejection is responsible for true missingness.
        features = {
            "Duration": np.full((n_epochs, n_channels), np.nan),
            "Amplitude": np.full((n_epochs, n_channels), np.nan),
            "Frequency": np.full((n_epochs, n_channels), np.nan),
            "Density": np.zeros((n_epochs, n_channels)),
        }

        # Fill feature arrays
        if not events_df.empty:
            grouped = events_df.groupby(["Epoch", "ChanIdx"])

            for (epoch_idx, chan_idx), group in grouped:
                if epoch_idx < n_epochs and chan_idx < n_channels:
                    features["Duration"][epoch_idx, chan_idx] = group[
                        "Duration"
                    ].mean()
                    features["Amplitude"][epoch_idx, chan_idx] = group[
                        "Amplitude"
                    ].mean()
                    features["Frequency"][epoch_idx, chan_idx] = group[
                        "Frequency"
                    ].mean()
                    features["Density"][epoch_idx, chan_idx] = len(group)

        return features
