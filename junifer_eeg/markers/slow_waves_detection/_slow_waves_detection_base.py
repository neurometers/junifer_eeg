"""Base slow waves detection computation with caching (singleton pattern).

Structure (2-file, matching kolmogorov_complexity_new):
- _slow_waves_detection_base.py: Singleton with caching + compute logic (this file)
- slow_waves_detection.py: Main marker extending EEGEpochsMarker
"""

from typing import TYPE_CHECKING, ClassVar, Dict, Tuple

import numpy as np
import pandas as pd
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


__all__ = ["SlowWavesDetectionBase"]


# Valid features that can be computed
SLOW_WAVE_FEATURES = ("Duration", "PTP", "Frequency", "Slope", "Density")


class SlowWavesDetectionBase(metaclass=Singleton):
    """Slow waves detection computation with caching.

    Singleton class that detects slow waves and computes ALL features with
    internal caching for efficient reuse across multiple markers.

    The caching uses (epochs_id, params) as key - valid while the
    epochs object exists in memory. This follows the same pattern
    as KolmogorovComplexityBase.

    Features computed:
    - Duration: Mean slow wave duration per epoch/channel
    - PTP: Mean peak-to-peak amplitude per epoch/channel
    - Frequency: Mean slow wave frequency per epoch/channel
    - Slope: Mean slow wave slope per epoch/channel
    - Density: Count of slow waves per epoch/channel
    """

    _DEPENDENCIES: ClassVar = {"numpy", "yasa", "pandas", "mne"}

    # Internal cache: {(epochs_id, params_tuple): {feature_name: array}}
    _cache: ClassVar[dict] = {}

    def __init__(self) -> None:
        """Initialize slow waves detection base."""
        pass

    def __del__(self) -> None:  # pragma: no cover
        """Terminate and clear cache."""
        logger.debug("Clearing cache for slow waves detection computation")
        SlowWavesDetectionBase._cache.clear()

    def compute(
        self,
        epochs: "mne.Epochs",
        freq_sw: Tuple[float, float],
        amp_ptp_initial: float,
        freq_threshold: float,
        artifact_threshold: float,
        reference_channels: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """Compute slow wave detection with caching.

        Returns ALL features in a dictionary. The caller selects which
        feature to return based on user request.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object (already filtered to desired channels).
        freq_sw : tuple of float
            Slow wave frequency range in Hz (fmin, fmax).
        amp_ptp_initial : float
            Initial peak-to-peak amplitude threshold in µV.
        freq_threshold : float
            Maximum frequency threshold for filtering (Hz).
        artifact_threshold : float
            Positive peak amplitude threshold for artifact removal (µV).
        reference_channels : tuple of str
            Reference channel names for re-referencing.

        Returns
        -------
        features : dict of np.ndarray
            Dictionary with keys 'Duration', 'PTP', 'Frequency', 'Slope', 'Density'.
            Each value is an array of shape (n_epochs, n_channels).
        """
        # Build cache key from hashable parameters
        cache_key = (
            id(epochs),
            freq_sw,
            amp_ptp_initial,
            freq_threshold,
            artifact_threshold,
            reference_channels,
        )

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"SlowWaves cache hit: epochs_id={id(epochs)}")
            return self._cache[cache_key]

        logger.debug(
            f"SlowWaves cache miss: computing freq_sw={freq_sw}, "
            f"amp_ptp={amp_ptp_initial}"
        )

        # Perform actual computation
        features = self._detect_slow_waves(
            epochs,
            freq_sw,
            amp_ptp_initial,
            freq_threshold,
            artifact_threshold,
            reference_channels,
        )

        # Store in cache
        self._cache[cache_key] = features

        return features

    def _detect_slow_waves(
        self,
        epochs: "mne.Epochs",
        freq_sw: Tuple[float, float],
        amp_ptp_initial: float,
        freq_threshold: float,
        artifact_threshold: float,
        reference_channels: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """Perform slow wave detection on epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object.
        freq_sw : tuple of float
            Slow wave frequency range in Hz.
        amp_ptp_initial : float
            Initial peak-to-peak amplitude threshold in µV.
        freq_threshold : float
            Maximum frequency threshold for filtering.
        artifact_threshold : float
            Positive peak amplitude threshold for artifact removal.
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

            # Run YASA slow waves detection
            res = yasa.sw_detect(
                data=epoch_data_uV,
                sf=sf,
                ch_names=ch_names,
                freq_sw=freq_sw,
                amp_ptp=(amp_ptp_initial, np.inf),
                coupling=False,
                remove_outliers=False,
                verbose=False,
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

        # Apply post-processing filtering (Andrillon & Pinggal criteria)
        if not events_df.empty:
            events_df = self._apply_dynamic_threshold(
                events_df, freq_threshold, artifact_threshold
            )

        # Get dimensions
        n_epochs = len(epochs_copy)
        n_channels = len(ch_names)

        # Initialize feature arrays with NaN
        features = {
            "Duration": np.full((n_epochs, n_channels), np.nan),
            "PTP": np.full((n_epochs, n_channels), np.nan),
            "Frequency": np.full((n_epochs, n_channels), np.nan),
            "Slope": np.full((n_epochs, n_channels), np.nan),
            "Density": np.full((n_epochs, n_channels), np.nan),
        }

        # Fill feature arrays
        if not events_df.empty:
            grouped = events_df.groupby(["Epoch", "ChanIdx"])

            for (epoch_idx, chan_idx), group in grouped:
                if epoch_idx < n_epochs and chan_idx < n_channels:
                    features["Duration"][epoch_idx, chan_idx] = group[
                        "Duration"
                    ].mean()
                    features["PTP"][epoch_idx, chan_idx] = group["PTP"].mean()
                    features["Frequency"][epoch_idx, chan_idx] = group[
                        "Frequency"
                    ].mean()
                    features["Slope"][epoch_idx, chan_idx] = group[
                        "Slope"
                    ].mean()
                    features["Density"][epoch_idx, chan_idx] = len(group)

        return features

    def _apply_dynamic_threshold(
        self,
        sw_df: pd.DataFrame,
        freq_threshold: float,
        artifact_threshold: float,
    ) -> pd.DataFrame:
        """Apply dynamic thresholding and filtering to slow waves.

        This implements the Andrillon and Pinggal filtering criteria:
        1. Remove waves with frequency > freq_threshold Hz
        2. Calculate 90th percentile of PTP per channel
        3. Keep only waves with PTP > channel-specific 90th percentile
        4. Remove artifacts with positive peak > artifact_threshold µV

        Parameters
        ----------
        sw_df : pd.DataFrame
            DataFrame with all detected slow waves.
        freq_threshold : float
            Maximum frequency threshold.
        artifact_threshold : float
            Positive peak amplitude threshold for artifact removal.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        # Rule 1: Remove high frequency waves
        sw_df = sw_df[sw_df["Frequency"] <= freq_threshold].copy()

        if sw_df.empty:
            return sw_df

        # Rule 2 & 3: Dynamic amplitude threshold (90th percentile per channel)
        channel_thresholds = {}
        for channel in sw_df["Channel"].unique():
            channel_sw = sw_df[sw_df["Channel"] == channel]
            channel_thresholds[channel] = np.percentile(channel_sw["PTP"], 90)

        # Filter: keep only waves above channel-specific threshold
        keep_mask = [
            row["PTP"] >= channel_thresholds[row["Channel"]]
            for _, row in sw_df.iterrows()
        ]
        sw_df = sw_df[keep_mask].copy()

        return sw_df
