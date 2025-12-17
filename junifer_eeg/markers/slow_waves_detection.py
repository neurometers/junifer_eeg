"""Slow waves detection marker using YASA."""

from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


@register_marker
class SlowWavesDetection(BaseMarker):
    """Sleep slow waves detection marker using YASA.

    Detects sleep slow waves in EEG data and returns mean values per epoch/channel
    for the main slow wave features: Duration, PTP, Frequency, Slope, and Density.
    """

    _DEPENDENCIES: ClassVar = {"mne", "yasa", "pandas", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"slowwavesdetection": "timeseries"}
    }

    def __init__(
        self,
        freq_sw: tuple = (0.3, 1.5),
        amp_ptp_initial: float = 15.0,
        freq_threshold: float = 7.0,
        artifact_threshold: float = 75.0,
        reference_channels: tuple = ("TP7", "TP8"),
        channel_method: Optional[str] = None,
        epoch_aggregation_method: Optional[str] = None,
        on: str | list[str] = "EEG",
        name: str = "slowwaves",
    ) -> None:
        """Initialize SlowWavesDetection marker.

        Parameters
        ----------
        freq_sw : tuple, default=(0.3, 1.5)
            Slow wave frequency range in Hz.
        amp_ptp_initial : float, default=15.0
            Initial peak-to-peak amplitude threshold in µV.
        freq_threshold : float, default=7.0
            Maximum frequency threshold for filtering (Hz).
        artifact_threshold : float, default=75.0
            Positive peak amplitude threshold for artifact removal (µV).
        reference_channels : tuple, default=("TP7", "TP8")
            Reference channel names for re-referencing.
        channel_method : str, optional
            Method to aggregate across channels: 'mean', 'std', 'median', etc.
            If None, keeps per-channel events.
        epoch_aggregation_method : str, optional
            Method to aggregate across epochs: 'mean', 'std', 'median', etc.
            If None, keeps per-epoch events.
        on : str or list of str, default="EEG"
            Name of the input marker to use.
        name : str, default="slowwaves"
            Name of the marker.
        """
        self.freq_sw = freq_sw
        self.amp_ptp_initial = amp_ptp_initial
        self.freq_threshold = freq_threshold
        self.artifact_threshold = artifact_threshold
        self.reference_channels = reference_channels
        self.channel_method = channel_method
        self.epoch_aggregation_method = epoch_aggregation_method

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict:
        """Detect slow waves in EEG data.

        Parameters
        ----------
        input : dict
            Input data dictionary containing 'data' key with MNE Epochs object.
        extra_input : dict, optional
            Extra input data (not used).

        Returns
        -------
        dict
            Dictionary with slow waves detection results.
            Returns 3D tensor with shape (n_features, n_epochs, n_channels) where
            features are [Duration, PTP, Frequency, Slope, Density] and each value is the mean
            of all slow waves for that epoch/channel combination, except Density which is the
            count of detected events. Empty cells contain NaN.
            Aggregation reduces dimensions following SpectralPower pattern.
        """
        import mne
        import yasa

        # Get epochs object
        data_obj = input["data"]
        if not isinstance(data_obj, mne.BaseEpochs):
            raise ValueError(
                "SlowWavesDetection requires MNE Epochs object. "
                f"Got {type(data_obj)}"
            )

        # Make a copy to avoid modifying original data
        epochs_copy = data_obj.copy()

        # Re-reference to specified reference channels
        try:
            # Check if reference channels exist
            missing_channels = [
                ch
                for ch in self.reference_channels
                if ch not in epochs_copy.ch_names
            ]
            if missing_channels:
                raise ValueError(
                    f"Reference channels {missing_channels} not found in data. "
                    f"Available channels: {epochs_copy.ch_names}"
                )

            # Check if reference channels have valid data (not all zeros)
            for ch in self.reference_channels:
                ch_data = epochs_copy.get_data(picks=ch)
                if np.allclose(ch_data, 0):
                    raise ValueError(
                        f"Reference channel {ch} contains all zeros"
                    )

            # Apply custom referencing
            ref_data = epochs_copy.get_data(
                picks=self.reference_channels
            ).mean(axis=1)
            ref_data = ref_data[:, np.newaxis, :]  # Add channel dimension
            epochs_copy._data -= ref_data

        except Exception as e:
            print(f"Warning: Re-referencing failed ({e}), using original data")

        # Get sampling frequency and channel info
        sf = epochs_copy.info["sfreq"]
        ch_names = epochs_copy.ch_names
        chan2idx = {ch: i for i, ch in enumerate(ch_names)}

        # Collect all events from all epochs
        all_events = []

        # Process each epoch individually
        for epoch_idx in range(len(epochs_copy)):
            epoch_data = epochs_copy[epoch_idx].get_data()[
                0
            ]  # (n_channels, n_times)

            # Convert from volts to microvolts (YASA expects µV)
            epoch_data_uV = epoch_data * 1e6

            # Run YASA slow waves detection (multi-channel native)
            res = yasa.sw_detect(
                data=epoch_data_uV,
                sf=sf,
                ch_names=ch_names,
                freq_sw=self.freq_sw,
                amp_ptp=(
                    self.amp_ptp_initial,
                    np.inf,
                ),  # Correct API: tuple (min, max)
                coupling=False,
                remove_outliers=False,
                verbose=False,
            )

            # Collect events from this epoch
            df = res.summary() if res is not None else None
            if df is not None and len(df):
                df = df.copy()
                df["Epoch"] = epoch_idx
                # YASA already has "Channel" column for multichannel results
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
            events_df = self._apply_dynamic_threshold(events_df)

        # Get dimensions for 3D tensor
        n_epochs = len(epochs_copy)
        n_channels = len(ch_names)
        n_features = 5  # Duration, PTP, Frequency, Slope, Density

        # Initialize 3D tensor with NaN
        tensor = np.full((n_features, n_epochs, n_channels), np.nan)

        # Fill tensor with mean values per (epoch, channel, feature)
        if not events_df.empty:
            # Group by (Epoch, ChanIdx) and compute means
            grouped = events_df.groupby(["Epoch", "ChanIdx"])
            feature_cols = ["Duration", "PTP", "Frequency", "Slope"]

            for (epoch_idx, chan_idx), group in grouped:
                if epoch_idx < n_epochs and chan_idx < n_channels:
                    # Add mean values for existing features
                    for feat_idx, feat in enumerate(feature_cols):
                        tensor[feat_idx, epoch_idx, chan_idx] = group[
                            feat
                        ].mean()
                    # Add density feature (count of events)
                    tensor[4, epoch_idx, chan_idx] = len(group)

        # Apply aggregation if specified (following SpectralPower pattern)
        if (
            self.channel_method is not None
            or self.epoch_aggregation_method is not None
        ):
            tensor = self._apply_aggregation(tensor)

        # Return structured results in junifer format
        result = {"slowwavesdetection": {"data": tensor}}

        # Include col_names only for non-aggregated data (like SpectralPower)
        if (
            self.channel_method is None
            and self.epoch_aggregation_method is None
        ):
            result["slowwavesdetection"]["col_names"] = ch_names

        return result

    def _apply_dynamic_threshold(self, sw_df: pd.DataFrame) -> pd.DataFrame:
        """Apply dynamic thresholding and filtering to slow waves.

        This implements the Andrillon and Pinggal filtering criteria:
        1. Remove waves with frequency > 7 Hz
        2. Calculate 90th percentile of PTP per channel
        3. Keep only waves with PTP > channel-specific 90th percentile
        4. Remove artifacts with positive peak > 75 µV

        Parameters
        ----------
        sw_df : pd.DataFrame
            DataFrame with all detected slow waves.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        # Rule 1: Remove high frequency waves (> 7 Hz)
        sw_df = sw_df[sw_df["Frequency"] <= self.freq_threshold].copy()

        if sw_df.empty:
            return sw_df

        # Rule 2 & 3: Dynamic amplitude threshold (90th percentile per channel)
        # Calculate 90th percentile of PTP for each channel
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

    def _apply_aggregation(self, tensor: np.ndarray) -> np.ndarray:
        """Apply aggregation method to 3D tensor (following SpectralPower pattern).

        Parameters
        ----------
        tensor : np.ndarray
            3D tensor with shape (n_features, n_epochs, n_channels).

        Returns
        -------
        np.ndarray
            Aggregated tensor with reduced dimensions.
        """
        from .utils import aggregate_data

        result_data = tensor

        # Channel aggregation (axis=2 for channels)
        if self.channel_method is not None:
            result_data = aggregate_data(
                result_data, self.channel_method, axis=2
            )

        # Epoch aggregation (axis=1 for epochs)
        if self.epoch_aggregation_method is not None:
            if result_data.ndim == 1:
                result_data = aggregate_data(
                    result_data, self.epoch_aggregation_method, axis=None
                )
            else:
                result_data = aggregate_data(
                    result_data, self.epoch_aggregation_method, axis=1
                )

        return result_data

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings (following SpectralPower pattern).

        Returns
        -------
        str
            Output type based on aggregation state.
        """
        # No aggregation → 3D tensor (features, epochs, channels) → timeseries
        if (
            self.channel_method is None
            and self.epoch_aggregation_method is None
        ):
            return "timeseries"

        # Both aggregations → scalar → use scalar_table
        if (
            self.channel_method is not None
            and self.epoch_aggregation_method is not None
        ):
            return "scalar_table"

        # One aggregation → 1D array → use vector
        return "vector"
