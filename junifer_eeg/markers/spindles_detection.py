"""Sleep spindles detection marker using YASA."""

from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


@register_marker
class SpindlesDetection(BaseMarker):
    """Sleep spindles detection marker using YASA.

    Detects sleep spindles in EEG data and returns mean values per epoch/channel
    for the main spindle features: Duration, Amplitude, and Frequency.
    """

    _DEPENDENCIES: ClassVar = {"mne", "yasa", "pandas", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"spindlesdetection": "timeseries"}
    }

    def __init__(
        self,
        freq_sp: tuple = (12, 15),
        freq_broad: tuple = (1, 30),
        duration: tuple = (0.5, 2.5),
        min_distance: int = 500,
        thresh_rms: float = 1.5,
        thresh_corr: float = 0.65,
        reference_channels: tuple = ("TP7", "TP8"),
        channel_aggregation_method: Optional[str] = None,
        epoch_aggregation_method: Optional[str] = None,
        on: str | list[str] = "EEG",
        name: str = "spindles",
    ) -> None:
        """Initialize SpindlesDetection marker.

        Parameters
        ----------
        freq_sp : tuple, default=(12, 15)
            Spindles frequency range in Hz.
        freq_broad : tuple, default=(1, 30)
            Broad band frequency range in Hz.
        duration : tuple, default=(0.5, 2.5)
            Min/max spindle duration (seconds).
        min_distance : int, default=500
            Minimum distance between spindles (ms).
        thresh_rms : float, default=1.5
            RMS threshold (SD above mean).
        thresh_corr : float, default=0.65
            Correlation threshold.
        reference_channels : tuple, default=("TP7", "TP8")
            Reference channel names for re-referencing.
        channel_aggregation_method : str, optional
            Method to aggregate across channels: 'mean', 'std', 'median', etc.
            If None, keeps per-channel events.
        epoch_aggregation_method : str, optional
            Method to aggregate across epochs: 'mean', 'std', 'median', etc.
            If None, keeps per-epoch events.
        on : str or list of str, default="EEG"
            Name of the input marker to use.
        name : str, default="spindles"
            Name of the marker.
        """

        self.freq_sp = freq_sp
        self.freq_broad = freq_broad
        self.duration = duration
        self.min_distance = min_distance
        self.thresh_rms = thresh_rms
        self.thresh_corr = thresh_corr
        self.reference_channels = reference_channels
        self.channel_aggregation_method = channel_aggregation_method
        self.epoch_aggregation_method = epoch_aggregation_method

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict:
        """Detect sleep spindles in EEG data.

        Parameters
        ----------
        input : dict
            Input data dictionary containing 'data' key with MNE Epochs object.
        extra_input : dict, optional
            Extra input data (not used).

        Returns
        -------
        dict
            Dictionary with spindles detection results.
            Returns 3D tensor with shape (n_features, n_epochs, n_channels) where
            features are [Duration, Amplitude, Frequency] and each value is the mean
            of all spindles for that epoch/channel combination. Empty cells contain NaN.
            Aggregation reduces dimensions following SpectralPower pattern.
        """
        import mne
        import yasa

        # Get epochs object
        data_obj = input["data"]
        if not isinstance(data_obj, mne.BaseEpochs):
            raise ValueError(
                "SpindlesDetection requires MNE Epochs object. "
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

            # Validate reference channels have valid data
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

            # Run YASA spindles detection (multi-channel native)
            res = yasa.spindles_detect(
                data=epoch_data_uV,
                sf=sf,
                ch_names=ch_names,
                freq_sp=self.freq_sp,
                freq_broad=self.freq_broad,
                duration=self.duration,
                min_distance=self.min_distance,
                thresh={"rms": self.thresh_rms, "corr": self.thresh_corr},
                multi_only=False,
                remove_outliers=False,
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

        # Get dimensions for 3D tensor
        n_epochs = len(epochs_copy)
        n_channels = len(ch_names)
        n_features = 3  # Duration, Amplitude, Frequency

        # Initialize 3D tensor with NaN
        tensor = np.full((n_features, n_epochs, n_channels), np.nan)

        # Fill tensor with mean values per (epoch, channel, feature)
        if not events_df.empty:
            # Group events by (Epoch, ChanIdx) and compute mean features
            grouped = events_df.groupby(["Epoch", "ChanIdx"])
            feature_cols = ["Duration", "Amplitude", "Frequency"]

            for (epoch_idx, chan_idx), group in grouped:
                if epoch_idx < n_epochs and chan_idx < n_channels:
                    for feat_idx, feat in enumerate(feature_cols):
                        tensor[feat_idx, epoch_idx, chan_idx] = group[
                            feat
                        ].mean()

        # Apply aggregation if specified (following SpectralPower pattern)
        if (
            self.channel_aggregation_method is not None
            or self.epoch_aggregation_method is not None
        ):
            tensor = self._apply_aggregation(tensor)

        # Return structured results in junifer format
        result = {"spindlesdetection": {"data": tensor}}

        # Include col_names only for non-aggregated data (like SpectralPower)
        if (
            self.channel_aggregation_method is None
            and self.epoch_aggregation_method is None
        ):
            result["spindlesdetection"]["col_names"] = ch_names

        return result

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
        if self.channel_aggregation_method is not None:
            result_data = aggregate_data(
                result_data, self.channel_aggregation_method, axis=2
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
            self.channel_aggregation_method is None
            and self.epoch_aggregation_method is None
        ):
            return "timeseries"

        # Both aggregations → scalar → use scalar_table
        if (
            self.channel_aggregation_method is not None
            and self.epoch_aggregation_method is not None
        ):
            return "scalar_table"

        # One aggregation → 1D array → use vector
        return "vector"
