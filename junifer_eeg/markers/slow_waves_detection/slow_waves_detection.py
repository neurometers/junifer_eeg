"""Slow waves detection marker implementation.

Structure (2-file, matching kolmogorov_complexity_new):
- _slow_waves_detection_base.py: Singleton with caching + compute logic
- slow_waves_detection.py: Main marker extending EEGEpochsMarker (this file)
"""

from typing import Any, ClassVar, Optional, Tuple

from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import filter_to_eeg_channels
from ._slow_waves_detection_base import (
    SLOW_WAVE_FEATURES,
    SlowWavesDetectionBase,
)

__all__ = ["SlowWavesDetection"]


@register_marker
class SlowWavesDetection(EEGEpochsMarker):
    """Sleep slow waves detection marker using YASA.

    Detects slow waves in EEG data and returns the REQUESTED feature
    per epoch/channel. Uses singleton pattern with caching - multiple
    SlowWavesDetection markers with different features on the same epochs
    will reuse cached detection results.

    **IMPORTANT**: The `feature` parameter is MANDATORY and must be exactly
    one of: 'Duration', 'PTP', 'Frequency', 'Slope', 'Density'.

    Parameters
    ----------
    feature : str
        **MANDATORY**. The feature to extract. Must be one of:
        - 'Duration': Mean slow wave duration per epoch/channel (seconds)
        - 'PTP': Mean peak-to-peak amplitude per epoch/channel (µV)
        - 'Frequency': Mean slow wave frequency per epoch/channel (Hz)
        - 'Slope': Mean slow wave slope per epoch/channel (µV/s)
        - 'Density': Count of slow waves per epoch/channel
    freq_sw : tuple of float, default=(0.3, 1.5)
        Slow wave frequency range in Hz.
    amp_ptp_initial : float, default=15.0
        Initial peak-to-peak amplitude threshold in µV.
    freq_threshold : float, default=7.0
        Maximum frequency threshold for filtering (Hz).
    artifact_threshold : float, default=75.0
        Positive peak amplitude threshold for artifact removal (µV).
    reference_channels : tuple of str, default=("TP7", "TP8")
        Reference channel names for re-referencing.
    channel_method : str, optional
        Method for channel aggregation: 'mean', 'median', 'std', 'trim_mean80',
        'trim_mean90', etc. If None, no channel aggregation.
    trial_method : str, optional
        Method for trial aggregation: 'mean', 'median', 'std', 'trim_mean80',
        'trim_mean90', etc. If None, no trial aggregation.
    equipment : str, default="egi256"
        Equipment configuration.
    on : str, optional
        Data type (default "EEG").
    name : str, optional
        Marker name.

    Examples
    --------
    >>> # Get slow wave duration per epoch/channel
    >>> marker = SlowWavesDetection(feature="Duration")

    >>> # Get slow wave PTP amplitude with channel aggregation
    >>> marker = SlowWavesDetection(
    ...     feature="PTP",
    ...     channel_method="mean"
    ... )

    >>> # Get slow wave density with full aggregation
    >>> marker = SlowWavesDetection(
    ...     feature="Density",
    ...     channel_method="mean",
    ...     trial_method="mean"
    ... )
    """

    _DEPENDENCIES: ClassVar = {"mne", "yasa", "pandas", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"slowwavesdetection": "timeseries"}
    }

    def __init__(
        self,
        feature: str,
        freq_sw: Tuple[float, float] = (0.3, 1.5),
        amp_ptp_initial: float = 15.0,
        freq_threshold: float = 7.0,
        artifact_threshold: float = 75.0,
        reference_channels: Tuple[str, ...] = ("TP7", "TP8"),
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SlowWavesDetection marker."""
        super().__init__(
            tmin=None, tmax=None, equipment=equipment, on=on, name=name
        )

        # Validate feature parameter - MANDATORY
        if feature not in SLOW_WAVE_FEATURES:
            raise ValueError(
                f"'feature' parameter is MANDATORY and must be one of: "
                f"{SLOW_WAVE_FEATURES}. Got: '{feature}'"
            )

        self.feature = feature
        self.freq_sw = tuple(freq_sw)
        self.amp_ptp_initial = amp_ptp_initial
        self.freq_threshold = freq_threshold
        self.artifact_threshold = artifact_threshold
        self.reference_channels = tuple(reference_channels)
        self.channel_method = channel_method
        self.trial_method = trial_method

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute slow wave detection and return requested feature.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed slow wave feature with keys:
            - 'data': array of shape (n_epochs, n_channels)
            - 'col_names': channel names
        """
        logger.debug(
            f"Computing slow waves detection (feature={self.feature})"
        )

        # Get and validate epochs
        data_obj = input["data"]
        self._validate_input(data_obj)

        # Filter to EEG channels only
        data_obj, _, _ = filter_to_eeg_channels(data_obj)
        ch_names = list(data_obj.ch_names)

        # Compute all features using singleton (with caching)
        sw_base = SlowWavesDetectionBase()
        all_features = sw_base.compute(
            data_obj,
            self.freq_sw,
            self.amp_ptp_initial,
            self.freq_threshold,
            self.artifact_threshold,
            self.reference_channels,
        )

        # Extract the requested feature
        output_data = all_features[self.feature]

        # Apply aggregation using common helper
        from ..utils import apply_channel_trial_aggregation

        output_data = apply_channel_trial_aggregation(
            output_data, self.channel_method, self.trial_method
        )

        # Use centralized format_marker_result
        return format_marker_result(
            feature_name="slowwavesdetection",
            data=output_data,
            col_names=ch_names,
            channel_aggregated=(self.channel_method is not None),
        )

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns
        -------
        str
            Output type based on aggregation state.
        """
        # No aggregation → 2D tensor (epochs, channels) → timeseries
        if self.channel_method is None and self.trial_method is None:
            return "timeseries"

        # Both aggregations → scalar → scalar_table
        if self.channel_method is not None and self.trial_method is not None:
            return "scalar_table"

        # One aggregation → 1D array → vector
        return "vector"
