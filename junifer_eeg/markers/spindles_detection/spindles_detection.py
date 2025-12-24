"""Spindles detection marker implementation.

Structure (2-file, matching kolmogorov_complexity_new):
- _spindles_detection_base.py: Singleton with caching + compute logic
- spindles_detection.py: Main marker extending EEGEpochsMarker (this file)
"""

from typing import Any, ClassVar, Optional, Tuple

from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import apply_aggregation_preserve_dims, filter_to_eeg_channels
from ._spindles_detection_base import SPINDLE_FEATURES, SpindlesDetectionBase

__all__ = ["SpindlesDetection"]


@register_marker
class SpindlesDetection(EEGEpochsMarker):
    """Sleep spindles detection marker using YASA.

    Detects sleep spindles in EEG data and returns the REQUESTED feature
    per epoch/channel. Uses singleton pattern with caching - multiple
    SpindlesDetection markers with different features on the same epochs
    will reuse cached detection results.

    **IMPORTANT**: The `feature` parameter is MANDATORY and must be exactly
    one of: 'Duration', 'Amplitude', 'Frequency', 'Density'.

    Parameters
    ----------
    feature : str
        **MANDATORY**. The feature to extract. Must be one of:
        - 'Duration': Mean spindle duration per epoch/channel (seconds)
        - 'Amplitude': Mean spindle amplitude per epoch/channel (µV)
        - 'Frequency': Mean spindle frequency per epoch/channel (Hz)
        - 'Density': Count of spindles per epoch/channel
    freq_sp : tuple of float, default=(12, 15)
        Spindles frequency range in Hz.
    freq_broad : tuple of float, default=(1, 30)
        Broad band frequency range in Hz.
    duration : tuple of float, default=(0.5, 2.5)
        Min/max spindle duration in seconds.
    min_distance : int, default=500
        Minimum distance between spindles in ms.
    thresh_rms : float, default=1.5
        RMS threshold (SD above mean).
    thresh_corr : float, default=0.65
        Correlation threshold.
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
    >>> # Get spindle duration per epoch/channel
    >>> marker = SpindlesDetection(feature="Duration")

    >>> # Get spindle density with channel aggregation
    >>> marker = SpindlesDetection(
    ...     feature="Density",
    ...     channel_method="mean"
    ... )

    >>> # Get spindle amplitude with full aggregation
    >>> marker = SpindlesDetection(
    ...     feature="Amplitude",
    ...     channel_method="mean",
    ...     trial_method="mean"
    ... )
    """

    _DEPENDENCIES: ClassVar = {"mne", "yasa", "pandas", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"spindlesdetection": "timeseries"}
    }

    def __init__(
        self,
        feature: str,
        freq_sp: Tuple[float, float] = (12, 15),
        freq_broad: Tuple[float, float] = (1, 30),
        duration: Tuple[float, float] = (0.5, 2.5),
        min_distance: int = 500,
        thresh_rms: float = 1.5,
        thresh_corr: float = 0.65,
        reference_channels: Tuple[str, ...] = ("TP7", "TP8"),
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SpindlesDetection marker."""
        super().__init__(
            tmin=None, tmax=None, equipment=equipment, on=on, name=name
        )

        # Validate feature parameter - MANDATORY
        if feature not in SPINDLE_FEATURES:
            raise ValueError(
                f"'feature' parameter is MANDATORY and must be one of: "
                f"{SPINDLE_FEATURES}. Got: '{feature}'"
            )

        self.feature = feature
        self.freq_sp = tuple(freq_sp)
        self.freq_broad = tuple(freq_broad)
        self.duration = tuple(duration)
        self.min_distance = min_distance
        self.thresh_rms = thresh_rms
        self.thresh_corr = thresh_corr
        self.reference_channels = tuple(reference_channels)
        self.channel_method = channel_method
        self.trial_method = trial_method

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute spindle detection and return requested feature.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed spindle feature with keys:
            - 'data': array of shape (n_epochs, n_channels)
            - 'col_names': channel names
        """
        logger.debug(f"Computing spindles detection (feature={self.feature})")

        # Get and validate epochs
        data_obj = input["data"]
        self._validate_input(data_obj)

        # Filter to EEG channels only
        data_obj, _, _ = filter_to_eeg_channels(data_obj)
        ch_names = list(data_obj.ch_names)

        # Compute all features using singleton (with caching)
        spindles_base = SpindlesDetectionBase()
        all_features = spindles_base.compute(
            data_obj,
            self.freq_sp,
            self.freq_broad,
            self.duration,
            self.min_distance,
            self.thresh_rms,
            self.thresh_corr,
            self.reference_channels,
        )

        # Extract the requested feature
        output_data = all_features[self.feature]

        # Apply aggregation while preserving 2D structure
        # Use trial→channel order (matches original NICE behavior)
        output_data = apply_aggregation_preserve_dims(
            output_data,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
            aggregation_order="trial_channel",
        )

        # Use centralized format_marker_result
        return format_marker_result(
            feature_name="spindlesdetection",
            data=output_data,  # Shape: (n_epochs, n_channels) with size-1 for aggregated dims
            col_names=ch_names,
            channel_aggregated=(self.channel_method is not None),
        )

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type - always returns timeseries for 2D tensor data.

        Always returns 'timeseries' since we now always return
        2D tensors with shape (n_epochs, n_channels)
        where dimensions can be size 1 when aggregated.
        """
        return "timeseries"
