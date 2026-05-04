"""Slow waves detection marker implementation.

Structure (2-file, matching kolmogorov_complexity_new):
- _slow_waves_detection_base.py: Singleton with caching + compute logic
- slow_waves_detection.py: Main marker extending EEGEpochsMarker (this file)
"""

from typing import Any, ClassVar, Literal, Optional, Tuple

from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import apply_aggregation_preserve_dims, filter_to_eeg_channels
from ._slow_waves_detection_base import (
    SLOW_WAVE_FEATURES,
    SlowWavesDetectionBase,
)

__all__ = ["SlowWavesDetection"]


@register_marker
class SlowWavesDetection(EEGEpochsMarker):
    """Sleep slow waves detection marker using YASA or a custom detector.

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
    detection_method : {"yasa", "custom"}, default="yasa"
        Detection backend to use. ``"yasa"`` delegates detection to
        :func:`yasa.sw_detect`. ``"custom"`` uses a zero-crossing detector
        inspired by Andrillon et al. 2021.
    freq_sw : tuple of float, default=(0.3, 1.5)
        Slow wave frequency range in Hz. For ``detection_method="yasa"``
        this is passed directly to YASA. For ``detection_method="custom"``,
        it defines the band-pass used before zero-crossing detection.
    amp_ptp_initial : float, default=15.0
        Initial peak-to-peak amplitude threshold in µV.
    freq_threshold : float, default=7.0
        Maximum frequency threshold for filtering (Hz).
    artifact_threshold : float, default=75.0
        Positive peak amplitude threshold for artifact removal (µV).
    ptp_threshold_mode : {"adaptive", "fixed"}, default="adaptive"
        Strategy used after detection for peak-to-peak filtering. ``"adaptive"``
        computes one threshold per channel. ``"fixed"`` skips this percentile
        filtering stage.
    ptp_percentile : float, default=90.0
        Percentile used when ``ptp_threshold_mode="adaptive"`` and no
        precomputed thresholds file is provided.
    max_ptp_amplitude : float, default=150.0
        Maximum peak-to-peak amplitude allowed during post-detection filtering.
    ptp_thresholds_path : str, optional
        Optional CSV with fixed per-subject, per-channel PTP thresholds.
        When provided, adaptive percentile computation is skipped and the CSV
        thresholds are applied instead.
    ptp_thresholds_strict : bool, default=True
        Behaviour when ``ptp_thresholds_path`` is set and the current
        subject is not present in the CSV. ``True`` raises (preserves
        previous behaviour). ``False`` logs a warning and emits empty
        slow-wave results for the element so other markers in the same
        MarkerCollection can still complete.
    proximity_amplitude : float, optional
        Opt-in (Pinggal 2022 / Andrillon 2021): if set, drop any wave whose
        center falls within ``proximity_window`` seconds of a sample where
        ``|filtered_signal| > proximity_amplitude`` (µV) on the same
        epoch/channel. ``None`` (default) disables the rule.
    proximity_window : float, default=1.0
        Half-window in seconds for the proximity rule above. Only used when
        ``proximity_amplitude`` is set.
    filter_design : {"chebyshev2", "fir"}, default="chebyshev2"
        Bandpass filter used by the custom detector and by the proximity
        reference signal. ``"chebyshev2"`` keeps the previous Chebyshev II
        IIR (default). ``"fir"`` uses an MNE zero-phase FIR (Le Coz 2025).
    slope_uv_per_s_range : tuple of float, optional
        Opt-in (Le Coz 2025): when set, drop any wave whose
        ``AscendingSlope`` or ``DescendingSlope`` (µV/s, always emitted)
        falls outside ``[lo, hi]``. ``None`` (default) disables the rule.
    downsample_to : float, optional
        Opt-in (Le Coz 2025): if set, resample the epochs to this sampling
        frequency (Hz) before detection. ``None`` (default) leaves the
        sampling frequency unchanged.
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

    _DEPENDENCIES: ClassVar = {"mne", "yasa", "pandas", "numpy", "scipy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"slowwavesdetection": "timeseries"}
    }

    def __init__(
        self,
        feature: str,
        detection_method: Literal["yasa", "custom"] = "yasa",
        freq_sw: Tuple[float, float] = (0.3, 1.5),
        amp_ptp_initial: float = 15.0,
        freq_threshold: float = 7.0,
        artifact_threshold: float = 75.0,
        ptp_threshold_mode: Literal["adaptive", "fixed"] = "adaptive",
        ptp_percentile: float = 90.0,
        max_ptp_amplitude: float = 150.0,
        ptp_thresholds_path: Optional[str] = None,
        ptp_thresholds_strict: bool = True,
        proximity_amplitude: Optional[float] = None,
        proximity_window: float = 1.0,
        filter_design: Literal["chebyshev2", "fir"] = "chebyshev2",
        slope_uv_per_s_range: Optional[Tuple[float, float]] = None,
        downsample_to: Optional[float] = None,
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

        if detection_method not in {"yasa", "custom"}:
            raise ValueError(
                "'detection_method' must be 'yasa' or 'custom'. "
                f"Got: '{detection_method}'"
            )
        if ptp_threshold_mode not in {"adaptive", "fixed"}:
            raise ValueError(
                "'ptp_threshold_mode' must be 'adaptive' or 'fixed'. "
                f"Got: '{ptp_threshold_mode}'"
            )
        if filter_design not in {"chebyshev2", "fir"}:
            raise ValueError(
                "'filter_design' must be 'chebyshev2' or 'fir'. "
                f"Got: '{filter_design}'"
            )
        if slope_uv_per_s_range is not None:
            lo, hi = slope_uv_per_s_range
            if not (lo < hi):
                raise ValueError(
                    "'slope_uv_per_s_range' must be (lo, hi) with lo < hi. "
                    f"Got: {slope_uv_per_s_range}"
                )

        self.feature = feature
        self.detection_method = detection_method
        self.freq_sw = tuple(freq_sw)
        self.amp_ptp_initial = amp_ptp_initial
        self.freq_threshold = freq_threshold
        self.artifact_threshold = artifact_threshold
        self.ptp_threshold_mode = ptp_threshold_mode
        self.ptp_percentile = ptp_percentile
        self.max_ptp_amplitude = max_ptp_amplitude
        self.ptp_thresholds_path = ptp_thresholds_path
        self.ptp_thresholds_strict = ptp_thresholds_strict
        self.proximity_amplitude = proximity_amplitude
        self.proximity_window = proximity_window
        self.filter_design = filter_design
        self.slope_uv_per_s_range = (
            tuple(slope_uv_per_s_range)
            if slope_uv_per_s_range is not None
            else None
        )
        self.downsample_to = downsample_to
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
        subject_id = self._extract_subject_id(extra_input)

        # Compute all features using singleton (with caching)
        sw_base = SlowWavesDetectionBase()
        all_features = sw_base.compute(
            data_obj,
            self.detection_method,
            self.freq_sw,
            self.amp_ptp_initial,
            self.freq_threshold,
            self.artifact_threshold,
            self.ptp_threshold_mode,
            self.ptp_percentile,
            self.max_ptp_amplitude,
            self.ptp_thresholds_path,
            subject_id,
            self.reference_channels,
            proximity_amplitude=self.proximity_amplitude,
            proximity_window=self.proximity_window,
            ptp_thresholds_strict=self.ptp_thresholds_strict,
            filter_design=self.filter_design,
            slope_uv_per_s_range=self.slope_uv_per_s_range,
            downsample_to=self.downsample_to,
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
            feature_name="slowwavesdetection",
            data=output_data,  # Shape: (n_epochs, n_channels) with size-1 for aggregated dims
            col_names=ch_names,
            channel_aggregated=(self.channel_method is not None),
        )

    @staticmethod
    def _extract_subject_id(
        extra_input: Optional[dict[str, Any]],
    ) -> Optional[str]:
        """Extract subject metadata from Junifer's extra_input structure."""
        if not extra_input:
            return None

        element = extra_input.get("element")
        if isinstance(element, dict):
            for key in ("subject", "subject_id", "sub"):
                value = element.get(key)
                if value is not None:
                    return str(value)

        for key in ("subject", "subject_id", "sub"):
            value = extra_input.get(key)
            if value is not None:
                return str(value)

        return None

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type - always returns timeseries for 2D tensor data.

        Always returns 'timeseries' since we now always return
        2D tensors with shape (n_epochs, n_channels)
        where dimensions can be size 1 when aggregated.
        """
        return "timeseries"
