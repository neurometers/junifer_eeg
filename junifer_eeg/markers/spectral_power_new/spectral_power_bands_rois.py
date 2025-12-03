"""Spectral power with frequency bands and ROI aggregation."""

from typing import Any, ClassVar, Optional, Union

from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker
from ..eeg_roi_aggregation import EEGROIAggregation
from .spectral_power_bands import SpectralPowerBands

__all__ = ["SpectralPowerBandsROIs"]


@register_marker
class SpectralPowerBandsROIs(EEGEpochsMarker):
    """Spectral power with frequency bands and ROI aggregation.

    Combines SpectralPowerBands with EEGROIAggregation for a complete
    pipeline: PSD computation → band extraction → ROI filtering → aggregation.

    Parameters
    ----------
    bands : dict, optional
        Frequency bands as {name: (fmin, fmax)}. If None, uses standard bands.
    rois : list of str or int, optional
        Channel specifications for filtering. If None, uses all channels.
    channel_method : str, optional
        Channel aggregation method: 'mean', 'median', 'trim_mean80', etc.
        If None, no channel aggregation.
    channel_method_params : dict, optional
        Parameters for channel aggregation.
    trial_method : str, optional
        Trial aggregation method. If None, no trial aggregation.
    trial_method_params : dict, optional
        Parameters for trial aggregation.
    normalize : bool, default=False
        If True, normalize power by total power.
    dB : bool, default=True
        If True, convert to decibels.
    entropy : bool, default=False
        If True, compute spectral entropy. Requires normalize=True.
    n_fft : int, optional
        FFT length. If None, adaptive.
    n_per_seg : int, optional
        Segment length. If None, adaptive.
    n_overlap : int, optional
        Overlap points. If None, adaptive.
    db_threshold : float, optional
        dB conversion threshold. If None, adaptive.
    tmin : float, optional
        Start time. If None, use epoch start.
    tmax : float, optional
        End time. If None, use epoch end.
    equipment : str, default="egi256"
        Equipment configuration.
    on : str, optional
        Data type (default "EEG").
    name : str, optional
        Marker name.

    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"spectralpower": "vector"}}

    def __init__(
        self,
        bands: Optional[dict] = None,
        rois: Union[list[str], list[int], None] = None,
        channel_method: Optional[str] = None,
        channel_method_params: Optional[dict[str, Any]] = None,
        trial_method: Optional[str] = None,
        trial_method_params: Optional[dict[str, Any]] = None,
        normalize: bool = False,
        dB: bool = True,
        entropy: bool = False,
        n_fft: Optional[int] = None,
        n_per_seg: Optional[int] = None,
        n_overlap: Optional[int] = None,
        db_threshold: Optional[float] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize spectral power bands with ROIs marker."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        self.bands = bands
        self.rois = rois
        self.channel_method = channel_method
        self.channel_method_params = channel_method_params
        self.trial_method = trial_method
        self.trial_method_params = trial_method_params
        self.normalize = normalize
        self.dB = dB
        self.entropy = entropy
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.db_threshold = db_threshold

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute spectral power with ROI aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Aggregated spectral power.

        """
        logger.debug("Computing spectral power with ROI aggregation")

        # Step 1: Compute band powers using SpectralPowerBands
        bands_marker = SpectralPowerBands(
            bands=self.bands,
            rois=self.rois,  # Pass rois for filtering BEFORE computation
            normalize=self.normalize,
            dB=self.dB,
            entropy=self.entropy,
            n_fft=self.n_fft,
            n_per_seg=self.n_per_seg,
            n_overlap=self.n_overlap,
            db_threshold=self.db_threshold,
            tmin=self.tmin,
            tmax=self.tmax,
            equipment=self.equipment,
            on=self._on,
        )
        band_result = bands_marker.compute(input, extra_input)

        # Step 2: Apply ROI aggregation
        aggregation_input = dict(input.items())
        aggregation_input["data"] = band_result["spectralpower"]["data"]
        aggregation_input["col_names"] = band_result["spectralpower"].get(
            "col_names"
        )

        roi_aggregator = EEGROIAggregation(
            rois=self.rois,
            channel_method=self.channel_method,
            channel_method_params=self.channel_method_params,
            trial_method=self.trial_method,
            trial_method_params=self.trial_method_params,
            equipment=self.equipment,
            on=self._on,
        )

        aggregation_result = roi_aggregator.compute(
            aggregation_input, extra_input
        )

        # Step 3: Format output
        return {"spectralpower": aggregation_result["aggregation"]}
