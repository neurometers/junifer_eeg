"""Permutation entropy at multiple temporal scales with ROI aggregation."""

from typing import Any, ClassVar, Optional

from junifer.api.decorators import register_marker

from ..base import EEGEpochsMarker
from ..eeg_roi_aggregation import EEGROIAggregation
from .permutation_entropy_bands import PermutationEntropy

__all__ = ["PermutationEntropyROIs"]


@register_marker
class PermutationEntropyROIs(EEGEpochsMarker):
    """Permutation entropy at multiple temporal scales with ROI/channel/trial aggregation.

    Combines PermutationEntropy with EEGROIAggregation for a complete
    pipeline: PE computation → band extraction → ROI filtering → aggregation.

    Parameters
    ----------
    taus : int or list of int, default=8
        Time delay(s) for ordinal patterns. Different tau values represent
        different temporal scales. Can be single int or list for multiple scales.
    rois : list of dict, optional
        ROI definitions for aggregation AFTER computation.
        Each dict can have 'name' and 'channels' keys.
        If None, aggregates across all channels.
    channel_method : str, optional
        Method for channel aggregation ('mean', 'median', 'std', etc.).
    channel_method_params : dict, optional
        Parameters for channel aggregation method.
    trial_method : str, optional
        Method for trial aggregation ('mean', 'median', 'std', etc.).
    trial_method_params : dict, optional
        Parameters for trial aggregation method.
    kernel : int, default=3
        Length of ordinal patterns.
    tmin : float, optional
        Start time. If None, use epoch start.
    tmax : float, optional
        End time. If None, use epoch end.
    equipment : str, default="egi256"
    on : str, optional
        Data type (default "EEG").
    name : str, optional
        Marker name.

    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scipy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"permutationentropy": "vector"}
    }

    def __init__(
        self,
        taus: int | list[int] = 8,
        rois: Optional[list[dict]] = None,
        channel_method: Optional[str] = None,
        channel_method_params: Optional[dict] = None,
        trial_method: Optional[str] = None,
        trial_method_params: Optional[dict] = None,
        kernel: int = 3,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize PE with ROI aggregation."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        # Convert single tau to list
        self.taus = [taus] if isinstance(taus, int) else list(taus)
        self.rois = rois
        self.channel_method = channel_method
        self.channel_method_params = channel_method_params
        self.trial_method = trial_method
        self.trial_method_params = trial_method_params
        self.kernel = kernel

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute permutation entropy at multiple temporal scales with ROI aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Aggregated permutation entropy.

        """

        # Step 1: Compute PE at multiple temporal scales using PermutationEntropy
        pe_marker = PermutationEntropy(
            taus=self.taus,
            rois=self.rois,  # Pass rois for filtering BEFORE computation
            kernel=self.kernel,
            tmin=self.tmin,
            tmax=self.tmax,
            equipment=self.equipment,
            on=self._on,
        )
        pe_result = pe_marker.compute(input, extra_input)

        # Step 2: Apply ROI aggregation
        aggregation_input = dict(input.items())
        aggregation_input["data"] = pe_result["permutationentropy"]["data"]
        aggregation_input["col_names"] = pe_result["permutationentropy"].get(
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
        return {"permutationentropy": aggregation_result["aggregation"]}
