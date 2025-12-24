"""Contingent Negative Variation marker for junifer_eeg."""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_aggregation_preserve_dims, get_data_for_rois


@register_marker
class ContingentNegativeVariation(BaseMarker):
    """Contingent Negative Variation (CNV) marker.

    This marker computes the CNV by fitting a linear regression to the time
    course of each channel, representing the slow negative drift in the EEG
    signal. The CNV is quantified as the slope of the linear regression.

    Adapted from the NICE package implementation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scipy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {
            "cnvslope": "timeseries",
            "cnvintercept": "timeseries",
        },  # 2D: (epochs, channels)
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        rois: Union[List[str], List[int], None] = None,
        channel_method: str | None = None,
        trial_method: str | None = None,
        equipment: str = "egi256",
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the ContingentNegativeVariation marker.

        Parameters
        ----------
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE computation.
            Each item can be:
            - int: channel index (e.g., 6, 7, 14)
            - str: channel name (e.g., 'E7', 'E15') OR semantic ROI (e.g., 'cnv_roi')

            Examples:
            - [6, 7, 14, 15, 16, 22, 23] - NICE CNV ROI via indices (EGI/256)
            - ['cnv_roi'] - Semantic task-specific ROI

            **NOTE:** NICE uses task-specific 'cnv_roi', NOT 'scalp' ROI.
        channel_method : str, optional
            Methods to aggregate across channels: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        trial_method : str, optional
            Methods to aggregate across epochs: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        equipment : str, default="egi256"
            Equipment configuration for ROI resolution.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method
        self.equipment = equipment
        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 2D tensor data (no aggregation)
        - 'vector': 1D array (one aggregation applied)
        - 'scalar_table': scalar value (both aggregations applied)
        """
        if self.channel_method is None and self.trial_method is None:
            return "timeseries"

        if self.channel_method is not None and self.trial_method is not None:
            return "scalar_table"

        return "vector"

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Contingent Negative Variation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed CNV slope and intercept features.

        Raises
        ------
        ValueError
            If input data is not Epochs or if epochs are empty.
        """
        from mne.utils import _time_mask
        from scipy import linalg

        # Get the MNE data object - must be Epochs
        data_obj = input["data"]

        if not hasattr(data_obj, "events"):
            raise ValueError(
                "ContingentNegativeVariation requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(data_obj) == 0:
            raise ValueError("Cannot compute CNV on empty epochs.")

        epochs_data = (
            data_obj.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)
        ch_names = list(data_obj.ch_names)
        times = data_obj.times

        # Apply time cropping if specified
        if self.tmin is not None or self.tmax is not None:
            time_mask = _time_mask(times, self.tmin, self.tmax)
            epochs_data = epochs_data[:, :, time_mask]
            times = times[time_mask]

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Apply ROI filtering BEFORE computation if specified
        if self.rois is not None:
            data_transposed = epochs_data.transpose(1, 0, 2)
            roi_data_dict = get_data_for_rois(
                data_transposed,
                ch_names,
                self.rois,
                self.equipment,
            )
            if "selected_channels" not in roi_data_dict:
                raise ValueError(f"ROI filtering failed for rois: {self.rois}")
            epochs_data = roi_data_dict["selected_channels"].transpose(1, 0, 2)
            n_epochs, n_channels, n_samples = epochs_data.shape
            ch_names = self.rois

        # Get scaling factors using MNE's defaults
        from mne._fiff.pick import _picks_by_type, pick_info
        from mne.defaults import _handle_default

        scales = np.zeros(n_channels)
        picks = np.arange(n_channels)
        info_ = pick_info(data_obj.info, picks)
        for this_type, this_picks in _picks_by_type(info_):
            scales[this_picks] = _handle_default("scalings")[this_type]

        # Compute CNV slopes and intercepts for each channel and epoch
        slope_values = np.zeros((n_epochs, n_channels), dtype=np.float64)
        intercept_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        tmin = self.tmin if self.tmin is not None else times[0]
        tmax = self.tmax if self.tmax is not None else times[-1]
        fit_range = np.where(_time_mask(times, tmin, tmax))[0]

        if len(fit_range) < 2:
            raise ValueError(
                f"Time range [{tmin}, {tmax}] is too short for CNV computation. "
                f"Need at least 2 time points, got {len(fit_range)}."
            )

        # Design matrix: intercept + increasing time
        design_matrix = np.c_[
            np.ones(len(fit_range)),
            times[fit_range] - tmin,
        ]

        for epoch_idx in range(n_epochs):
            epoch_data = epochs_data[
                epoch_idx
            ]  # Shape: (n_channels, n_samples)

            for ch in range(n_channels):
                y = epoch_data[ch, fit_range] * scales[ch]
                betas, _, _, _ = linalg.lstsq(a=design_matrix, b=y)
                intercept_values[epoch_idx, ch] = betas[0]
                slope_values[epoch_idx, ch] = betas[1]

        # Apply aggregation
        slope_output = self._aggregate_output(slope_values, ch_names, "slope")
        intercept_output = self._aggregate_output(
            intercept_values, ch_names, "intercept"
        )

        return {
            "cnvslope": slope_output,
            "cnvintercept": intercept_output,
        }

    def _aggregate_output(
        self, data: np.ndarray, ch_names: List[str], prefix: str
    ) -> dict[str, Any]:
        """Aggregate CNV data (slope or intercept) while preserving dimensions.

        Always returns a 2D tensor with shape (n_epochs, n_channels).
        Even when dimensions have size 1, they are preserved for consistency.

        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_epochs, n_channels).
        ch_names : list of str
            Channel names.
        prefix : str
            Prefix for column names.

        Returns
        -------
        dict
            Dictionary with 'data' and 'col_names' keys.
        """
        # Use centralized aggregation function
        result_data = apply_aggregation_preserve_dims(
            data,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
        )

        # Generate column names
        if self.channel_method is not None and self.trial_method is not None:
            col_names = [f"{prefix}_all_channels_all_trials"]
        elif self.channel_method is not None:
            n_trials = result_data.shape[0] if result_data.ndim >= 1 else 1
            col_names = [f"{prefix}_trial_{i}" for i in range(n_trials)]
        elif self.trial_method is not None:
            col_names = [f"{prefix}_{ch}" for ch in ch_names]
        else:
            col_names = [f"{prefix}_{ch}" for ch in ch_names]

        return {"data": result_data, "col_names": col_names}
