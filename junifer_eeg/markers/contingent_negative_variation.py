"""Contingent Negative Variation marker for junifer_eeg."""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import aggregate_data, get_data_for_rois


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
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
        equipment: str = "egi256",
        epoch_length: float = 2.0,
        overlap: float = 0.0,
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
        channel_aggregation_method : str, optional
            Methods to aggregate across channels: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        trial_aggregation_method : str, optional
            Methods to aggregate across epochs: 'mean', 'std', 'median',
            'trim_mean80', 'trim_mean90', etc.
        epoch_length : float, default=2.0
            Length of epochs in seconds for trial aggregation.
        overlap : float, default=0.0
            Overlap between epochs (0.0 to 0.9).
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.rois = rois
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        self.epoch_length = epoch_length
        self.overlap = overlap
        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 2D tensor data (no aggregation)
        - 'vector': 1D array (one aggregation applied)
        - 'scalar_table': scalar value (both aggregations applied)
        """
        # No aggregation → 2D tensor (epochs, channels) → use timeseries
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            return "timeseries"

        # Both aggregations → scalar → use scalar_table
        if (
            self.channel_aggregation_method is not None
            and self.trial_aggregation_method is not None
        ):
            return "scalar_table"

        # One aggregation → 1D array → use vector
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
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed CNV slope and intercept features.
        """
        from mne.utils import _time_mask
        from scipy import linalg

        # Get the MNE data object (can be Raw or Epochs)
        data_obj = input["data"]

        # Check if we have Epochs or Raw data
        if hasattr(data_obj, "get_data") and hasattr(data_obj, "events"):
            # This is Epochs data
            # Check for empty epochs first
            if len(data_obj) == 0:
                # Return empty results for empty epochs
                return {
                    "cnvslope": {"data": np.array([[]]), "col_names": []},
                    "cnvintercept": {"data": np.array([[]]), "col_names": []},
                }

            epochs_data = (
                data_obj.get_data()
            )  # Shape: (n_epochs, n_channels, n_times)

            # Apply time cropping if specified
            if self.tmin is not None or self.tmax is not None:
                from mne.utils import _time_mask

                time_mask = _time_mask(data_obj.times, self.tmin, self.tmax)
                epochs_data = epochs_data[:, :, time_mask]

        else:
            # This is Raw data - create epochs from continuous data if needed
            raw = data_obj
            if self.trial_aggregation_method is not None:
                epochs_data = self._create_epochs_from_continuous(raw)
                # epochs_data shape: (n_epochs, n_channels, n_samples)
            else:
                # Single "epoch" from continuous data
                data = raw.get_data()  # Shape: (n_channels, n_times)

                # Check if data length matches times length to avoid indexing errors
                current_times = raw.times
                if len(current_times) != data.shape[1]:
                    # Data has been preprocessed and time dimension reduced
                    # Create new times array matching the data
                    sfreq = raw.info["sfreq"]
                    n_samples = data.shape[1]
                    current_times = np.arange(n_samples) / sfreq + raw.times[0]

                if self.tmin is not None or self.tmax is not None:
                    time_mask = _time_mask(current_times, self.tmin, self.tmax)
                    data = data[:, time_mask]
                    current_times = current_times[time_mask]
                epochs_data = data[
                    np.newaxis,
                    :,
                    :,
                ]  # Shape: (1, n_channels, n_samples)

        # Ensure epochs_data has correct 3D shape
        if epochs_data.ndim == 2:
            epochs_data = epochs_data[np.newaxis, :, :]

        n_epochs, n_channels, n_samples = epochs_data.shape
        ch_names = (
            list(data_obj.ch_names)
            if hasattr(data_obj, "ch_names")
            else [f"ch_{i}" for i in range(n_channels)]
        )

        # Apply ROI filtering BEFORE computation if specified
        if self.rois is not None:
            # Transpose to (n_channels, n_epochs, n_samples)
            data_transposed = epochs_data.transpose(1, 0, 2)

            # Get ROI-filtered data
            roi_data_dict = get_data_for_rois(
                data_transposed,
                ch_names,
                self.rois,
                self.equipment,
            )

            # Extract the filtered data (returns {"selected_channels": data})
            if "selected_channels" in roi_data_dict:
                data_filtered = roi_data_dict["selected_channels"]
                # Transpose back to (n_epochs, n_channels, n_samples)
                epochs_data = data_filtered.transpose(1, 0, 2)
                n_epochs, n_channels, n_samples = epochs_data.shape
                # Preserve the actual ROI channel names (self.rois contains the channel names we filtered to)
                ch_names = self.rois

        # Compute CNV slopes and intercepts for each channel and epoch
        slope_values = np.zeros((n_epochs, n_channels), dtype=np.float64)
        intercept_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            epoch_data = epochs_data[
                epoch_idx
            ]  # Shape: (n_channels, n_samples)
            n_times = epoch_data.shape[1]

            # Use appropriate time array based on data type
            if hasattr(data_obj, "times"):
                # For Epochs data, times are already available
                if self.tmin is not None or self.tmax is not None:
                    # If we cropped the data, create times for the cropped data
                    if hasattr(data_obj, "events"):  # Epochs
                        original_times = data_obj.times
                        time_mask = _time_mask(
                            original_times, self.tmin, self.tmax
                        )
                        times = original_times[time_mask]
                    else:  # Raw
                        times = (
                            current_times
                            if "current_times" in locals()
                            else np.arange(n_times) / data_obj.info["sfreq"]
                        )
                else:
                    times = data_obj.times[:n_times]  # Use original times
            else:
                # Fallback: create time array
                sfreq = (
                    data_obj.info["sfreq"]
                    if hasattr(data_obj, "info")
                    else 250.0
                )
                times = np.arange(n_times) / sfreq

            # Set time range for this epoch
            tmax = self.tmax if self.tmax is not None else times[-1]
            tmin = self.tmin if self.tmin is not None else times[0]

            # Get time mask for this epoch
            fit_range = np.where(_time_mask(times, tmin, tmax))[0]

            if len(fit_range) < 2:
                # Return NaN for too short segments
                slope_values[epoch_idx, :] = np.nan
                intercept_values[epoch_idx, :] = np.nan
                continue

            # Design matrix: intercept + increasing time
            design_matrix = np.c_[
                np.ones(len(fit_range)),
                times[fit_range] - tmin,
            ]

            # Get scaling factors using MNE's defaults (match NICE implementation)
            from mne._fiff.pick import _picks_by_type, pick_info
            from mne.defaults import _handle_default

            scales = np.zeros(n_channels)

            # Get all picks and compute scales by channel type
            if hasattr(data_obj, "info"):
                picks = np.arange(n_channels)
                info_ = pick_info(data_obj.info, picks)
                for this_type, this_picks in _picks_by_type(info_):
                    scales[this_picks] = _handle_default("scalings")[this_type]
            else:
                # Default scaling for EEG (1e-6 to convert V to µV)
                scales[:] = 1e-6

            # Estimate single trial regression over time samples
            for ch in range(n_channels):
                y = epoch_data[ch, fit_range] * scales[ch]  # Apply scaling

                try:
                    betas, _, _, _ = linalg.lstsq(a=design_matrix, b=y)
                    intercept_values[epoch_idx, ch] = betas[
                        0
                    ]  # intercept is first coefficient
                    slope_values[epoch_idx, ch] = betas[
                        1
                    ]  # slope is second coefficient
                except linalg.LinAlgError:
                    # Handle singular matrix case
                    intercept_values[epoch_idx, ch] = np.nan
                    slope_values[epoch_idx, ch] = np.nan

        # Helper function to aggregate CNV data (slope or intercept)
        def aggregate_cnv_output(data, ch_names, prefix):
            # Apply ROI filtering if specified
            if self.rois is not None:
                roi_data = get_data_for_rois(
                    data.T, list(ch_names), self.rois, self.equipment
                )
                if "selected_channels" in roi_data:
                    data = roi_data["selected_channels"].T
                    ch_names = self.rois

            # Check for no aggregation
            if (
                self.channel_aggregation_method is None
                and self.trial_aggregation_method is None
            ):
                col_names = [f"{prefix}_{ch}" for ch in ch_names]
                return {"data": data, "col_names": col_names}

            # Apply aggregation
            result_data = data

            # Channel aggregation
            if self.channel_aggregation_method is not None:
                result_data = aggregate_data(
                    result_data, self.channel_aggregation_method, axis=1
                )

            # Trial aggregation
            if self.trial_aggregation_method is not None:
                if result_data.ndim == 1:
                    result_data = aggregate_data(
                        result_data, self.trial_aggregation_method, axis=None
                    )
                else:
                    result_data = aggregate_data(
                        result_data, self.trial_aggregation_method, axis=0
                    )

            # Return result data without unnecessary reshaping - preserve tensor structure
            # Scalar: keep as scalar
            # 1D array: keep as 1D (n_trials) or (n_channels)
            # 2D array: keep as 2D (n_trials, n_channels)

            # Generate column names based on result shape and aggregation
            if (
                self.channel_aggregation_method is not None
                and self.trial_aggregation_method is not None
            ):
                col_names = [f"{prefix}_all_channels_all_trials"]
            elif self.channel_aggregation_method is not None:
                # result_data shape: (n_trials,) after channel aggregation
                n_trials = result_data.shape[0] if result_data.ndim >= 1 else 1
                col_names = [f"{prefix}_trial_{i}" for i in range(n_trials)]
            elif self.trial_aggregation_method is not None:
                # result_data shape: (n_channels,) after trial aggregation
                col_names = [f"{prefix}_{ch}" for ch in ch_names]
            else:
                col_names = [f"{prefix}_{ch}" for ch in ch_names]

            return {"data": result_data, "col_names": col_names}

        # Apply aggregation for slopes and intercepts
        slope_output = aggregate_cnv_output(slope_values, ch_names, "slope")
        intercept_output = aggregate_cnv_output(
            intercept_values, ch_names, "intercept"
        )

        # Combine results
        return {
            "cnvslope": slope_output,
            "cnvintercept": intercept_output,
        }

    def _create_epochs_from_continuous(self, raw):
        """Create epochs from continuous data."""
        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_times)

        # Apply time mask if specified
        if self.tmin is not None or self.tmax is not None:
            from mne.utils import _time_mask

            # Check if data length matches times length to avoid indexing errors
            current_times = raw.times
            if len(current_times) != data.shape[1]:
                # Data has been preprocessed and time dimension reduced
                # Create new times array matching the data
                sfreq = raw.info["sfreq"]
                n_samples = data.shape[1]
                current_times = np.arange(n_samples) / sfreq + raw.times[0]

            time_mask = _time_mask(current_times, self.tmin, self.tmax)
            data = data[:, time_mask]

        n_channels, n_samples = data.shape
        sfreq = raw.info["sfreq"]

        # Calculate epoch parameters
        epoch_samples = int(self.epoch_length * sfreq)
        overlap_samples = int(self.overlap * epoch_samples)
        step_samples = epoch_samples - overlap_samples

        # Calculate number of epochs
        n_epochs = max(1, (n_samples - epoch_samples) // step_samples + 1)

        # Create epochs
        epochs_data = np.zeros((n_epochs, n_channels, epoch_samples))

        for epoch_idx in range(n_epochs):
            start_sample = epoch_idx * step_samples
            end_sample = start_sample + epoch_samples

            if end_sample <= n_samples:
                epochs_data[epoch_idx] = data[:, start_sample:end_sample]
            else:
                # Pad with last available samples if needed
                available_samples = n_samples - start_sample
                epochs_data[epoch_idx, :, :available_samples] = data[
                    :,
                    start_sample:,
                ]
                # Pad with zeros or repeat last sample
                epochs_data[epoch_idx, :, available_samples:] = data[:, -1:]

        return epochs_data
