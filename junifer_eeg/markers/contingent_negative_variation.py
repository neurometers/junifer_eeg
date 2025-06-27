"""Contingent Negative Variation marker for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


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
        "EEG": {"cnvslope": "vector", "cnvintercept": "vector"}
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        rois: list[str] | None = None,
        roi_aggregation_method: list[str] | None = None,
        trial_aggregation_method: list[str] | None = None,
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
        rois : list of str, optional
            List of ROI names or electrode names to aggregate.
        roi_aggregation_method : list of str, optional
            List of aggregation methods for ROIs ('mean', 'std', 'median', 'min', 'max').
        trial_aggregation_method : list of str, optional
            List of aggregation methods for trials ('mean', 'std', 'median', 'min', 'max').
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
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.epoch_length = epoch_length
        self.overlap = overlap
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
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
        from mne._fiff.pick import _picks_by_type
        from mne.defaults import _handle_default
        from mne.utils import _time_mask
        from scipy import linalg

        # Get the MNE Raw object
        raw = input["data"]

        # Create epochs from continuous data if needed
        if self.trial_aggregation_method is not None:
            epochs_data = self._create_epochs_from_continuous(raw)
            # epochs_data shape: (n_epochs, n_channels, n_samples)
        else:
            # Single "epoch" from continuous data
            data = raw.get_data()  # Shape: (n_channels, n_times)
            if self.tmin is not None or self.tmax is not None:
                time_mask = _time_mask(raw.times, self.tmin, self.tmax)
                data = data[:, time_mask]
            epochs_data = data[
                np.newaxis, :, :
            ]  # Shape: (1, n_channels, n_samples)

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Compute CNV slopes and intercepts for each channel and epoch
        slope_values = np.zeros((n_epochs, n_channels), dtype=np.float64)
        intercept_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            epoch_data = epochs_data[
                epoch_idx
            ]  # Shape: (n_channels, n_samples)
            n_times = epoch_data.shape[1]
            times = np.arange(n_times) / raw.info["sfreq"]

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
                np.ones(len(fit_range)), times[fit_range] - tmin
            ]

            # Get scaling factors using MNE's defaults
            scales = np.ones(n_channels)
            try:
                for this_type, this_picks in _picks_by_type(raw.info):
                    if len(this_picks) > 0:
                        scale_factor = _handle_default("scalings").get(
                            this_type, 1.0
                        )
                        scales[this_picks] = scale_factor
            except (KeyError, AttributeError):
                # If scaling fails, use unity scaling
                pass

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

        # Handle ROI selection for slopes
        if self.rois is not None:
            slope_roi_data = get_data_for_rois(
                slope_values.T,  # Transpose to (n_channels, n_epochs)
                list(raw.ch_names),
                self.rois,
            )
            intercept_roi_data = get_data_for_rois(
                intercept_values.T,  # Transpose to (n_channels, n_epochs)
                list(raw.ch_names),
                self.rois,
            )
        else:
            # Use all channels as individual ROIs
            slope_roi_data = {
                ch: slope_values[:, i : i + 1].T
                for i, ch in enumerate(raw.ch_names)
            }
            intercept_roi_data = {
                ch: intercept_values[:, i : i + 1].T
                for i, ch in enumerate(raw.ch_names)
            }

        # Apply aggregation for slopes
        slope_results = apply_roi_trial_aggregation(
            slope_roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="cnvslope",
        )

        # Apply aggregation for intercepts
        intercept_results = apply_roi_trial_aggregation(
            intercept_roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="cnvintercept",
        )

        # Combine the results
        result = {}
        result.update(slope_results)
        result.update(intercept_results)

        return result

    def _create_epochs_from_continuous(self, raw):
        """Create epochs from continuous data."""
        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_times)

        # Apply time mask if specified
        if self.tmin is not None or self.tmax is not None:
            from mne.utils import _time_mask

            time_mask = _time_mask(raw.times, self.tmin, self.tmax)
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
                    :, start_sample:
                ]
                # Pad with zeros or repeat last sample
                epochs_data[epoch_idx, :, available_samples:] = data[:, -1:]

        return epochs_data
