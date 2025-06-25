"""Contingent Negative Variation marker for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


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
        "EEG": {"cnv_slope": "vector", "cnv_intercept": "vector"}
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
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
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
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

        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_times)
        n_channels, n_times = data.shape

        # Set time range
        tmax = self.tmax if self.tmax is not None else raw.times[-1]
        tmin = self.tmin if self.tmin is not None else raw.times[0]

        # Get time mask
        fit_range = np.where(_time_mask(raw.times, tmin, tmax))[0]

        if len(fit_range) < 2:
            raise ValueError("Time range too short for linear regression")

        # Design matrix: intercept + increasing time
        design_matrix = np.c_[
            np.ones(len(fit_range)), raw.times[fit_range] - tmin
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

        # Initialize output arrays
        slopes = np.zeros(n_channels)
        intercepts = np.zeros(n_channels)

        # Estimate single trial regression over time samples
        for ch in range(n_channels):
            y = data[ch, fit_range] * scales[ch]  # Apply scaling

            try:
                betas, _, _, _ = linalg.lstsq(a=design_matrix, b=y)
                intercepts[ch] = betas[0]
                slopes[ch] = betas[1]
            except linalg.LinAlgError:
                # Handle singular matrix case
                intercepts[ch] = np.nan
                slopes[ch] = np.nan

        # Return data in junifer format
        return {
            "cnv_slope": {
                "data": slopes.reshape(1, -1),  # Shape: (1, n_channels)
                "col_names": [f"{ch}_cnv_slope" for ch in raw.ch_names],
            },
            "cnv_intercept": {
                "data": intercepts.reshape(1, -1),  # Shape: (1, n_channels)
                "col_names": [f"{ch}_cnv_intercept" for ch in raw.ch_names],
            },
        }
