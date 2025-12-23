"""Power Spectral Density Summary marker implementation."""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.api.decorators import register_marker

from ..base import format_marker_result
from ..utils import aggregate_data, get_data_for_rois
from ._psd_base import PowerSpectralDensityBase


@register_marker
class PowerSpectralDensitySummary(PowerSpectralDensityBase):
    """Power Spectral Density Summary using percentile statistics.

    This marker computes summary statistics (percentiles) of the power
    spectral density across frequencies, adapted from the NICE package.
    """

    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"psdsummary": "timeseries"}
    }  # 2D: (epochs, channels)

    def __init__(
        self,
        percentile: float = 0.5,
        rois: Union[List[str], List[int], None] = None,
        channel_method: str | None = None,
        trial_method: str | None = None,
        equipment: str = "egi256",
        **kwargs,
    ) -> None:
        """Initialize the PowerSpectralDensitySummary marker.

        Parameters
        ----------
        percentile : float, default=0.5
            Percentile to compute (0-1, where 0.5 = 50th percentile).
        rois : list of str or int, optional
            Flat list of channel specifications for filtering AFTER PSD computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp')

            If None, uses all channels.
        channel_method : str, optional
            Aggregation method for ROIs ('mean', 'std', 'median', 'min', 'max').
        trial_method : str, optional
            Aggregation method for trials ('mean', 'std', 'median', 'min', 'max').
        equipment : str, default="egi256"
            Equipment configuration for ROI resolution.
        **kwargs : dict
            Additional parameters passed to base class (tmin, tmax, fmin, fmax, etc.).
        """
        # Use percentile directly as fraction (0-1) to match original implementation
        self.percentile = percentile
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method
        self.equipment = equipment
        super().__init__(**kwargs)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 2D tensor data (no aggregation)
        - 'vector': 1D array (one aggregation applied)
        - 'scalar_table': scalar value (both aggregations applied)
        """
        # No aggregation → 2D tensor (epochs, channels) → use timeseries
        if self.channel_method is None and self.trial_method is None:
            return "timeseries"

        # Both aggregations → scalar → use scalar_table
        if self.channel_method is not None and self.trial_method is not None:
            return "scalar_table"

        # One aggregation → 1D array → use vector
        return "vector"

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Power Spectral Density Summary.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD summary statistics.

        Raises
        ------
        ValueError
            If input data is not Epochs or if epochs are empty.
        """
        # Get the MNE data object - must be Epochs
        data_obj = input["data"]

        if not hasattr(data_obj, "events"):
            raise ValueError(
                "PowerSpectralDensitySummary requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(data_obj) == 0:
            raise ValueError("Cannot compute PSD summary on empty epochs.")

        # Prepare and filter data
        data_obj, ch_names = self._prepare_data(input)

        # Compute PSD (preserve epochs for Summary)
        psd_data, freqs, ch_names = self._compute_psd(
            data_obj, preserve_epochs=True
        )  # Shape: (n_epochs, n_channels, n_freqs)

        # Compute SEF values
        psd_summary_values = self._compute_sef(psd_data, freqs)

        # Apply ROI filtering BEFORE aggregation if specified
        if self.rois is not None:
            roi_data_dict = get_data_for_rois(
                psd_summary_values.T,  # Transpose to (n_channels, n_epochs)
                ch_names,
                self.rois,
                self.equipment,
            )
            # Extract filtered data and transpose back to (n_epochs, n_channels)
            if "selected_channels" in roi_data_dict:
                psd_summary_values = roi_data_dict["selected_channels"].T
                ch_names = (
                    self.rois if isinstance(self.rois, list) else [self.rois]
                )

        # Apply aggregation
        results = self._apply_aggregation(psd_summary_values, ch_names)
        return {"psdsummary": results}

    def _compute_sef(self, psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Compute Spectral Edge Frequency (SEF) where cumulative power reaches percentile.

        Parameters
        ----------
        psd : np.ndarray
            PSD data with shape (n_epochs, n_channels, n_freqs)
        freqs : np.ndarray
            Frequency array

        Returns
        -------
        np.ndarray
            SEF values with shape (n_epochs, n_channels)
        """
        n_epochs, n_channels, n_freqs = psd.shape

        # Vectorized SEF/MSF computation
        # Compute Spectral Edge Frequency (SEF) where cumulative power reaches percentile
        if n_freqs > 1:
            # Compute cumulative power for all epochs and channels at once
            cum = np.cumsum(psd, axis=-1)  # (n_epochs, n_channels, n_freqs)
            total = cum[..., -1:]  # (n_epochs, n_channels, 1)

            # Threshold for percentile
            # percentile is already a fraction (0-1), not a percentage (0-100)
            thresh = total * self.percentile  # (n_epochs, n_channels, 1)

            # Find first index where cumulative power >= threshold
            ge = cum >= thresh  # (n_epochs, n_channels, n_freqs)
            idx = np.argmax(ge, axis=-1)  # (n_epochs, n_channels)

            # Handle edge cases
            none_true = ~np.any(ge, axis=-1)  # (n_epochs, n_channels)
            nonzero = total.squeeze(-1) > 0  # (n_epochs, n_channels)

            # Map indices to frequency values
            sef = freqs[idx]  # (n_epochs, n_channels)

            # Rows with no power -> minimum frequency
            sef[~nonzero] = freqs[0]

            # Rows with some power but percentile never reached -> last frequency
            sef[none_true & nonzero] = freqs[-1]

            psd_summary_values = sef  # (n_epochs, n_channels)
        else:
            # Single frequency bin - return the frequency value
            psd_summary_values = np.full(
                (n_epochs, n_channels), freqs[0] if len(freqs) > 0 else 0.0
            )

        return psd_summary_values

    def _apply_aggregation(
        self, psd_summary_values: np.ndarray, ch_names: List[str]
    ) -> dict:
        """Apply aggregation to the computed SEF values.

        Parameters
        ----------
        psd_summary_values : np.ndarray
            Computed SEF values with shape (n_epochs, n_channels)
        ch_names : List[str]
            Channel names

        Returns
        -------
        dict
            Results dictionary with aggregated data and column names
        """
        # SPECIAL HANDLING for PowerSpectralDensitySummary:
        # Clinical literature requires computing percentiles per channel, then aggregating
        # (not combining channels before computing percentiles)
        if self.channel_method is not None and self.trial_method is not None:
            # Both channel and trial aggregation: compute single scalar value

            # Step 1: Trial aggregation per channel
            if psd_summary_values.shape[0] > 1:  # Multiple epochs
                channel_values = np.array(
                    [
                        aggregate_data(
                            psd_summary_values[:, ch_idx],
                            self.trial_method,
                        )
                        for ch_idx in range(psd_summary_values.shape[1])
                    ]
                )
            else:
                # Single epoch: use values directly
                channel_values = psd_summary_values[0, :]

            # Step 2: Channel aggregation across all channels
            final_value = aggregate_data(channel_values, self.channel_method)

            # Return single scalar result using format_marker_result
            agg_name = f"trial_{self.trial_method}_roi_{self.channel_method}"
            # Extract inner dict from format_marker_result
            result = format_marker_result(
                feature_name="_temp",
                data=float(final_value),
                col_names=[f"all_channels_{agg_name}"],
                channel_aggregated=True,
            )
            return result["_temp"]
        else:
            # Standard aggregation: apply aggregation manually
            # Check for no aggregation
            if self.channel_method is None and self.trial_method is None:
                col_names = [f"{ch}" for ch in ch_names]
                # Use format_marker_result and extract inner dict
                result = format_marker_result(
                    feature_name="_temp",
                    data=psd_summary_values,
                    col_names=col_names,
                    channel_aggregated=False,
                )
                return result["_temp"]
            else:
                # Apply aggregation
                result_data = psd_summary_values

                # Channel aggregation
                if self.channel_method is not None:
                    result_data = aggregate_data(
                        result_data, self.channel_method, axis=1
                    )

                # Trial aggregation
                if self.trial_method is not None:
                    if result_data.ndim == 1:
                        result_data = aggregate_data(
                            result_data,
                            self.trial_method,
                            axis=None,
                        )
                    else:
                        result_data = aggregate_data(
                            result_data, self.trial_method, axis=0
                        )

                # Return result data without unnecessary reshaping - preserve tensor structure
                # Scalar: keep as scalar
                # 1D array: keep as 1D (n_trials) or (n_channels)
                # 2D array: keep as 2D (n_trials, n_channels)

                # Generate column names based on aggregation and result shape
                if (
                    self.channel_method is not None
                    and self.trial_method is not None
                ):
                    col_names = ["all_channels_all_trials"]
                elif self.channel_method is not None:
                    # result_data shape: (n_trials,) after channel aggregation
                    n_trials = (
                        result_data.shape[0] if result_data.ndim >= 1 else 1
                    )
                    col_names = [f"trial_{i}" for i in range(n_trials)]
                elif self.trial_method is not None:
                    col_names = [f"{ch}" for ch in ch_names]
                else:
                    col_names = [f"{ch}" for ch in ch_names]

                # Use format_marker_result and extract inner dict
                result = format_marker_result(
                    feature_name="_temp",
                    data=result_data,
                    col_names=col_names,
                    channel_aggregated=(self.channel_method is not None),
                )
                return result["_temp"]
