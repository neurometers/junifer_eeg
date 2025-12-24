"""Power Spectral Density Summary marker implementation."""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.api.decorators import register_marker

from ..base import format_marker_result
from ..utils import apply_aggregation_preserve_dims, get_data_for_rois
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

        Always returns a 2D tensor with shape (n_epochs, n_channels).
        Even when dimensions have size 1, they are preserved for consistency.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD summary statistics with shape (n_epochs, n_channels)

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
                ch_names = roi_data_dict["selected_channels_names"]

        # Apply aggregation while preserving 2D structure
        # Use trial→channel order (matches original NICE behavior)
        psd_summary_values = apply_aggregation_preserve_dims(
            psd_summary_values,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
            aggregation_order="trial_channel",
        )

        # Return using format_marker_result
        return format_marker_result(
            feature_name="psdsummary",
            data=psd_summary_values,  # Shape: (n_epochs, n_channels)
            col_names=list(ch_names),
            channel_aggregated=(self.channel_method is not None),
        )

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
