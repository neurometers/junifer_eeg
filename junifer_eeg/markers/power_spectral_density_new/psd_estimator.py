"""Power Spectral Density Estimator marker implementation."""

from typing import Any, ClassVar

from junifer.api.decorators import register_marker

from ..base import format_marker_result
from ._psd_base import PowerSpectralDensityBase


@register_marker
class PowerSpectralDensityEstimator(PowerSpectralDensityBase):
    """Power Spectral Density Estimator using MNE-Python's compute_psd method.

    This marker provides configurable PSD estimation with various methods
    and parameters, adapted from the NICE package implementation.
    """

    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {
            "psd_data": "matrix",
            "psd_freqs": "vector",
            "psd_data_norm": "matrix",
        },
    }

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Power Spectral Density.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw or Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD data, frequencies, and normalized PSD.
        """
        # Prepare and filter data
        data_obj, ch_names = self._prepare_data(input)

        # Compute PSD (average across epochs for Estimator)
        psd_data, freqs, ch_names = self._compute_psd(
            data_obj, preserve_epochs=False
        )  # Shape: (n_channels, n_freqs)

        # Compute normalized version
        psd_data_norm = psd_data / psd_data.sum(axis=-1, keepdims=True)

        # Generate column names for frequencies
        freq_names = [f"freq_{freq:.2f}Hz" for freq in freqs]

        # Return data using centralized format_marker_result
        # Note: PSD Estimator outputs multiple features, each with different structure
        psd_result = format_marker_result(
            feature_name="psd_data",
            data=psd_data,  # Shape: (n_channels, n_freqs)
            col_names=freq_names,
            channel_aggregated=False,
        )
        # Add row_names for channel dimension
        psd_result["psd_data"]["row_names"] = ch_names

        freqs_result = format_marker_result(
            feature_name="psd_freqs",
            data=freqs.reshape(1, -1),  # Shape: (1, n_freqs)
            col_names=freq_names,
            channel_aggregated=True,  # No channel dim
        )

        norm_result = format_marker_result(
            feature_name="psd_data_norm",
            data=psd_data_norm,  # Shape: (n_channels, n_freqs)
            col_names=freq_names,
            channel_aggregated=False,
        )
        norm_result["psd_data_norm"]["row_names"] = ch_names

        # Combine all results
        return {
            **psd_result,
            **freqs_result,
            **norm_result,
        }
