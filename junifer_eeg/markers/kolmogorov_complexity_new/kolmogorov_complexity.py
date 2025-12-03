"""Kolmogorov Complexity marker implementation."""

from typing import Any

import numpy as np
from junifer.api.decorators import register_marker

from ._kolmogorov_complexity_base import KolmogorovComplexityBase


@register_marker
class KolmogorovComplexity(KolmogorovComplexityBase):
    """Kolmogorov Complexity marker with flexible ROI and trial aggregation.

    This marker computes the Kolmogorov complexity of EEG signals using
    compression-based approximation. The signal is first symbolized into
    discrete bins, then compressed using zlib, and the complexity is
    measured as the compression ratio.

    Supports channel-wise computation with flexible ROI and trial aggregation.
    """

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Kolmogorov complexity with flexible aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed Kolmogorov complexity features with aggregation.

        Raises
        ------
        ValueError
            If input data is not Epochs or if epochs are empty.
        """
        # Prepare and filter data
        epochs_data, ch_names = self._prepare_data(input)
        n_epochs, n_channels, _ = epochs_data.shape

        # Apply ROI filtering if specified
        epochs_data, ch_names = self._apply_roi_filtering(
            epochs_data, ch_names
        )
        n_epochs, n_channels, _ = epochs_data.shape

        # Compute Kolmogorov complexity for each channel and epoch
        k_values = np.zeros((n_epochs, n_channels), dtype=np.float64)

        for epoch_idx in range(n_epochs):
            for channel_idx in range(n_channels):
                signal = epochs_data[epoch_idx, channel_idx, :]
                k_values[epoch_idx, channel_idx] = (
                    self._compute_kolmogorov_for_signal(signal)
                )

        # Apply aggregation and return results
        results = self._apply_aggregation(k_values, ch_names)
        return {"kolmogorovcomplexity": results}
