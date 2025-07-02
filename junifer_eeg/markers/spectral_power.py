"""Simple spectral power marker using MNE."""

from typing import Any, ClassVar, List, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class SpectralPower(BaseMarker):
    """Simple spectral power marker using MNE with flexible ROI and trial aggregation.

    Computes power in standard EEG frequency bands with support for
    channel-wise computation and flexible aggregation methods.
    """

    _DEPENDENCIES: ClassVar = {"mne", "pandas"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"spectralpower": "vector"}}

    def __init__(
        self,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the SpectralPower marker.

        Parameters
        ----------
        epoch_length : float, default=2.0
            Length of epochs to create from continuous data in seconds.
        overlap : float, default=0.0
            Overlap between epochs (0.0 = no overlap, 0.9 = 90% overlap).
        rois : list of str, optional
            List of ROI names. If None, use all channels.
        roi_aggregation_method : list of str, optional
            Methods to aggregate across ROI electrodes: ['mean', 'std'].
        trial_aggregation_method : list of str, optional
            Methods to aggregate across trials/epochs: ['mean', 'std'].
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute spectral power features with flexible aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed spectral power features with aggregation.
        """
        # Get the MNE Epochs object
        epochs = input["data"]

        # Get epochs data: Shape (n_epochs, n_channels, n_times)
        epochs_data = epochs.get_data()
        ch_names = epochs.ch_names
        info = epochs.info

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Define frequency bands
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
        }

        # Compute spectral power for each epoch and channel
        all_band_powers = {}
        for band_name in bands.keys():
            all_band_powers[band_name] = np.zeros((n_epochs, n_channels))

        for epoch_idx in range(n_epochs):
            # Create temporary raw object for this epoch
            import mne

            epoch_data = epochs_data[
                epoch_idx
            ]  # Shape: (n_channels, n_samples)

            epoch_raw = mne.io.RawArray(epoch_data, info.copy(), verbose=False)

            # Compute PSD using MNE
            psd = epoch_raw.compute_psd(fmin=1, fmax=30, verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)

            # Compute band powers for this epoch
            for ch_idx in range(n_channels):
                for band_name, (fmin, fmax) in bands.items():
                    # Find frequency indices
                    freq_mask = (freqs >= fmin) & (freqs < fmax)

                    # Compute mean power in band
                    band_power = psds[ch_idx, freq_mask].mean()
                    all_band_powers[band_name][epoch_idx, ch_idx] = float(
                        band_power
                    )

        # Combine all bands into single feature set
        all_values = []
        col_names = []

        for band_name, band_data in all_band_powers.items():
            # Handle ROI selection
            if self.rois is not None:
                # Extract data for specified ROIs
                roi_data = get_data_for_rois(
                    band_data.T,  # Transpose to (n_channels, n_epochs)
                    list(ch_names),
                    self.rois,
                )
            else:
                # Use all channels as individual ROIs
                roi_data = {
                    ch: band_data[:, i : i + 1].T
                    for i, ch in enumerate(ch_names)
                }

            # Apply aggregation for this band
            band_results = apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="spectralpower",
            )

            # Extract values and update column names with band info
            for _feature_name, feature_data in band_results.items():
                band_values = feature_data["data"].flatten()
                band_col_names = [
                    f"{band_name}_{col}" for col in feature_data["col_names"]
                ]

                all_values.extend(band_values)
                col_names.extend(band_col_names)

        # Return combined results
        return {
            "spectralpower": {
                "data": np.array(all_values).reshape(1, -1),
                "col_names": col_names,
            }
        }
