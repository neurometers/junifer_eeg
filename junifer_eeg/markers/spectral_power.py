"""Simple spectral power marker using MNE."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


@register_marker
class SpectralPower(BaseMarker):
    """Simple spectral power marker using MNE.

    Computes power in standard EEG frequency bands.
    """

    _DEPENDENCIES: ClassVar = {"mne", "pandas"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"spectral_power": "vector"}}

    def __init__(self, on: str | None = None, name: str | None = None) -> None:
        """Initialize the SpectralPower marker."""
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute spectral power features.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed spectral power features.

        """
        # Get the MNE Raw object
        raw = input["data"]

        # Define frequency bands
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
        }

        # Compute PSD using MNE
        psd = raw.compute_psd(fmin=1, fmax=30, n_fft=None)
        psds, freqs = psd.get_data(return_freqs=True)

        # Compute band powers
        features = []
        feature_names = []

        for ch_idx, ch_name in enumerate(raw.ch_names):
            for band_name, (fmin, fmax) in bands.items():
                # Find frequency indices
                freq_mask = (freqs >= fmin) & (freqs < fmax)

                # Compute mean power in band
                band_power = psds[ch_idx, freq_mask].mean()

                features.append(float(band_power))
                feature_names.append(f"{ch_name}_{band_name}")

        # Return data as 1D array with proper format for junifer storage
        return {
            "spectral_power": {
                "data": np.array(features).reshape(
                    1, -1
                ),  # Shape: (1, n_features)
                "col_names": feature_names,
            }
        }
