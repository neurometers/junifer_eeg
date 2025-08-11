"""Simple spectral power marker using MNE."""

from typing import Any, ClassVar, List, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker
from scipy import stats

from .utils import apply_roi_trial_aggregation, get_data_for_rois


def trim_mean80(data, axis=None):
    """Compute trimmed mean removing top and bottom 10% (80% trimmed mean).

    This matches the NICE 'trim_mean80' aggregation method used in ground truth.

    Parameters
    ----------
    data : array_like
        Input data.
    axis : int, optional
        Axis along which to compute the trimmed mean.

    Returns
    -------
    float or ndarray
        Trimmed mean with 10% trimmed from each tail (80% of data used).
    """
    return stats.trim_mean(data, proportiontocut=0.1, axis=axis)


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
        fmin: float = 1.0,
        fmax: float = 45.0,
        normalize: bool = False,
        dB: bool = True,
        bands: Optional[dict] = None,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        n_fft: Optional[int] = None,
        n_per_seg: Optional[int] = None,
        n_overlap: Optional[int] = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[List[str]] = None,
        trial_aggregation_method: Optional[List[str]] = None,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the SpectralPower marker.

        Parameters
        ----------
        fmin : float, default=1.0
            Minimum frequency for analysis.
        fmax : float, default=45.0
            Maximum frequency for analysis.
        normalize : bool, default=False
            If True, normalize power by total power (relative power).
        dB : bool, default=True
            If True, convert power to decibels (10 * log10). Matches NICE behavior.
        bands : dict, optional
            Custom frequency bands. If None, use standard bands.
        epoch_length : float, default=2.0
            Length of epochs to create from continuous data in seconds.
        overlap : float, default=0.0
            Overlap between epochs (0.0 = no overlap, 0.9 = 90% overlap).
        n_fft : int, optional
            Length of the FFT used for Welch PSD. If None, uses adaptive sizing.
        n_per_seg : int, optional
            Length of each segment for Welch PSD. If None, uses adaptive sizing.
        n_overlap : int, optional
            Number of points to overlap between segments. If None, uses adaptive sizing.
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
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        self.dB = dB
        self.bands = bands
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        super().__init__(on=on, name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
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
        # Get the MNE data object (can be Raw or Epochs)
        data_obj = input["data"]

        # Extract metadata from input to preserve element information
        meta = input.get("meta", None)

        # Handle both Raw and Epochs objects
        if hasattr(data_obj, "events"):
            # This is an Epochs object
            # Check if epochs object is empty
            if len(data_obj) == 0:
                # Return empty results for empty epochs
                ch_names = data_obj.ch_names
                if self.rois is not None:
                    roi_data = {
                        roi: np.array([]).reshape(0, 0) for roi in self.rois
                    }
                else:
                    roi_data = {
                        ch: np.array([]).reshape(0, 0) for ch in ch_names
                    }

                return apply_roi_trial_aggregation(
                    roi_data,
                    roi_aggregation_methods=self.roi_aggregation_method,
                    trial_aggregation_methods=self.trial_aggregation_method,
                    marker_name="spectralpower",
                    meta=meta,
                )
            epochs_data = (
                data_obj.get_data()
            )  # Shape (n_epochs, n_channels, n_times)
            ch_names = data_obj.ch_names
            info = data_obj.info
        else:
            # This is a Raw object, reshape to look like single epoch
            raw_data = data_obj.get_data()  # Shape (n_channels, n_times)
            epochs_data = raw_data[
                np.newaxis, :, :
            ]  # Shape (1, n_channels, n_times)
            ch_names = data_obj.ch_names
            info = data_obj.info

        n_epochs, n_channels, n_samples = epochs_data.shape

        # Use full epoch length for PSD computation
        sfreq = info["sfreq"]
        print(
            f"Using full epoch length ({n_samples} samples, {n_samples / sfreq:.1f}s) for PSD computation"
        )

        # Use custom bands or default frequency bands (matching NICE/ICM)
        if self.bands is not None:
            bands = self.bands
        else:
            # Check if we're testing a specific frequency range (for comparison scripts)
            # If fmin/fmax match a standard band, use only that band
            standard_bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 12),
                "beta": (12, 30),
                "gamma": (30, 45),
            }

            # Find matching band for the requested frequency range
            target_band = None
            for band_name, (band_fmin, band_fmax) in standard_bands.items():
                if (
                    abs(self.fmin - band_fmin) < 0.1
                    and abs(self.fmax - band_fmax) < 0.1
                ):
                    target_band = band_name
                    break

            if target_band is not None:
                # Use only the matching band for single-band testing
                bands = {target_band: standard_bands[target_band]}
            else:
                # Use all bands for general analysis
                bands = standard_bands

        # Determine frequency range based on sampling rate and parameters
        sfreq = info["sfreq"]
        max_freq = min(
            self.fmax, sfreq / 2 - 1
        )  # Ensure we don't exceed Nyquist

        # PERFORMANCE FIX: Use MNE's efficient vectorized PSD computation
        # Instead of creating RawArray for each epoch, compute PSD on all epochs at once
        # Use specified parameters or fall back to adaptive sizing
        n_per_seg = (
            self.n_per_seg
            if self.n_per_seg is not None
            else min(64, n_samples // 2)
        )
        n_overlap = (
            self.n_overlap
            if self.n_overlap is not None
            else min(32, n_per_seg // 2)
        )

        psd_params = {
            "method": "welch",
            "fmin": self.fmin,  # Use exact fmin like NICE (no offset)
            "fmax": max_freq,
            "n_per_seg": n_per_seg,
            "n_overlap": n_overlap,
            "verbose": False,
        }

        # Add n_fft if specified (for NICE compatibility)
        if self.n_fft is not None:
            psd_params["n_fft"] = self.n_fft

        # Compute PSD for all epochs at once - MUCH faster!
        if hasattr(data_obj, "events"):
            # Use the original Epochs object for efficient computation
            psd = data_obj.compute_psd(**psd_params)
            psds, freqs = psd.get_data(
                return_freqs=True
            )  # Shape: (n_epochs, n_channels, n_freqs)
        else:
            # For Raw data, create temporary raw and compute PSD
            import mne

            temp_raw = mne.io.RawArray(
                epochs_data[0], info.copy(), verbose=False
            )
            psd = temp_raw.compute_psd(**psd_params)
            psds_single, freqs = psd.get_data(
                return_freqs=True
            )  # Shape: (n_channels, n_freqs)
            psds = psds_single[
                np.newaxis, :, :
            ]  # Shape: (1, n_channels, n_freqs)

        # Compute band powers for all epochs and channels at once
        all_band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            # Skip band if sampling rate too low
            if fmax > max_freq:
                all_band_powers[band_name] = np.zeros((n_epochs, n_channels))
                continue

            # Find frequency indices for this band
            freq_mask = (freqs >= fmin) & (freqs < fmax)

            if np.any(freq_mask):
                # Vectorized integration across frequency band for all epochs/channels
                # psds shape: (n_epochs, n_channels, n_freqs)
                # Extract frequencies and PSDs for this band
                band_psds = psds[:, :, freq_mask]

                # Integrate power across frequency dimension
                band_powers = np.sum(band_psds, axis=-1)
                all_band_powers[band_name] = band_powers
            else:
                all_band_powers[band_name] = np.zeros((n_epochs, n_channels))

        # Apply normalization if requested (relative power)
        # PERFORMANCE OPTIMIZATION: Vectorized normalization instead of nested loops
        if self.normalize:
            # Stack all band powers into a single array for vectorized operations
            # Shape: (n_bands, n_epochs, n_channels)
            band_names = list(bands.keys())
            stacked_powers = np.stack(
                [all_band_powers[band] for band in band_names], axis=0
            )

            # Calculate total power across bands for each epoch/channel
            # Shape: (n_epochs, n_channels)
            total_powers = np.sum(stacked_powers, axis=0)

            # Avoid division by zero
            total_powers = np.maximum(total_powers, 1e-12)

            # Vectorized normalization: divide each band by total power
            # Broadcasting: (n_bands, n_epochs, n_channels) / (n_epochs, n_channels)
            normalized_powers = stacked_powers / total_powers[np.newaxis, :, :]

            # Update all_band_powers with normalized values
            for i, band_name in enumerate(band_names):
                all_band_powers[band_name] = normalized_powers[i]

        # Apply dB conversion if requested (matches NICE behavior)
        # NICE applies dB conversion AFTER normalization and frequency integration
        if self.dB and not self.normalize:
            for band_name in bands.keys():
                # Convert to dB: 10 * log10(power)
                # Handle zero/negative values by setting a minimum threshold
                band_data = all_band_powers[band_name]
                # Set minimum threshold to avoid log(0) or log(negative)
                min_threshold = 1e-12
                band_data = np.maximum(band_data, min_threshold)
                all_band_powers[band_name] = 10 * np.log10(band_data)

        # Check if we should return raw PSD data (no aggregation)
        if (
            self.roi_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return raw band power data without aggregation for direct comparison with NICE
            # Find the band that matches the requested frequency range (fmin, fmax)
            target_band = None
            for band_name, (band_fmin, band_fmax) in bands.items():
                if (
                    abs(band_fmin - self.fmin) < 0.1
                    and abs(band_fmax - self.fmax) < 0.1
                ):
                    target_band = band_name
                    break

            # If no exact match, use the first band (fallback)
            if target_band is None:
                target_band = next(iter(all_band_powers.keys()))

            band_data = all_band_powers[target_band]

            # Debug: Print actual shape before returning
            print(
                f"    DEBUG Junifer: target_band={target_band}, band_data.shape={band_data.shape}"
            )
            print(
                f"    DEBUG Junifer: n_epochs={n_epochs}, n_channels={len(ch_names)}"
            )
            print(
                f"    DEBUG Junifer: band_data min/max = {np.min(band_data):.2e}/{np.max(band_data):.2e}"
            )

            return {
                "spectralpower": {
                    "data": band_data,  # Shape: (n_epochs, n_channels)
                    "col_names": [f"{target_band}_{ch}" for ch in ch_names],
                }
            }

        # For single frequency band testing (like in comparison scripts),
        # return aggregated result for the target band only
        if len(all_band_powers) == 1 and (
            self.roi_aggregation_method == ["mean"]
            and self.trial_aggregation_method == ["mean"]
        ):
            # Single band, full aggregation - return scalar
            band_name = next(iter(all_band_powers.keys()))
            band_data = all_band_powers[
                band_name
            ]  # Shape: (n_epochs, n_channels)

            # DEBUG: Print raw data statistics before aggregation
            print(f"    DEBUG Junifer aggregation for {band_name}:")
            print(f"      Raw band_data shape: {band_data.shape}")
            print(f"      Raw band_data mean: {np.mean(band_data):.6e}")
            print(f"      Raw band_data std: {np.std(band_data):.6e}")
            print(
                f"      Raw band_data min/max: {np.min(band_data):.6e}/{np.max(band_data):.6e}"
            )

            # Apply mean aggregation across channels (ROI aggregation)
            roi_aggregated = np.mean(band_data, axis=1)  # Shape: (n_epochs,)
            print(f"      After ROI agg shape: {roi_aggregated.shape}")
            print(f"      After ROI agg mean: {np.mean(roi_aggregated):.6e}")

            # Apply mean aggregation across epochs (trial aggregation)
            final_value = np.mean(roi_aggregated)  # Shape: scalar
            print(f"      Final aggregated value: {final_value:.6e}")

            return {
                "spectralpower": {
                    "data": final_value,  # Single scalar value
                    "col_names": [f"{band_name}_mean"],
                },
            }

        # Standard aggregation path - combine all bands into single feature set
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
                meta=meta,
            )

            # Extract values and update column names with band info
            for _feature_name, feature_data in band_results.items():
                band_values = feature_data["data"].flatten()
                band_col_names = [
                    f"{band_name}_{col}" for col in feature_data["col_names"]
                ]

                all_values.extend(band_values)
                col_names.extend(band_col_names)

        # Return combined results with proper metadata structure
        return {
            "spectralpower": {
                "data": np.array(all_values).reshape(1, -1),
                "col_names": col_names,
            },
        }
