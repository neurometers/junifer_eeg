"""Simple spectral power marker using MNE."""

from typing import Any, ClassVar, List, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker
from scipy import stats


def trim_mean80(data, axis=None):
    """Compute trimmed mean removing top and bottom 10% (80% trimmed mean).

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
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"spectralpower": "timeseries"}
    }  # 2D: (epochs, channels)

    def __init__(
        self,
        fmin: float = 1.0,
        fmax: float = 45.0,
        normalize: bool = False,
        dB: bool = True,
        entropy: bool = False,
        bands: Optional[dict] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        n_fft: Optional[int] = None,
        n_per_seg: Optional[int] = None,
        n_overlap: Optional[int] = None,
        db_threshold: Optional[float] = None,
        rois: Union[List[str], List[int], None] = None,
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
        equipment: str = "egi256",
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
            If True, convert power to decibels (10 * log10).
        entropy : bool, default=False
            If True, compute spectral entropy instead of band power. Requires normalize=True.
        bands : dict, optional
            Custom frequency bands. If None, use standard bands.
        tmin : float, optional
            Start time for analysis in seconds. If None, use start of epoch.
        tmax : float, optional
            End time for analysis in seconds. If None, use end of epoch.
        n_fft : int, optional
            Length of the FFT used for Welch PSD. If None, uses adaptive sizing.
        n_per_seg : int, optional
            Length of each segment for Welch PSD. If None, uses adaptive sizing.
        n_overlap : int, optional
            Number of points to overlap between segments. If None, uses adaptive sizing.
        db_threshold : float, optional
            Minimum threshold for dB conversion to avoid log(0). If None (default),
            uses adaptive threshold based on data (1% of minimum non-zero power value).
            Lower values (e.g., 1e-15) may be needed for low-amplitude signals.
            Only used when dB=True.
        rois : list of str or int, optional
            Flat list of channel specifications for filtering AFTER PSD computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp')

            If None, uses all channels.
        channel_aggregation_method : str, optional
            Methods to aggregate across ROI electrodes: 'mean', 'std', 'median', 'trim_mean80', 'trim_mean90', etc.
        trial_aggregation_method : str, optional
            Methods to aggregate across trials/epochs: 'mean', 'std', 'median', 'trim_mean80', 'trim_mean90', etc.
        equipment : str, default="egi256"
            Equipment configuration for ROI resolution.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        self.dB = dB
        self.entropy = entropy
        self.bands = bands
        self.tmin = tmin
        self.tmax = tmax
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.db_threshold = db_threshold
        self.rois = rois
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment

        # Validate entropy usage
        if self.entropy and not self.normalize:
            raise ValueError("Spectral entropy requires normalize=True")

        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 2D/3D tensor data (no aggregation)
        - 'vector': 1D array (one aggregation applied)
        - 'scalar_table': scalar value (both aggregations applied)
        """
        # No aggregation → 2D/3D tensor (epochs, channels) or (bands, epochs, channels) → timeseries
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

        Raises
        ------
        ValueError
            If input data is not Epochs or if epochs are empty.
        """
        from .utils import filter_to_eeg_channels

        # Get the MNE data object - must be Epochs
        data_obj = input["data"]

        if not hasattr(data_obj, "events"):
            raise ValueError(
                "SpectralPower requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(data_obj) == 0:
            raise ValueError("Cannot compute spectral power on empty epochs.")

        # Filter to only EEG channels (E1-E256), excluding D/DI auxiliary channels
        data_obj, _, _ = filter_to_eeg_channels(data_obj)

        ch_names = data_obj.ch_names
        info = data_obj.info
        sfreq = info["sfreq"]

        # Get number of time samples from epochs
        epochs_data = (
            data_obj.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)
        n_samples = epochs_data.shape[2]

        # Use custom bands or default frequency bands
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
            # Only use single band if BOTH fmin AND fmax match exactly AND they are scalars
            target_band = None
            if not isinstance(self.fmin, list) and not isinstance(
                self.fmax, list
            ):
                for band_name, (
                    band_fmin,
                    band_fmax,
                ) in standard_bands.items():
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
                # CRITICAL FIX: If fmin/fmax don't match a standard band, create a custom band
                # Use all standard bands ONLY if this is the default initialization (fmin=1, fmax=45)
                if abs(self.fmin - 1.0) < 0.1 and abs(self.fmax - 45.0) < 0.1:
                    # Full spectrum request - create single custom band
                    bands = {"full_spectrum": (self.fmin, self.fmax)}
                else:
                    # Use all bands for general analysis (default case)
                    bands = standard_bands

        # Determine frequency range based on sampling rate and parameters
        # Handle both single values and lists for fmax
        if isinstance(self.fmax, list):
            max_freq = min(max(self.fmax), sfreq / 2 - 1)
        else:
            max_freq = min(self.fmax, sfreq / 2 - 1)

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

        # CRITICAL FIX: For normalized power, MUST compute full spectrum (1-45 Hz)
        if self.normalize:
            # Always compute full spectrum for normalization
            # Do NOT use max_freq here as it's band-specific! Use Nyquist directly.
            psd_fmin = 1.0
            psd_fmax = min(45.0, sfreq / 2 - 1)
        elif isinstance(self.fmin, list) or isinstance(self.fmax, list):
            # For multi-band analysis, compute full spectrum and extract bands later
            fmin_vals = (
                self.fmin if isinstance(self.fmin, list) else [self.fmin]
            )
            fmax_vals = (
                self.fmax if isinstance(self.fmax, list) else [self.fmax]
            )
            psd_fmin = min(fmin_vals)
            psd_fmax = max(fmax_vals)
        else:
            psd_fmin = self.fmin
            psd_fmax = max_freq

        psd_params = {
            "method": "welch",
            "fmin": psd_fmin,
            "fmax": psd_fmax,
            "n_per_seg": n_per_seg,
            "n_overlap": n_overlap,
            "verbose": False,
        }

        if self.n_fft is not None:
            psd_params["n_fft"] = self.n_fft

        # Crop to time window if specified
        if self.tmin is not None or self.tmax is not None:
            cropped_epochs = data_obj.copy().crop(
                tmin=self.tmin, tmax=self.tmax
            )
        else:
            cropped_epochs = data_obj

        # Compute PSD for all epochs at once
        psd = cropped_epochs.compute_psd(**psd_params)
        psds, freqs = psd.get_data(
            return_freqs=True
        )  # Shape: (n_epochs, n_channels, n_freqs)

        n_epochs, n_channels = psds.shape[:2]

        # Apply normalization if requested (relative power)
        if self.normalize:
            # data_norm = data / data.sum(axis=-1, keepdims=True)
            # Then extract band and sum
            total_power_per_epoch_channel = np.sum(
                psds, axis=-1, keepdims=True
            )  # Shape: (n_epochs, n_channels, 1)

            # Avoid division by zero
            total_power_per_epoch_channel = np.maximum(
                total_power_per_epoch_channel, 1e-12
            )

            # Normalize each frequency bin by total power
            psds_normalized = psds / total_power_per_epoch_channel
        else:
            psds_normalized = psds

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
                # Use normalized PSD if normalization is requested
                band_psds = psds_normalized[:, :, freq_mask]

                if self.entropy:
                    # Spectral entropy: -sum(p * log(p)) / log(n_bins)
                    n_bins = band_psds.shape[-1]
                    # Handle zeros by replacing with small value
                    band_psds_safe = np.where(band_psds > 0, band_psds, 1e-12)
                    band_powers = -np.sum(
                        band_psds_safe * np.log(band_psds_safe), axis=-1
                    ) / np.log(n_bins)
                else:
                    # Sum across frequency (standard band power)
                    band_powers = np.sum(band_psds, axis=-1)

                all_band_powers[band_name] = band_powers
            else:
                all_band_powers[band_name] = np.zeros((n_epochs, n_channels))

        if self.dB:
            # Determine threshold: user-specified or adaptive
            if self.db_threshold is not None:
                threshold = self.db_threshold
            else:
                # Use data-adaptive threshold
                # The threshold should be based on the actual noise floor of the data
                # Use the minimum non-zero power value across all bands, then go 2 orders of magnitude lower
                all_nonzero_values = []
                for band_data in all_band_powers.values():
                    nonzero = band_data[band_data > 0]
                    if len(nonzero) > 0:
                        all_nonzero_values.extend(nonzero)

                if len(all_nonzero_values) == 0:
                    raise ValueError(
                        "No non-zero power values found. Cannot compute dB conversion."
                    )
                # Use 1% of the minimum non-zero value as threshold
                # This ensures we don't clip real data while still avoiding log(0)
                data_min = np.min(all_nonzero_values)
                threshold = data_min * 0.01
                # But don't go below machine epsilon for float64
                threshold = max(threshold, np.finfo(np.float64).eps)

            for band_name in bands.keys():
                # Convert to dB: 10 * log10(power)
                # Handle zero/negative values by setting a minimum threshold
                band_data = all_band_powers[band_name]

                # Apply threshold
                band_data = np.maximum(band_data, threshold)
                all_band_powers[band_name] = 10 * np.log10(band_data)

        # Check if we should return raw PSD data (no aggregation)
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return raw per-epoch, per-channel results without any aggregation
            if len(all_band_powers) == 1:
                band_name = next(iter(all_band_powers.keys()))
                band_data = all_band_powers[band_name]
                return {
                    "spectralpower": {
                        "data": band_data,  # Shape: (n_epochs, n_channels)
                        "col_names": ch_names,
                    }
                }

            # Multiple bands: stack into tensor (bands, epochs, channels)
            band_order = list(all_band_powers.keys())
            tensor = np.stack(
                [np.asarray(all_band_powers[band]) for band in band_order],
                axis=0,
            )  # Shape: (n_bands, n_epochs, n_channels)

            return {
                "spectralpower": {
                    "data": tensor,
                    "col_names": ch_names,
                }
            }

        # Remove the single-band debug path to ensure consistent array output format

        # Standard aggregation path - preserve tensor structure
        from .utils import aggregate_data, get_data_for_rois

        # Initialize storage for structured results
        band_results = {}

        for band_name, band_data in all_band_powers.items():
            # Apply ROI filtering if specified
            if self.rois is not None:
                roi_data = get_data_for_rois(
                    band_data.T,
                    list(ch_names),
                    self.rois,
                    self.equipment,
                )
                if "selected_channels" in roi_data:
                    band_data = roi_data["selected_channels"].T

            # Apply aggregation for this band
            result_data = band_data

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

            # Store result with proper structure - NO FLATTENING
            band_results[band_name] = result_data

        # Return structured results preserving tensor dimensions
        if len(band_results) == 1:
            # Single band - return the tensor directly
            band_name = next(iter(band_results.keys()))
            result_data = band_results[band_name]

            return {
                "spectralpower": {
                    "data": result_data,  # Preserves original tensor structure
                }
            }

        # Multiple bands - stack results to preserve band dimension
        band_order = list(band_results.keys())
        tensor = np.stack(
            [np.asarray(band_results[band]) for band in band_order], axis=0
        )

        return {
            "spectralpower": {
                "data": tensor,
            }
        }
