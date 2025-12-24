"""Spectral power with frequency band extraction."""

from typing import Any, ClassVar, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import apply_aggregation_preserve_dims, filter_to_eeg_channels
from ._spectral_power_base import SpectralPowerBase

__all__ = ["SpectralPowerBands"]


@register_marker
class SpectralPowerBands(EEGEpochsMarker):
    """Spectral power with frequency band extraction and optional aggregation.

    Computes PSD and extracts power in frequency bands, with optional
    channel and trial aggregation.

    Parameters
    ----------
    bands : dict, optional
        Frequency bands as {name: (fmin, fmax)}. If None, uses standard EEG
        bands: delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz),
        gamma (30-45 Hz).
    rois : list of str or int, optional
        Channel specifications for filtering BEFORE computation.
        Each item can be:
        - int: channel index (e.g., 0, 1, 223)
        - str: channel name (e.g., 'E1', 'E224') OR semantic ROI (e.g., 'scalp')
        If None, uses all channels.
    channel_method : str, optional
        Method for channel aggregation: 'mean', 'median', 'std', 'trim_mean80',
        'trim_mean90', etc. If None, no channel aggregation.
    trial_method : str, optional
        Method for trial aggregation: 'mean', 'median', 'std', 'trim_mean80',
        'trim_mean90', etc. If None, no trial aggregation.
    normalize : bool, default=False
        If True, normalize power by total power (relative power).
    dB : bool, default=True
        If True, convert power to decibels (10 * log10).
    entropy : bool, default=False
        If True, compute spectral entropy instead of band power.
        Requires normalize=True.
    n_fft : int, optional
        Length of FFT for Welch PSD. If None, uses adaptive sizing.
    n_per_seg : int, optional
        Length of each segment for Welch PSD. If None, uses adaptive sizing.
    n_overlap : int, optional
        Number of overlap points. If None, uses adaptive sizing.
    db_threshold : float, optional
        Minimum threshold for dB conversion. If None, uses adaptive threshold.
    tmin : float, optional
        Start time for analysis. If None, use start of epochs.
    tmax : float, optional
        End time for analysis. If None, use end of epochs.
    equipment : str, default="egi256"
        Equipment configuration for ROI resolution.
    on : str, optional
        Data type to compute on (default "EEG").
    name : str, optional
        Name of the marker.

    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"spectralpower": "timeseries"}}

    def __init__(
        self,
        bands: Optional[dict] = None,
        rois: Union[list[str], list[int], None] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        normalize: bool = False,
        dB: bool = True,
        entropy: bool = False,
        n_fft: Optional[int] = None,
        n_per_seg: Optional[int] = None,
        n_overlap: Optional[int] = None,
        db_threshold: Optional[float] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize spectral power bands marker."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        self.bands = bands
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method
        self.normalize = normalize
        self.dB = dB
        self.entropy = entropy
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.db_threshold = db_threshold

        # Validate
        if self.entropy and not self.normalize:
            raise ValueError("Spectral entropy requires normalize=True")

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute spectral power in frequency bands.

        Always returns a 3D tensor with shape (n_bands, n_epochs, n_channels).
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
            Spectral power with keys:
            - 'data': array of shape (n_bands, n_epochs, n_channels)
            - 'col_names': channel names
            - 'row_names': band names

        """
        logger.debug("Computing spectral power bands")

        # Get and validate epochs
        data_obj = input["data"]
        self._validate_input(data_obj)

        # Filter to EEG channels only
        data_obj, _, _ = filter_to_eeg_channels(data_obj)
        ch_names = data_obj.ch_names
        sfreq = data_obj.info["sfreq"]

        # Apply ROI filtering BEFORE computation if specified
        if self.rois is not None:
            from ..utils import apply_roi_filtering_to_epochs

            data_obj, ch_names = apply_roi_filtering_to_epochs(
                data_obj,
                self.rois,
                self.equipment,
                marker_name="SpectralPowerBands",
            )

        # Get standard bands if not specified
        if self.bands is None:
            self.bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 12),
                "beta": (12, 30),
                "gamma": (30, 45),
            }

        # Compute PSD using base (with caching potential)
        psd_base = SpectralPowerBase()

        # For normalization, need full spectrum
        if self.normalize:
            fmin = 1.0
            fmax = min(45.0, sfreq / 2 - 1)
        else:
            # Compute PSD for widest band range
            fmin = min(band[0] for band in self.bands.values())
            fmax = max(band[1] for band in self.bands.values())
            fmax = min(fmax, sfreq / 2 - 1)

        # Use base to compute PSD (with caching)
        psds, freqs = psd_base.compute(
            data_obj,
            self.n_fft,
            self.n_per_seg,
            self.n_overlap,
            self.tmin,
            self.tmax,
            fmin,
            fmax,
        )

        # Normalize if requested
        if self.normalize:
            total_power = np.sum(psds, axis=-1, keepdims=True)
            total_power = np.maximum(total_power, 1e-12)
            psds = psds / total_power

        # Extract band powers
        all_band_powers = {}
        for band_name, (band_fmin, band_fmax) in self.bands.items():
            freq_mask = (freqs >= band_fmin) & (freqs < band_fmax)

            if np.any(freq_mask):
                band_psds = psds[:, :, freq_mask]

                if self.entropy:
                    # Spectral entropy
                    n_bins = band_psds.shape[-1]
                    band_psds_safe = np.where(band_psds > 0, band_psds, 1e-12)
                    band_powers = -np.sum(
                        band_psds_safe * np.log(band_psds_safe), axis=-1
                    ) / np.log(n_bins)
                else:
                    # Sum across frequency
                    band_powers = np.sum(band_psds, axis=-1)

                all_band_powers[band_name] = band_powers
            else:
                n_epochs, n_channels = psds.shape[:2]
                all_band_powers[band_name] = np.zeros((n_epochs, n_channels))

        # Convert to dB if requested
        if self.dB and not self.entropy:
            # Determine threshold
            if self.db_threshold is not None:
                threshold = self.db_threshold
            else:
                # Adaptive threshold
                all_nonzero = []
                for band_data in all_band_powers.values():
                    nonzero = band_data[band_data > 0]
                    if len(nonzero) > 0:
                        all_nonzero.extend(nonzero)

                if len(all_nonzero) == 0:
                    raise ValueError(
                        "No non-zero power values for dB conversion"
                    )

                threshold = max(
                    np.min(all_nonzero) * 0.01, np.finfo(np.float64).eps
                )

            # Apply dB conversion
            for band_name in all_band_powers:
                band_data = all_band_powers[band_name]
                band_data = np.maximum(band_data, threshold)
                all_band_powers[band_name] = 10 * np.log10(band_data)

        # Format output - always return 3D tensor
        band_order = list(all_band_powers.keys())
        output_data = np.stack(
            [all_band_powers[band] for band in band_order],
            axis=0,
        )  # Shape: (n_bands, n_epochs, n_channels)

        output_ch_names = ch_names

        # Apply aggregation while preserving 3D structure
        # Use trial→channel order (matches original NICE behavior)
        output_data = apply_aggregation_preserve_dims(
            output_data,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
            aggregation_order="trial_channel",
        )

        return format_marker_result(
            feature_name="spectralpower",
            data=output_data,  # Shape: (n_bands, n_epochs, n_channels)
            col_names=output_ch_names,
            channel_aggregated=(self.channel_method is not None),
        )
