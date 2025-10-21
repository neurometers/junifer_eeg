"""Enhanced Power Spectral Density markers for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class PowerSpectralDensityEstimator(BaseMarker):
    """Power Spectral Density Estimator using MNE-Python's compute_psd method.

    This marker provides configurable PSD estimation with various methods
    and parameters, adapted from the NICE package implementation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {
            "psd_data": "matrix",
            "psd_freqs": "vector",
            "psd_data_norm": "matrix",
        },
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        fmin: float = 0.0,
        fmax: float | None = None,
        psd_method: str = "welch",
        n_per_seg: int | None = None,
        n_overlap: int | None = None,
        n_fft: int | None = None,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the PowerSpectralDensityEstimator marker.

        Parameters
        ----------
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        fmin : float, default=0.0
            Minimum frequency for PSD computation.
        fmax : float, optional
            Maximum frequency for PSD computation. If None, use Nyquist.
        psd_method : str, default='welch'
            Method for PSD computation ('welch').
        n_per_seg : int, optional
            Length of each segment for Welch's method.
        n_overlap : int, optional
            Number of points to overlap between segments.
        n_fft : int, optional
            Length of the FFT used.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        self.psd_method = psd_method
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.n_fft = n_fft
        super().__init__(on=on, name=name)

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
        from .utils import filter_to_eeg_channels

        # Get the MNE data object
        data_obj = input["data"]

        # CRITICAL FIX: Filter to only EEG channels (E1-E256), excluding D/DI auxiliary channels
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Handle both Raw and Epochs data
        if hasattr(data_obj, "events"):  # This is Epochs
            epochs = data_obj

            # Crop to time window if specified
            if self.tmin is not None or self.tmax is not None:
                epochs_cropped = epochs.copy().crop(
                    tmin=self.tmin, tmax=self.tmax
                )
            else:
                epochs_cropped = epochs

            # Set frequency limits
            fmax = (
                self.fmax
                if self.fmax is not None
                else epochs.info["sfreq"] / 2
            )

            # Prepare MNE parameters
            mne_params = {}
            if self.n_per_seg is not None:
                mne_params["n_per_seg"] = self.n_per_seg
            if self.n_overlap is not None:
                mne_params["n_overlap"] = self.n_overlap
            if self.n_fft is not None:
                mne_params["n_fft"] = self.n_fft

            # Compute PSD using MNE on epochs
            spectrum = epochs_cropped.compute_psd(
                method="welch",
                fmin=self.fmin,
                fmax=fmax,
                **mne_params,
            )

            # Extract data and frequencies
            psd_data = (
                spectrum.get_data()
            )  # Shape: (n_epochs, n_channels, n_freqs)
            freqs = spectrum.freqs

            # Average across epochs to get (n_channels, n_freqs)
            psd_data = np.mean(
                psd_data, axis=0
            )  # Shape: (n_channels, n_freqs)

            # Use epochs channel names
            ch_names = epochs.ch_names

        else:  # Raw data
            raw = data_obj

            # Crop to time window if specified
            if self.tmin is not None or self.tmax is not None:
                raw_cropped = raw.copy().crop(tmin=self.tmin, tmax=self.tmax)
            else:
                raw_cropped = raw

            # Set frequency limits
            fmax = (
                self.fmax if self.fmax is not None else raw.info["sfreq"] / 2
            )

            # Prepare MNE parameters
            mne_params = {}
            if self.n_per_seg is not None:
                mne_params["n_per_seg"] = self.n_per_seg
            if self.n_overlap is not None:
                mne_params["n_overlap"] = self.n_overlap
            if self.n_fft is not None:
                mne_params["n_fft"] = self.n_fft

            if self.psd_method == "welch":
                # Compute PSD using MNE
                spectrum = raw_cropped.compute_psd(
                    method="welch",
                    fmin=self.fmin,
                    fmax=fmax,
                    **mne_params,
                )

                # Extract data and frequencies
                psd_data = spectrum.get_data()  # Shape: (n_channels, n_freqs)
                freqs = spectrum.freqs

            else:
                raise ValueError(
                    f"PSD method '{self.psd_method}' not supported. Use 'welch'.",
                )

            # Use raw channel names
            ch_names = raw.ch_names

        # Compute normalized version
        psd_data_norm = psd_data / psd_data.sum(axis=-1, keepdims=True)

        # Generate column names for frequencies
        freq_names = [f"freq_{freq:.2f}Hz" for freq in freqs]

        # Return data in junifer format
        return {
            "psd_data": {
                "data": psd_data,  # Shape: (n_channels, n_freqs)
                "col_names": freq_names,
                "row_names": ch_names,
            },
            "psd_freqs": {
                "data": freqs.reshape(1, -1),  # Shape: (1, n_freqs)
                "col_names": freq_names,
            },
            "psd_data_norm": {
                "data": psd_data_norm,  # Shape: (n_channels, n_freqs)
                "col_names": freq_names,
                "row_names": ch_names,
            },
        }


@register_marker
class PowerSpectralDensitySummary(BaseMarker):
    """Power Spectral Density Summary using percentile statistics.

    This marker computes summary statistics (percentiles) of the power
    spectral density across frequencies, adapted from the NICE package.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"psdsummary": "vector"}}

    def __init__(
        self,
        percentile: float = 50.0,
        tmin: float | None = None,
        tmax: float | None = None,
        fmin: float = 0.0,
        fmax: float | None = None,
        psd_method: str = "welch",
        n_per_seg: int | None = None,
        n_overlap: int | None = None,
        n_fft: int | None = None,
        rois: list[str] | None = None,
        roi_aggregation_method: list[str] | None = None,
        trial_aggregation_method: list[str] | None = None,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the PowerSpectralDensitySummary marker.

        Parameters
        ----------
        percentile : float, default=50.0
            Percentile to compute (0-100).
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        fmin : float, default=0.0
            Minimum frequency for PSD computation.
        fmax : float, optional
            Maximum frequency for PSD computation.
        psd_method : str, default='welch'
            Method for PSD computation.
        n_per_seg : int, optional
            Length of each segment for Welch's method.
        n_overlap : int, optional
            Number of points to overlap between segments.
        n_fft : int, optional
            Length of the FFT used.
        rois : list of str, optional
            List of ROI names or electrode names to aggregate.
        roi_aggregation_method : list of str, optional
            List of aggregation methods for ROIs ('mean', 'std', 'median', 'min', 'max').
        trial_aggregation_method : list of str, optional
            List of aggregation methods for trials ('mean', 'std', 'median', 'min', 'max').
        epoch_length : float, default=2.0
            Length of epochs in seconds for trial aggregation.
        overlap : float, default=0.0
            Overlap between epochs (0.0 to 0.9).
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.percentile = percentile
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        self.psd_method = psd_method
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.n_fft = n_fft
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.epoch_length = epoch_length
        self.overlap = overlap
        super().__init__(on=on, name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Power Spectral Density Summary.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw or Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD summary statistics.
        """
        import mne
        from mne.utils import _time_mask

        from .utils import filter_to_eeg_channels

        # Get the MNE data object
        data_obj = input["data"]

        # Filter to EEG channels to match PowerSpectralDensityEstimator behavior
        data_obj, eeg_ch_names, _ = filter_to_eeg_channels(data_obj)

        # Handle both Raw and Epochs data
        if hasattr(data_obj, "events"):  # This is Epochs
            epochs = data_obj

            # Crop to time window if specified
            if self.tmin is not None or self.tmax is not None:
                epochs = epochs.copy().crop(tmin=self.tmin, tmax=self.tmax)

            fmax = (
                self.fmax
                if self.fmax is not None
                else epochs.info["sfreq"] / 2
            )

            # Prepare MNE parameters
            mne_params = {}
            if self.n_per_seg is not None:
                mne_params["n_per_seg"] = self.n_per_seg
            if self.n_overlap is not None:
                mne_params["n_overlap"] = self.n_overlap
            if self.n_fft is not None:
                mne_params["n_fft"] = self.n_fft

            # Compute PSD once for all epochs
            spectrum = epochs.compute_psd(
                method="welch",
                fmin=self.fmin,
                fmax=fmax,
                **mne_params,
            )
            psd = spectrum.get_data().astype(
                np.float64, copy=False
            )  # (n_epochs, n_channels, n_freqs)
            freqs = spectrum.freqs
            ch_names = epochs.ch_names

        else:  # Raw data
            raw = data_obj

            # Crop to time window if specified
            if self.tmin is not None or self.tmax is not None:
                raw = raw.copy().crop(tmin=self.tmin, tmax=self.tmax)

            # Create epochs from continuous data if needed
            if self.trial_aggregation_method is not None:
                epochs_data = self._create_epochs_from_continuous(
                    raw
                )  # (n_epochs, n_channels, n_samples)
            else:
                # Single "epoch" from continuous data
                data = raw.get_data()  # Shape: (n_channels, n_times)
                if self.tmin is not None or self.tmax is not None:
                    time_mask = _time_mask(raw.times, self.tmin, self.tmax)
                    data = data[:, time_mask]
                epochs_data = data[
                    np.newaxis, :, :
                ]  # (1, n_channels, n_samples)

            # Wrap as EpochsArray and compute PSD once for all epochs
            info_copy = raw.info.copy()
            epochs_like = mne.EpochsArray(
                epochs_data, info_copy, verbose=False
            )

            fmax = (
                self.fmax if self.fmax is not None else info_copy["sfreq"] / 2
            )

            # Prepare MNE parameters
            mne_params = {}
            if self.n_per_seg is not None:
                mne_params["n_per_seg"] = self.n_per_seg
            if self.n_overlap is not None:
                mne_params["n_overlap"] = self.n_overlap
            if self.n_fft is not None:
                mne_params["n_fft"] = self.n_fft

            spectrum = epochs_like.compute_psd(
                method="welch",
                fmin=self.fmin,
                fmax=fmax,
                **mne_params,
            )
            psd = spectrum.get_data().astype(
                np.float64, copy=False
            )  # (n_epochs, n_channels, n_freqs)
            freqs = spectrum.freqs
            ch_names = epochs_like.ch_names

        n_epochs, n_channels, n_freqs = psd.shape

        # Check if we have any valid epochs
        if n_epochs == 0 or n_channels == 0 or n_freqs == 0:
            # Return empty results for empty epochs
            if self.rois is not None:
                roi_data = {
                    roi: np.array([]).reshape(0, 0) for roi in self.rois
                }
            else:
                roi_data = {ch: np.array([]).reshape(0, 0) for ch in ch_names}

            return apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="psdsummary",
            )

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

        # Handle ROI selection and aggregation logic
        if self.rois is not None:
            # Extract data for specified ROIs
            roi_data = get_data_for_rois(
                psd_summary_values.T,  # Transpose to (n_channels, n_epochs)
                ch_names,
                self.rois,
            )
            # Apply standard aggregation for ROI-based analysis
            results = apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="psdsummary",
            )
        else:
            # SPECIAL HANDLING for PowerSpectralDensitySummary:
            # Clinical literature requires computing percentiles per channel, then aggregating
            # (not combining channels before computing percentiles)
            if (
                self.roi_aggregation_method is not None
                and self.trial_aggregation_method is not None
            ):
                # Both ROI and trial aggregation: compute single scalar value
                from .utils import aggregate_data

                # Step 1: Trial aggregation per channel (if multiple epochs)
                if psd_summary_values.shape[0] > 1:  # Multiple epochs
                    trial_agg_method = self.trial_aggregation_method[0]
                    channel_values = np.array(
                        [
                            aggregate_data(
                                psd_summary_values[:, ch_idx],
                                trial_agg_method,
                            )
                            for ch_idx in range(psd_summary_values.shape[1])
                        ]
                    )
                else:
                    # Single epoch: use values directly
                    channel_values = psd_summary_values[0, :]

                # Step 2: ROI aggregation across channels
                roi_agg_method = self.roi_aggregation_method[0]
                final_value = aggregate_data(channel_values, roi_agg_method)

                # Return single scalar result
                agg_name = f"trial_{trial_agg_method}_roi_{roi_agg_method}"
                results = {
                    "psdsummary": {
                        "data": np.array([[final_value]], dtype=np.float64),
                        "col_names": [f"all_channels_{agg_name}"],
                    }
                }
            else:
                # Fallback to standard aggregation for other cases
                if self.roi_aggregation_method is not None:
                    # ROI aggregation requested: treat all channels as one ROI
                    roi_data = {
                        "all_channels": psd_summary_values.T
                    }  # Shape: (n_channels, n_epochs)
                else:
                    # No ROI aggregation: use each channel as individual ROI
                    roi_data = {
                        ch: psd_summary_values[:, i : i + 1].T
                        for i, ch in enumerate(ch_names)
                    }

                # Apply standard aggregation
                results = apply_roi_trial_aggregation(
                    roi_data,
                    roi_aggregation_methods=self.roi_aggregation_method,
                    trial_aggregation_methods=self.trial_aggregation_method,
                    marker_name="psdsummary",
                )

        return results

    def _create_epochs_from_continuous(self, raw):
        """Create epochs from continuous data.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data object.

        Returns
        -------
        np.ndarray
            Epochs data with shape (n_epochs, n_channels, epoch_samples).
        """
        from mne.utils import _time_mask

        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_times)

        # Apply time mask if specified
        if self.tmin is not None or self.tmax is not None:
            time_mask = _time_mask(raw.times, self.tmin, self.tmax)
            data = data[:, time_mask]

        n_channels, n_samples = data.shape
        sfreq = raw.info["sfreq"]

        # Calculate epoch parameters
        epoch_samples = int(self.epoch_length * sfreq)
        overlap_samples = int(self.overlap * epoch_samples)
        step_samples = max(1, epoch_samples - overlap_samples)

        # Calculate number of epochs
        n_epochs = max(1, (n_samples - epoch_samples) // step_samples + 1)

        # Create epochs array
        epochs_data = np.zeros(
            (n_epochs, n_channels, epoch_samples), dtype=data.dtype
        )

        for epoch_idx in range(n_epochs):
            start_sample = epoch_idx * step_samples
            end_sample = start_sample + epoch_samples

            if end_sample <= n_samples:
                epochs_data[epoch_idx] = data[:, start_sample:end_sample]
            else:
                # Pad with last available samples if needed
                available_samples = n_samples - start_sample
                if available_samples > 0:
                    epochs_data[epoch_idx, :, :available_samples] = data[
                        :, start_sample:
                    ]
                    # Pad with last sample
                    epochs_data[epoch_idx, :, available_samples:] = data[
                        :, -1:
                    ]
                else:
                    # Degenerate case: all padding
                    epochs_data[epoch_idx, :, :] = data[:, -1:]

        return epochs_data
