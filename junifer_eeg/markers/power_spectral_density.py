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
        # Get the MNE data object
        data_obj = input["data"]

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
        import mne.io
        from mne.utils import _time_mask

        # Get the MNE data object
        data_obj = input["data"]

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

            # Get epochs data directly
            epochs_data = (
                epochs_cropped.get_data()
            )  # Shape: (n_epochs, n_channels, n_samples)
            ch_names = epochs.ch_names

        else:  # Raw data
            raw = data_obj

            # Create epochs from continuous data if needed
            if self.trial_aggregation_method is not None:
                epochs_data = self._create_epochs_from_continuous(raw)
                # epochs_data shape: (n_epochs, n_channels, n_samples)
            else:
                # Single "epoch" from continuous data
                data = raw.get_data()  # Shape: (n_channels, n_times)
                if self.tmin is not None or self.tmax is not None:
                    time_mask = _time_mask(raw.times, self.tmin, self.tmax)
                    data = data[:, time_mask]
                epochs_data = data[
                    np.newaxis,
                    :,
                    :,
                ]  # Shape: (1, n_channels, n_samples)

            ch_names = raw.ch_names

        n_epochs, n_channels_raw, n_samples = epochs_data.shape

        # We will determine the actual number of PSD channels dynamically from the first epoch
        psd_summary_values: np.ndarray | None = None

        for epoch_idx in range(n_epochs):
            epoch_data = epochs_data[
                epoch_idx
            ]  # Shape: (n_channels, n_samples)

            # Create a temporary Raw object for this epoch
            if hasattr(data_obj, "events"):  # Epochs
                temp_info = epochs.info.copy()
            else:  # Raw
                temp_info = raw.info.copy()

            temp_raw = mne.io.RawArray(epoch_data, temp_info, verbose=False)

            # Use the PowerSpectralDensityEstimator to get PSD data for this epoch
            estimator = PowerSpectralDensityEstimator(
                tmin=None,  # Already segmented
                tmax=None,  # Already segmented
                fmin=self.fmin,
                fmax=self.fmax,
                psd_method=self.psd_method,
                n_per_seg=self.n_per_seg,
                n_overlap=self.n_overlap,
                n_fft=self.n_fft,
            )

            psd_result = estimator.compute({"data": temp_raw}, extra_input)
            psd_data = psd_result["psd_data"][
                "data"
            ]  # Shape: (n_psd_ch, n_freqs)
            n_psd_ch = psd_data.shape[0]
            # allocate storage on first epoch
            if psd_summary_values is None:
                psd_summary_values = np.zeros(
                    (n_epochs, n_psd_ch), dtype=np.float64
                )

            # Compute percentile across frequencies for each channel
            if psd_data.ndim == 2 and psd_data.shape[1] > 1:
                # Multiple frequency bins - compute percentile across frequencies
                psd_summary_values[epoch_idx, :] = np.percentile(
                    psd_data,
                    self.percentile,
                    axis=1,
                )
            else:
                # Single frequency bin or 1D array - use the values directly
                if psd_data.ndim == 2:
                    psd_summary_values[epoch_idx, :] = psd_data[:, 0]
                else:
                    # Ensure the array has the right shape
                    if psd_data.shape[0] == psd_summary_values.shape[1]:
                        psd_summary_values[epoch_idx, :] = psd_data
                    else:
                        # If shape doesn't match, use the first value for all channels
                        psd_summary_values[epoch_idx, :] = psd_data[0]

        # Check if we have any valid epochs
        if psd_summary_values is None or n_epochs == 0:
            # Return empty results for empty epochs
            if self.rois is not None:
                roi_data = {
                    roi: np.array([]).reshape(0, 0) for roi in self.rois
                }
            else:
                roi_data = {ch: np.array([]).reshape(0, 0) for ch in ch_names}
        else:
            # Handle ROI selection
            if self.rois is not None:
                # Extract data for specified ROIs
                roi_data = get_data_for_rois(
                    psd_summary_values.T,  # Transpose to (n_channels, n_epochs)
                    ch_names,
                    self.rois,
                )
            else:
                # Use all channels as individual ROIs
                roi_data = {
                    ch: psd_summary_values[:, i : i + 1].T
                    for i, ch in enumerate(ch_names)
                }

        # Apply aggregation
        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="psdsummary",
        )

        return results

    def _create_epochs_from_continuous(self, raw):
        """Create epochs from continuous data."""
        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_times)

        # Apply time mask if specified
        if self.tmin is not None or self.tmax is not None:
            from mne.utils import _time_mask

            time_mask = _time_mask(raw.times, self.tmin, self.tmax)
            data = data[:, time_mask]

        n_channels, n_samples = data.shape
        sfreq = raw.info["sfreq"]

        # Calculate epoch parameters
        epoch_samples = int(self.epoch_length * sfreq)
        overlap_samples = int(self.overlap * epoch_samples)
        step_samples = epoch_samples - overlap_samples

        # Calculate number of epochs
        n_epochs = max(1, (n_samples - epoch_samples) // step_samples + 1)

        # Create epochs
        epochs_data = np.zeros((n_epochs, n_channels, epoch_samples))

        for epoch_idx in range(n_epochs):
            start_sample = epoch_idx * step_samples
            end_sample = start_sample + epoch_samples

            if end_sample <= n_samples:
                epochs_data[epoch_idx] = data[:, start_sample:end_sample]
            else:
                # Pad with last available samples if needed
                available_samples = n_samples - start_sample
                epochs_data[epoch_idx, :, :available_samples] = data[
                    :,
                    start_sample:,
                ]
                # Pad with zeros or repeat last sample
                epochs_data[epoch_idx, :, available_samples:] = data[:, -1:]

        return epochs_data
