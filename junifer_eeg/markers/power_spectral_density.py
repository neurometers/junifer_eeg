"""Enhanced Power Spectral Density markers for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


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
        }
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
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute Power Spectral Density.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD data, frequencies, and normalized PSD.
        """
        # Get the MNE Raw object
        raw = input["data"]

        # Crop to time window if specified
        if self.tmin is not None or self.tmax is not None:
            raw_cropped = raw.copy().crop(tmin=self.tmin, tmax=self.tmax)
        else:
            raw_cropped = raw

        # Set frequency limits
        fmax = self.fmax if self.fmax is not None else raw.info["sfreq"] / 2

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
                method="welch", fmin=self.fmin, fmax=fmax, **mne_params
            )

            # Extract data and frequencies
            psd_data = spectrum.get_data()  # Shape: (n_channels, n_freqs)
            freqs = spectrum.freqs

        else:
            raise ValueError(
                f"PSD method '{self.psd_method}' not supported. Use 'welch'."
            )

        # Compute normalized version
        psd_data_norm = psd_data / psd_data.sum(axis=-1, keepdims=True)

        # Generate column names for frequencies
        freq_names = [f"freq_{freq:.2f}Hz" for freq in freqs]

        # Return data in junifer format
        return {
            "psd_data": {
                "data": psd_data,  # Shape: (n_channels, n_freqs)
                "col_names": freq_names,
                "row_names": list(raw.ch_names),
            },
            "psd_freqs": {
                "data": freqs.reshape(1, -1),  # Shape: (1, n_freqs)
                "col_names": freq_names,
            },
            "psd_data_norm": {
                "data": psd_data_norm,  # Shape: (n_channels, n_freqs)
                "col_names": freq_names,
                "row_names": list(raw.ch_names),
            },
        }


@register_marker
class PowerSpectralDensitySummary(BaseMarker):
    """Power Spectral Density Summary using percentile statistics.

    This marker computes summary statistics (percentiles) of the power
    spectral density across frequencies, adapted from the NICE package.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {"EEG": {"psd_summary": "vector"}}

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
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute Power Spectral Density Summary.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD summary statistics.
        """
        # Use the PowerSpectralDensityEstimator to get PSD data
        estimator = PowerSpectralDensityEstimator(
            tmin=self.tmin,
            tmax=self.tmax,
            fmin=self.fmin,
            fmax=self.fmax,
            psd_method=self.psd_method,
            n_per_seg=self.n_per_seg,
            n_overlap=self.n_overlap,
            n_fft=self.n_fft,
        )

        psd_result = estimator.compute(input, extra_input)
        psd_data = psd_result["psd_data"][
            "data"
        ]  # Shape: (n_channels, n_freqs)

        # Get the MNE Raw object for channel names
        raw = input["data"]

        # Compute percentile across frequencies for each channel
        summary_values = np.percentile(psd_data, self.percentile, axis=1)

        # Return data in junifer format
        return {
            "psd_summary": {
                "data": summary_values.reshape(
                    1, -1
                ),  # Shape: (1, n_channels)
                "col_names": [
                    f"{ch}_psd_p{self.percentile}" for ch in raw.ch_names
                ],
            }
        }
