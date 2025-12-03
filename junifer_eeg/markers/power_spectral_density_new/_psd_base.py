"""Base class for Power Spectral Density markers.

This refactoring follows the modular pattern established by other markers,
but with a different structure since PSD contains two separate markers
(Estimator and Summary) rather than variants of the same marker.

Structure:
- Base class: Shared PSD computation logic (Welch params, time/freq windowing, MNE compute_psd wrapper)
- Estimator: Inherits base, returns raw PSD data/freqs (handles Raw + Epochs)
- Summary: Inherits base, adds SEF/percentile computation, returns summary stats (Epochs only)
"""

from typing import Any, ClassVar, Tuple

import numpy as np
from junifer.markers import BaseMarker


class PowerSpectralDensityBase(BaseMarker):
    """Base class for Power Spectral Density markers with common functionality.

    Provides shared functionality for PSD computation including time masking,
    frequency windowing, channel filtering, and MNE compute_psd wrapper.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

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
        """Initialize the Power Spectral Density base marker.

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

    def _prepare_data(self, input: dict[str, Any]) -> Tuple[Any, list]:
        """Prepare and filter data for PSD computation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw or Epochs object.

        Returns
        -------
        tuple
            (data_obj, ch_names) where:
            - data_obj: Filtered MNE Raw or Epochs object
            - ch_names: list of channel names

        Raises
        ------
        ValueError
            If PSD method is not supported.
        """
        from ..utils import filter_to_eeg_channels

        # Get the MNE data object
        data_obj = input["data"]

        # Filter to only EEG channels (E1-E256), excluding D/DI auxiliary channels
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Validate PSD method
        if self.psd_method != "welch":
            raise ValueError(
                f"PSD method '{self.psd_method}' not supported. Use 'welch'."
            )

        return data_obj, eeg_ch_names

    def _compute_psd(
        self, data_obj: Any, preserve_epochs: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Compute Power Spectral Density using MNE.

        Parameters
        ----------
        data_obj : MNE Raw or Epochs object
            Input data object (already filtered to EEG channels)
        preserve_epochs : bool, default=True
            If True, keep epochs separate for Summary marker.
            If False, average across epochs for Estimator marker.

        Returns
        -------
        tuple
            (psd_data, freqs, ch_names) where:
            - psd_data: PSD data array (shape depends on preserve_epochs)
            - freqs: Frequency array
            - ch_names: Channel names
        """
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
            ch_names = epochs.ch_names

            # Average across epochs if requested (for Estimator)
            if not preserve_epochs:
                psd_data = np.mean(
                    psd_data, axis=0
                )  # Shape: (n_channels, n_freqs)

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
            ch_names = raw.ch_names

        return psd_data.astype(np.float64, copy=False), freqs, ch_names
