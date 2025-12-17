"""Enhanced Power Spectral Density markers for junifer_eeg."""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker

from .utils import get_data_for_rois


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
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"psdsummary": "timeseries"}
    }  # 2D: (epochs, channels)

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
        rois: Union[List[str], List[int], None] = None,
        channel_method: str | None = None,
        trial_method: str | None = None,
        equipment: str = "egi256",
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
        rois : list of str or int, optional
            Flat list of channel specifications for filtering AFTER PSD computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp')

            If None, uses all channels.
        channel_method : str, optional
            Aggregation method for ROIs ('mean', 'std', 'median', 'min', 'max').
        trial_method : str, optional
            Aggregation method for trials ('mean', 'std', 'median', 'min', 'max').
        equipment : str, default="egi256"
            Equipment configuration for ROI resolution.
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
        self.channel_method = channel_method
        self.trial_method = trial_method
        self.equipment = equipment
        super().__init__(on=on, name=name)

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns:
        - 'timeseries': 2D tensor data (no aggregation)
        - 'vector': 1D array (one aggregation applied)
        - 'scalar_table': scalar value (both aggregations applied)
        """
        # No aggregation → 2D tensor (epochs, channels) → use timeseries
        if self.channel_method is None and self.trial_method is None:
            return "timeseries"

        # Both aggregations → scalar → use scalar_table
        if self.channel_method is not None and self.trial_method is not None:
            return "scalar_table"

        # One aggregation → 1D array → use vector
        return "vector"

    def compute(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute Power Spectral Density Summary.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed PSD summary statistics.

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
                "PowerSpectralDensitySummary requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(data_obj) == 0:
            raise ValueError("Cannot compute PSD summary on empty epochs.")

        # Filter to EEG channels to match PowerSpectralDensityEstimator behavior
        data_obj, eeg_ch_names, _ = filter_to_eeg_channels(data_obj)

        epochs = data_obj

        # Crop to time window if specified
        if self.tmin is not None or self.tmax is not None:
            epochs = epochs.copy().crop(tmin=self.tmin, tmax=self.tmax)

        fmax = self.fmax if self.fmax is not None else epochs.info["sfreq"] / 2

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

        n_epochs, n_channels, n_freqs = psd.shape

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

        # Apply ROI filtering BEFORE aggregation if specified
        if self.rois is not None:
            roi_data_dict = get_data_for_rois(
                psd_summary_values.T,  # Transpose to (n_channels, n_epochs)
                ch_names,
                self.rois,
                self.equipment,
            )
            # Extract filtered data and transpose back to (n_epochs, n_channels)
            if "selected_channels" in roi_data_dict:
                psd_summary_values = roi_data_dict["selected_channels"].T
                ch_names = self.rois

        # SPECIAL HANDLING for PowerSpectralDensitySummary:
        # Clinical literature requires computing percentiles per channel, then aggregating
        # (not combining channels before computing percentiles)
        if self.channel_method is not None and self.trial_method is not None:
            # Both channel and trial aggregation: compute single scalar value
            from .utils import aggregate_data

            # Step 1: Trial aggregation per channel
            if psd_summary_values.shape[0] > 1:  # Multiple epochs
                channel_values = np.array(
                    [
                        aggregate_data(
                            psd_summary_values[:, ch_idx],
                            self.trial_method,
                        )
                        for ch_idx in range(psd_summary_values.shape[1])
                    ]
                )
            else:
                # Single epoch: use values directly
                channel_values = psd_summary_values[0, :]

            # Step 2: Channel aggregation across all channels
            final_value = aggregate_data(channel_values, self.channel_method)

            # Return single scalar result without unnecessary reshaping
            agg_name = f"trial_{self.trial_method}_roi_{self.channel_method}"
            results = {
                "psdsummary": {
                    "data": float(
                        final_value
                    ),  # Return scalar, not (1, 1) array
                    "col_names": [f"all_channels_{agg_name}"],
                }
            }
        else:
            # Standard aggregation: apply aggregation manually
            # Check for no aggregation
            if self.channel_method is None and self.trial_method is None:
                col_names = [f"{ch}" for ch in ch_names]
                results = {
                    "psdsummary": {
                        "data": psd_summary_values,
                        "col_names": col_names,
                    }
                }
            else:
                # Apply aggregation
                from .utils import aggregate_data

                result_data = psd_summary_values

                # Channel aggregation
                if self.channel_method is not None:
                    result_data = aggregate_data(
                        result_data, self.channel_method, axis=1
                    )

                # Trial aggregation
                if self.trial_method is not None:
                    if result_data.ndim == 1:
                        result_data = aggregate_data(
                            result_data,
                            self.trial_method,
                            axis=None,
                        )
                    else:
                        result_data = aggregate_data(
                            result_data, self.trial_method, axis=0
                        )

                # Return result data without unnecessary reshaping - preserve tensor structure
                # Scalar: keep as scalar
                # 1D array: keep as 1D (n_trials) or (n_channels)
                # 2D array: keep as 2D (n_trials, n_channels)

                # Generate column names based on aggregation and result shape
                if (
                    self.channel_method is not None
                    and self.trial_method is not None
                ):
                    col_names = ["all_channels_all_trials"]
                elif self.channel_method is not None:
                    # result_data shape: (n_trials,) after channel aggregation
                    n_trials = (
                        result_data.shape[0] if result_data.ndim >= 1 else 1
                    )
                    col_names = [f"trial_{i}" for i in range(n_trials)]
                elif self.trial_method is not None:
                    col_names = [f"{ch}" for ch in ch_names]
                else:
                    col_names = [f"{ch}" for ch in ch_names]

                results = {
                    "psdsummary": {
                        "data": result_data,
                        "col_names": col_names,
                    }
                }

        return results
