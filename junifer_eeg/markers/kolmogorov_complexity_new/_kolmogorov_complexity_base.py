"""Base class for Kolmogorov Complexity markers.

This refactoring follows the modular pattern established by spectral_power_new,
permutation_entropy_new, and symbolic_mutual_information_new markers.

Structure:
- Base class: Common functionality (aggregation, ROI filtering, time masking, output type logic)
- Main class: Core Kolmogorov complexity computation (inherits from base)
- Simplified 2-file structure (no bands dimension, unlike spectral/permutation markers)

The refactoring preserves all original functionality while improving code
maintainability and following established patterns.
"""

from typing import Any, ClassVar, List, Union

import numpy as np
from junifer.markers import BaseMarker

from ..utils import aggregate_data, get_data_for_rois


class KolmogorovComplexityBase(BaseMarker):
    """Base class for Kolmogorov Complexity markers with common functionality.

    Provides shared functionality for time masking, ROI filtering, aggregation,
    and output type determination that is common across Kolmogorov Complexity
    marker implementations.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {
            "kolmogorovcomplexity": "timeseries"
        },  # 2D: (epochs, channels)
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        nbins: int = 16,
        rois: Union[List[str], List[int], None] = None,
        channel_aggregation_method: str | None = None,
        trial_aggregation_method: str | None = None,
        equipment: str = "egi256",
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the KolmogorovComplexity marker.

        Parameters
        ----------
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        nbins : int, default=16
            Number of bins for signal symbolization.
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE computation.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1', 'E224') OR semantic ROI (e.g., 'frontal', 'scalp')

            Examples:
            - [0, 1, 2, ..., 223] - NICE scalp ROI via indices
            - ['E1', 'E2', ..., 'E224'] - NICE scalp ROI via names
            - ['scalp'] - Semantic ROI (expands to all scalp channels)
            - [0, 'E5', 10, 'frontal'] - Mix of types

            If None, uses all channels.
        channel_aggregation_method : str, optional
            Methods to aggregate across ROI electrodes/channels: 'mean', 'std',
            'median', 'trim_mean80', 'trim_mean90', etc.
        trial_aggregation_method : str, optional
            Methods to aggregate across trials/epochs: 'mean', 'std',
            'median', 'trim_mean80', 'trim_mean90', etc.
        equipment : str, optional
            Equipment name for named ROIs.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.nbins = nbins
        self.rois = rois
        self.channel_aggregation_method = channel_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
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

    def _prepare_data(
        self, input: dict[str, Any]
    ) -> tuple[np.ndarray, List[str]]:
        """Prepare and filter data for Kolmogorov complexity computation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.

        Returns
        -------
        tuple
            (epochs_data, ch_names) where:
            - epochs_data: (n_epochs, n_channels, n_times) array
            - ch_names: list of channel names

        Raises
        ------
        ValueError
            If input data is not Epochs or if epochs are empty.
        """
        from ..utils import filter_to_eeg_channels

        # Get the MNE object - must be Epochs
        data_obj = input["data"]

        if not hasattr(data_obj, "events"):
            raise ValueError(
                "KolmogorovComplexity requires Epochs data. "
                "Please epoch your data in preprocessing."
            )

        if len(data_obj) == 0:
            raise ValueError(
                "Cannot compute Kolmogorov complexity on empty epochs."
            )

        # Filter to only EEG channels (exclude EOG, stim, etc.)
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Apply time cropping if specified
        if self.tmin is not None or self.tmax is not None:
            data_obj = data_obj.copy().crop(tmin=self.tmin, tmax=self.tmax)

        epochs_data = (
            data_obj.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)
        ch_names = list(data_obj.ch_names)

        return epochs_data, ch_names

    def _apply_roi_filtering(
        self, epochs_data: np.ndarray, ch_names: List[str]
    ) -> tuple[np.ndarray, List[str]]:
        """Apply ROI filtering to the data.

        Parameters
        ----------
        epochs_data : np.ndarray
            Input data with shape (n_epochs, n_channels, n_times)
        ch_names : List[str]
            Channel names

        Returns
        -------
        tuple
            (filtered_data, filtered_ch_names) where:
            - filtered_data: (n_epochs, n_channels_filtered, n_times) array
            - filtered_ch_names: list of filtered channel names
        """
        if self.rois is not None:
            # Transpose to (n_channels, n_epochs, n_samples)
            data_transposed = epochs_data.transpose(1, 0, 2)

            # Get ROI-filtered data
            roi_data_dict = get_data_for_rois(
                data_transposed,
                ch_names,
                self.rois,
                self.equipment,
            )

            # Extract the filtered data (returns {"selected_channels": data})
            if "selected_channels" in roi_data_dict:
                data_filtered = roi_data_dict["selected_channels"]
                # Transpose back to (n_epochs, n_channels, n_samples)
                epochs_data = data_filtered.transpose(1, 0, 2)
                # Preserve the actual ROI channel names
                ch_names = (
                    self.rois if isinstance(self.rois, list) else [self.rois]
                )

        return epochs_data, ch_names

    def _apply_aggregation(
        self, k_values: np.ndarray, ch_names: List[str]
    ) -> dict[str, Any]:
        """Apply aggregation to the computed Kolmogorov complexity values.

        Parameters
        ----------
        k_values : np.ndarray
            Computed complexity values with shape (n_epochs, n_channels)
        ch_names : List[str]
            Channel names

        Returns
        -------
        dict
            Results dictionary with aggregated data and column names
        """
        # Check if we should return raw data without aggregation
        if (
            self.channel_aggregation_method is None
            and self.trial_aggregation_method is None
        ):
            # Return raw per-epoch, per-channel data
            col_names = [f"{ch}" for ch in ch_names]
            return {
                "data": k_values,
                "col_names": col_names,
            }

        # Apply aggregation
        result_data = k_values

        # Step 1: Channel aggregation (aggregate across axis=1)
        if self.channel_aggregation_method is not None:
            result_data = aggregate_data(
                result_data,
                self.channel_aggregation_method,
                axis=1,
            )
            # After channel agg: (n_epochs,)

        # Step 2: Trial aggregation
        if self.trial_aggregation_method is not None:
            if result_data.ndim == 1:
                # Already reduced by channel agg: (n_epochs,)
                result_data = aggregate_data(
                    result_data,
                    self.trial_aggregation_method,
                    axis=None,
                )
                # Result: scalar
            else:
                # No channel agg yet: (n_epochs, n_channels)
                result_data = aggregate_data(
                    result_data,
                    self.trial_aggregation_method,
                    axis=0,
                )
                # Result: (n_channels,)

        # Generate column names based on aggregation and result shape
        if (
            self.channel_aggregation_method is not None
            and self.trial_aggregation_method is not None
        ):
            col_names = ["all_channels_all_trials"]
        elif self.channel_aggregation_method is not None:
            # result_data shape: (n_trials,) after channel aggregation
            n_trials = result_data.shape[0] if result_data.ndim >= 1 else 1
            col_names = [f"trial_{i}" for i in range(n_trials)]
        elif self.trial_aggregation_method is not None:
            col_names = [f"{ch}" for ch in ch_names]
        else:
            col_names = [f"{ch}" for ch in ch_names]

        return {
            "data": result_data,
            "col_names": col_names,
        }

    def _compute_kolmogorov_for_signal(self, signal: np.ndarray) -> float:
        """Compute Kolmogorov complexity for a single signal.

        Uses compression-based approximation with zlib. The signal is first
        symbolized into discrete bins, then compressed, and the complexity
        is measured as the compression ratio.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D array)

        Returns
        -------
        float
            Kolmogorov complexity value (compression ratio)
        """
        import zlib

        # Symbolic transformation - match original algorithm exactly
        ssignal = np.sort(signal)
        items = signal.shape[0]
        first = int(items / 10)
        last = items - first if first > 1 else items - 1
        lower = ssignal[first]
        upper = ssignal[last]
        bsize = (upper - lower) / self.nbins

        # Create symbolic representation
        osignal = np.zeros(signal.shape, dtype=np.uint8)
        maxbin = self.nbins - 1

        for i in range(items):
            tbin = int((signal[i] - lower) / bsize) if bsize > 0 else 0
            osignal[i] = (0 if tbin < 0 else min(tbin, maxbin)) + ord("A")

        # Use zlib for compression
        string = osignal.tobytes()
        cstring = zlib.compress(string)
        return float(len(cstring)) / float(len(string))
