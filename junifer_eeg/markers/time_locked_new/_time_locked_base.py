"""Base class for time-locked markers with shared data preparation logic.

This module provides a conservative base class with helper methods for common
time-locked marker operations, following the same pattern as decoding markers
refactoring.
"""

from typing import Any, List, Tuple, Union

import numpy as np
from junifer.markers import BaseMarker

from .utils import get_data_for_rois


class TimeLockedBase(BaseMarker):
    """Base class for time-locked markers with shared data preparation helpers.

    This class provides common functionality for time-locked markers including:
    - ROI filtering with proper data shape handling
    - Time averaging operations
    - Data preparation utilities

    Following the conservative helper-based pattern from decoding markers.
    """

    def _apply_roi_filtering(
        self,
        data: np.ndarray,
        ch_names: List[str],
        rois: Union[List[str], List[int], None],
        equipment: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply ROI filtering to time-locked data.

        Parameters
        ----------
        data : np.ndarray
            Input data with shape (n_epochs, n_channels, n_times)
        ch_names : list of str
            Channel names corresponding to data
        rois : list of str or int, optional
            ROI specification for channel filtering
        equipment : str
            Equipment type for electrode mapping

        Returns
        -------
        tuple
            (filtered_data, filtered_ch_names) where:
            - filtered_data: np.ndarray with shape (n_epochs, n_channels, n_times)
            - filtered_ch_names: list of str corresponding to filtered channels

        Notes
        -----
        This helper implements the exact ROI filtering logic shared between
        TimeLockedTopography and TimeLockedContrast markers.
        """
        if rois is None:
            return data, ch_names

        # Transpose to (n_channels, n_epochs, n_times) for ROI filtering
        data_transposed = data.transpose(1, 0, 2)

        # Get ROI-filtered data
        roi_data_dict = get_data_for_rois(
            data_transposed,
            ch_names,
            rois,
            equipment,
        )

        # Extract the filtered data (returns {"selected_channels": data})
        if "selected_channels" not in roi_data_dict:
            raise ValueError(f"ROI filtering failed for rois: {rois}")
        data_filtered = roi_data_dict["selected_channels"]

        # Transpose back to (n_epochs, n_channels, n_times)
        data = data_filtered.transpose(1, 0, 2)

        # Update channel names
        ch_names = rois

        return data, ch_names

    def _average_across_time(self, data: np.ndarray) -> np.ndarray:
        """Average data across time dimension.

        Parameters
        ----------
        data : np.ndarray
            Input data with shape (n_epochs, n_channels, n_times)

        Returns
        -------
        np.ndarray
            Time-averaged data with shape (n_epochs, n_channels)

        Notes
        -----
        This helper implements the exact time averaging logic shared between
        TimeLockedTopography and TimeLockedContrast markers.
        """
        return np.mean(data, axis=2)

    def _prepare_epochs_data(
        self, epochs: Any, tmin: float, tmax: float
    ) -> np.ndarray:
        """Prepare epochs data by cropping to specified time window.

        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs object
        tmin : float
            Start time for analysis in seconds
        tmax : float
            End time for analysis in seconds

        Returns
        -------
        np.ndarray
            Cropped epochs data with shape (n_epochs, n_channels, n_times)

        Notes
        -----
        This helper implements the exact time cropping logic shared between
        time-locked markers, including proper boundary handling.
        """
        # Clamp to the available epoch time range to avoid MNE errors
        epoch_min = epochs.tmin
        epoch_max = epochs.tmax
        crop_tmin = tmin
        crop_tmax = tmax
        if crop_tmin is None or crop_tmin < epoch_min:
            crop_tmin = epoch_min
        if crop_tmax is None or crop_tmax > epoch_max:
            crop_tmax = epoch_max

        # Crop epochs to the specified time window
        epochs_cropped = epochs.crop(tmin=crop_tmin, tmax=crop_tmax)

        # Get the raw time-series data
        data = (
            epochs_cropped.get_data()
        )  # Shape: (n_epochs, n_channels, n_times)

        return data

    def _get_epochs_info(self, epochs: Any) -> Tuple[List[str], int, int, int]:
        """Extract basic information from epochs object.

        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs object

        Returns
        -------
        tuple
            (ch_names, n_epochs, n_channels, n_times) where:
            - ch_names: list of channel names
            - n_epochs: number of epochs
            - n_channels: number of channels
            - n_times: number of time points
        """
        ch_names = list(epochs.ch_names)
        n_epochs = len(epochs)
        n_channels = len(ch_names)

        # Get data shape to determine n_times
        data = epochs.get_data()
        n_times = data.shape[2]

        return ch_names, n_epochs, n_channels, n_times
