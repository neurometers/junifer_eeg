"""Abstract base classes for EEG markers."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from junifer.markers import BaseMarker
from junifer.utils import raise_error

__all__ = [
    "EEGBaseMarker",
    "EEGEpochsMarker",
    "format_marker_result",
]


def format_marker_result(
    feature_name: str,
    data: np.ndarray,
    col_names: Optional[List[str]] = None,
    channel_aggregated: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Format marker result with consistent col_names handling.

    This is the SINGLE SOURCE OF TRUTH for formatting marker outputs.
    All markers should use this function to ensure consistent storage
    of channel/column names alongside data.

    The col_names are CRITICAL for:
    1. Preserving epoch/channel ordering from input to output
    2. Enabling the reader to correctly interpret dimensions
    3. Ensuring data traceability back to original channels

    Parameters
    ----------
    feature_name : str
        Name of the output feature (e.g., 'spectralpower', 'permutationentropy').
    data : np.ndarray
        The computed marker data.
    col_names : list of str, optional
        Column/channel names. Should be provided when channel dimension exists.
        Typically this is the list of channel names from the input data.
    channel_aggregated : bool, default=False
        Whether channel aggregation was applied. If True, col_names will NOT
        be stored since they no longer correspond to individual channels.

    Returns
    -------
    dict
        Formatted result dictionary ready for storage:
        {feature_name: {'data': data, 'col_names': col_names}}  # if col_names valid
        {feature_name: {'data': data}}  # if no col_names or channel_aggregated

    Examples
    --------
    >>> # Marker without aggregation - ALWAYS include col_names
    >>> result = format_marker_result(
    ...     'spectralpower',
    ...     data,  # shape: (n_epochs, n_channels)
    ...     col_names=ch_names,  # ['E1', 'E2', ..., 'E256']
    ...     channel_aggregated=False
    ... )

    >>> # Marker with channel aggregation - col_names not stored
    >>> result = format_marker_result(
    ...     'spectralpower',
    ...     data,  # shape: (n_epochs,) after channel mean
    ...     col_names=ch_names,  # Will be ignored
    ...     channel_aggregated=True
    ... )

    """
    result: Dict[str, Any] = {"data": data}

    # Only store col_names if:
    # 1. They are provided
    # 2. Channel aggregation was NOT applied
    # 3. Data is not scalar
    if (
        col_names is not None
        and not channel_aggregated
        and not np.isscalar(data)
        and (hasattr(data, "ndim") and data.ndim > 0)
    ):
        # Ensure col_names is a list (not tuple or other)
        result["col_names"] = list(col_names)

    return {feature_name: result}


class EEGBaseMarker(BaseMarker):
    """Abstract base class for all EEG markers.

    Provides common functionality for EEG markers including:
    - Time window selection (tmin, tmax)
    - Equipment configuration
    - Data validation

    Parameters
    ----------
    tmin : float, optional
        Start time for analysis in seconds. If None, use start of data
        (default None).
    tmax : float, optional
        End time for analysis in seconds. If None, use end of data
        (default None).
    equipment : str, optional
        Equipment configuration for ROI resolution (default "egi256").
    on : str or list of str, optional
        Data types to apply the marker to (default "EEG").
    name : str, optional
        Name of the marker. If None, will use class name (default None).

    """

    def __init__(
        self,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize EEG base marker."""
        self.tmin = tmin
        self.tmax = tmax
        self.equipment = equipment
        super().__init__(on=on or "EEG", name=name)

    def _apply_time_window(self, data_obj):
        """Apply time window cropping to data.

        Parameters
        ----------
        data_obj : mne.io.Raw or mne.Epochs
            MNE data object to crop.

        Returns
        -------
        mne.io.Raw or mne.Epochs
            Cropped data object (copy if cropping applied, original otherwise).

        """
        if self.tmin is not None or self.tmax is not None:
            return data_obj.copy().crop(tmin=self.tmin, tmax=self.tmax)
        return data_obj

    @abstractmethod
    def _validate_input(self, data_obj) -> None:
        """Validate input data type.

        Parameters
        ----------
        data_obj : mne.io.Raw or mne.Epochs
            MNE data object to validate.

        Raises
        ------
        ValueError
            If data_obj is not the expected type.

        """
        raise_error(
            msg="Concrete classes need to implement _validate_input().",
            klass=NotImplementedError,
        )


class EEGEpochsMarker(EEGBaseMarker):
    """Abstract base class for markers that require Epochs data.

    Provides validation to ensure input data is MNE Epochs with events.

    Parameters
    ----------
    tmin : float, optional
        Start time for analysis in seconds. If None, use start of epochs
        (default None).
    tmax : float, optional
        End time for analysis in seconds. If None, use end of epochs
        (default None).
    equipment : str, optional
        Equipment configuration for ROI resolution (default "egi256").
    on : str or list of str, optional
        Data types to apply the marker to (default "EEG").
    name : str, optional
        Name of the marker. If None, will use class name (default None).

    """

    def _validate_input(self, data_obj) -> None:
        """Validate input is Epochs data.

        Parameters
        ----------
        data_obj : mne.Epochs
            MNE Epochs object to validate.

        Raises
        ------
        ValueError
            If data_obj is not Epochs or if epochs are empty.

        """
        if not hasattr(data_obj, "events"):
            raise_error(
                msg=f"{self.__class__.__name__} requires Epochs data. "
                "Please epoch your data in preprocessing.",
                klass=ValueError,
            )

        if len(data_obj) == 0:
            raise_error(
                msg=f"Cannot compute {self.__class__.__name__} on empty epochs.",
                klass=ValueError,
            )
