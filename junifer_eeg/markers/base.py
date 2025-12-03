"""Abstract base classes for EEG markers."""

from abc import abstractmethod
from typing import Optional

from junifer.markers import BaseMarker
from junifer.utils import raise_error

__all__ = ["EEGBaseMarker", "EEGEpochsMarker", "EEGRawMarker"]


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


class EEGRawMarker(EEGBaseMarker):
    """Abstract base class for markers that work on Raw data.

    Provides validation to ensure input data is MNE Raw continuous data.

    Parameters
    ----------
    tmin : float, optional
        Start time for analysis in seconds. If None, use start of recording
        (default None).
    tmax : float, optional
        End time for analysis in seconds. If None, use end of recording
        (default None).
    equipment : str, optional
        Equipment configuration for ROI resolution (default "egi256").
    on : str or list of str, optional
        Data types to apply the marker to (default "EEG").
    name : str, optional
        Name of the marker. If None, will use class name (default None).

    """

    def _validate_input(self, data_obj) -> None:
        """Validate input is Raw data.

        Parameters
        ----------
        data_obj : mne.io.Raw
            MNE Raw object to validate.

        Raises
        ------
        ValueError
            If data_obj is not Raw continuous data.

        """
        if not hasattr(data_obj, "times"):
            raise_error(
                msg=f"{self.__class__.__name__} requires Raw continuous data.",
                klass=ValueError,
            )
