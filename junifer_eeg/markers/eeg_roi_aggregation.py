"""EEG ROI and channel aggregation marker."""

from typing import Any, ClassVar, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker
from junifer.utils import logger

from .utils import aggregate_data, get_data_for_rois

__all__ = ["EEGROIAggregation"]


@register_marker
class EEGROIAggregation(BaseMarker):
    """Aggregator for EEG channel/ROI data.

    Similar to junifer's ParcelAggregation but designed for EEG data.
    Supports channel-wise aggregation, trial/epoch aggregation, and ROI
    selection for EEG channels.

    Parameters
    ----------
    rois : list of str or int, optional
        Flat list of channel specifications for filtering.
        Each item can be:
        - int: channel index (e.g., 0, 1, 223)
        - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp')

        If None, uses all channels (default None).
    channel_method : str, optional
        The method to perform aggregation across channels.
        Options: 'mean', 'std', 'median', 'min', 'max', 'sum',
        'trim_mean80', 'trim_mean90'.
        If None, will not aggregate across channels (default None).
    trial_method : str, optional
        The method to use to aggregate across trials/epochs.
        Options: 'mean', 'std', 'median', 'min', 'max', 'sum',
        'trim_mean80', 'trim_mean90'.
        If None, will not aggregate across trials (default None).
    equipment : str, optional
        Equipment configuration for ROI resolution (default "egi256").
    on : str or list of str, optional
        The data types to apply the marker to. If None, will work on all
        available data (default "EEG").
    name : str, optional
        The name of the marker. If None, will use the class name
        (default None).

    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {
            "aggregation": "vector",
        },
    }

    def __init__(
        self,
        rois: Union[list[str], list[int], None] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        equipment: str = "egi256",
        on: Union[list[str], str, None] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize EEG ROI aggregation marker."""
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method
        self.equipment = equipment

        super().__init__(on=on or "EEG", name=name)

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict] = None,
    ) -> dict:
        """Compute aggregation.

        Parameters
        ----------
        input : dict
            A single input from the pipeline data object in which to compute
            the marker. Expected to have 'data' key with array of shape:
            - (n_epochs, n_channels) for 2D data
            - (n_bands, n_epochs, n_channels) for 3D band data
            - (n_channels, n_times) for time-locked data
        extra_input : dict, optional
            The other fields in the pipeline data object. Useful for
            accessing other data kind that needs to be used in the
            computation (default None).

        Returns
        -------
        dict
            The computed result as dictionary. This will be either returned
            to the user or stored in the storage by calling the store method
            with this as a parameter. The dictionary has the following keys:

            * ``aggregation`` : dictionary with the following keys:

                - ``data`` : aggregated values as ``numpy.ndarray``
                - ``col_names`` : channel/ROI labels as list of str
                  (only if no aggregation applied)

        """
        logger.debug("EEG ROI aggregation")

        # Get input data
        input_data = input["data"]

        # Handle channel names if available
        ch_names = input.get("col_names", None)

        # Determine data shape and apply ROI filtering if needed
        if self.rois is not None and ch_names is not None:
            logger.debug(f"Filtering to ROIs: {self.rois}")

            # Handle different data shapes
            if input_data.ndim == 2:
                # (n_epochs, n_channels)
                roi_data = get_data_for_rois(
                    input_data.T,
                    list(ch_names),
                    self.rois,
                    self.equipment,
                )
                if "selected_channels" in roi_data:
                    input_data = roi_data["selected_channels"].T
                    ch_names = roi_data.get("channel_names", ch_names)
            elif input_data.ndim == 3:
                # (n_bands, n_epochs, n_channels)
                n_bands = input_data.shape[0]
                filtered_bands = []
                for band_idx in range(n_bands):
                    roi_data = get_data_for_rois(
                        input_data[band_idx].T,
                        list(ch_names),
                        self.rois,
                        self.equipment,
                    )
                    if "selected_channels" in roi_data:
                        filtered_bands.append(roi_data["selected_channels"].T)
                        ch_names = roi_data.get("channel_names", ch_names)
                    else:
                        filtered_bands.append(input_data[band_idx])
                input_data = np.stack(filtered_bands, axis=0)
            else:
                logger.warning(
                    f"Unsupported data shape {input_data.shape} for ROI "
                    "filtering. Skipping ROI selection."
                )

        # Apply channel aggregation if requested
        if self.channel_method is not None:
            logger.debug(
                f"Aggregating across channels using {self.channel_method}"
            )

            # Apply based on data shape - channels are always last axis
            if input_data.ndim == 2:
                # (n_epochs, n_channels) -> (n_epochs,)
                input_data = aggregate_data(
                    input_data, self.channel_method, axis=1
                )
            elif input_data.ndim == 3:
                # (n_bands, n_epochs, n_channels) -> (n_bands, n_epochs)
                input_data = aggregate_data(
                    input_data, self.channel_method, axis=2
                )
            else:
                # Handle other cases - last axis is typically channels
                input_data = aggregate_data(
                    input_data, self.channel_method, axis=-1
                )

            # Channel names no longer relevant after aggregation
            ch_names = None

        # Apply trial/epoch aggregation if requested
        if self.trial_method is not None:
            logger.debug(
                f"Aggregating across trials/epochs using {self.trial_method}"
            )

            # Apply based on data shape after channel aggregation
            if input_data.ndim == 1:
                # (n_epochs,) -> scalar
                input_data = aggregate_data(
                    input_data, self.trial_method, axis=None
                )
            elif input_data.ndim == 2:
                # Could be (n_epochs, n_channels) or (n_bands, n_epochs)
                # Determine based on previous channel aggregation
                if self.channel_method is None:
                    # (n_epochs, n_channels) -> (n_channels,)
                    input_data = aggregate_data(
                        input_data, self.trial_method, axis=0
                    )
                else:
                    # (n_bands, n_epochs) -> (n_bands,)
                    input_data = aggregate_data(
                        input_data, self.trial_method, axis=1
                    )
            elif input_data.ndim == 3:
                # (n_bands, n_epochs, n_channels) -> (n_bands, n_channels)
                input_data = aggregate_data(
                    input_data, self.trial_method, axis=1
                )
            else:
                # General case - first axis is typically trials/epochs
                input_data = aggregate_data(
                    input_data, self.trial_method, axis=0
                )

        # Format output
        result = {
            "aggregation": {
                "data": input_data,
            }
        }

        # Include channel names if they're still relevant
        if ch_names is not None:
            result["aggregation"]["col_names"] = ch_names

        return result
