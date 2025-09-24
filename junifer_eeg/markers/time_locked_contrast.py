"""Time-locked contrast marker for EEG analysis."""

from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class TimeLockedContrast(BaseMarker):
    """Time-locked contrast marker for condition comparisons.

    This marker computes contrasts between different experimental conditions
    in specific time windows, following the ICM Local Global paradigm patterns.
    Essential for analyzing event-related potentials (ERPs) and condition effects.

    Based on the NICE TimeLockedContrast implementation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "timelockedcontrast": "vector",
        },
    }

    def __init__(
        self,
        condition_a: str | List[str],
        condition_b: str | List[str],
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
        comment: Optional[str] = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[str | List[str]] = None,
        trial_aggregation_method: Optional[str | List[str]] = None,
        equipment: str = "standard",
        epoch_length: Optional[float] = None,
        overlap: float = 0.0,
        on: Optional[str | List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize TimeLockedContrast marker."""
        self.condition_a = (
            condition_a if isinstance(condition_a, list) else [condition_a]
        )
        self.condition_b = (
            condition_b if isinstance(condition_b, list) else [condition_b]
        )
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.comment = (
            comment
            or f"{'-'.join(self.condition_a)}_vs_{'-'.join(self.condition_b)}"
        )
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment
        self.epoch_length = epoch_length
        self.overlap = overlap

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute time-locked contrast between conditions."""
        epochs = input["data"]

        # Apply baseline correction if specified
        if self.baseline is not None:
            epochs = epochs.copy().apply_baseline(self.baseline)

        # Filter epochs by conditions
        epochs_a = (
            epochs[self.condition_a]
            if any(cond in epochs.event_id for cond in self.condition_a)
            else None
        )
        epochs_b = (
            epochs[self.condition_b]
            if any(cond in epochs.event_id for cond in self.condition_b)
            else None
        )

        if (
            epochs_a is None
            or epochs_b is None
            or len(epochs_a) == 0
            or len(epochs_b) == 0
        ):
            # Return zeros if conditions not found
            n_channels = len(epochs.ch_names)
            contrast_data = np.zeros((1, n_channels))

            # Handle ROI selection for missing conditions
            if self.rois is not None:
                contrast_roi_data = get_data_for_rois(
                    contrast_data.T,  # Transpose to (n_channels, 1)
                    list(epochs.ch_names),
                    self.rois,
                    equipment=self.equipment,
                )
            else:
                contrast_roi_data = {
                    ch: contrast_data[:, i : i + 1].T
                    for i, ch in enumerate(epochs.ch_names)
                }

            # Apply aggregation
            results = apply_roi_trial_aggregation(
                contrast_roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="timelockedcontrast",
            )

            return results

        # Crop to time window if specified
        if self.tmin is not None or self.tmax is not None:
            epochs_a = epochs_a.copy().crop(tmin=self.tmin, tmax=self.tmax)
            epochs_b = epochs_b.copy().crop(tmin=self.tmin, tmax=self.tmax)

        # Get data and compute contrast following NICE approach exactly
        data_a = epochs_a.get_data()  # (n_epochs, n_channels, n_times)
        data_b = epochs_b.get_data()  # (n_epochs, n_channels, n_times)

        # NICE approach: Average each condition separately first, then compute contrast
        # This matches NICE's TimeLockedTopography -> contrast workflow
        evoked_a = np.mean(
            data_a, axis=0
        )  # (n_channels, n_times) - average across epochs
        evoked_b = np.mean(
            data_b, axis=0
        )  # (n_channels, n_times) - average across epochs
        # Compute contrast between averaged conditions (like NICE)
        contrast_evoked = evoked_a - evoked_b  # (n_channels, n_times)
        # NICE returns the full temporal contrast, not time-averaged
        contrast = contrast_evoked  # Keep temporal dimension like NICE

        # Handle ROI selection - NICE preserves temporal dimension
        if self.rois is not None:
            # For temporal data, we need to handle ROI selection differently
            # Apply ROI selection to each time point
            n_channels, n_times = contrast.shape
            contrast_roi_list = []
            for t in range(n_times):
                time_slice = contrast[:, t].reshape(-1, 1)  # (n_channels, 1)
                roi_data = get_data_for_rois(
                    time_slice,
                    list(epochs.ch_names),
                    self.rois,
                    equipment=self.equipment,
                )
                contrast_roi_list.append(roi_data.flatten())
            contrast_roi_data = np.array(
                contrast_roi_list
            ).T  # (n_rois, n_times)
        else:
            # No ROI selection - create individual channel ROIs with temporal data
            if self.trial_aggregation_method is not None:
                # For trial aggregation, we want to aggregate across time dimension
                # Shape each channel as (1, n_times) so aggregation works correctly
                contrast_roi_data = {
                    ch: contrast[i : i + 1, :]  # (1, n_times) for each channel
                    for i, ch in enumerate(epochs.ch_names)
                }
            else:
                # For no aggregation, keep temporal structure
                contrast_roi_data = {
                    ch: contrast[
                        i : i + 1, :
                    ].T  # (n_times, 1) for each channel
                    for i, ch in enumerate(epochs.ch_names)
                }

        # For NICE compatibility, return temporal data directly without aggregation
        # when no specific aggregation methods are requested
        if (
            self.roi_aggregation_method is None
            and self.trial_aggregation_method is None
            and self.rois is None
        ):
            # Return raw temporal contrast data like NICE
            results = {
                "timelockedcontrast": {
                    "data": contrast,  # Keep as (n_channels, n_times) to match NICE exactly
                    "col_names": list(epochs.ch_names),
                }
            }
        else:
            # Handle temporal aggregation for contrast data
            if self.trial_aggregation_method is not None:
                # For contrast, "trial_aggregation" means temporal aggregation
                # since contrast is already computed between condition averages
                from .utils import aggregate_data

                if self.trial_aggregation_method == "mean":
                    # Average across time dimension
                    aggregated_contrast = np.mean(
                        contrast, axis=1, keepdims=True
                    )  # (n_channels, 1)
                else:
                    # Use general aggregation function
                    aggregated_contrast = aggregate_data(
                        contrast, self.trial_aggregation_method, axis=1
                    )
                    if aggregated_contrast.ndim == 1:
                        aggregated_contrast = aggregated_contrast.reshape(
                            -1, 1
                        )
                results = {
                    "timelockedcontrast": {
                        "data": aggregated_contrast.T,  # (1, n_channels) to match expected format
                        "col_names": list(epochs.ch_names),
                    }
                }
                # Skip the HDF5 transpose for aggregated case since we already have correct format
                return results
            else:
                # Apply ROI aggregation only
                # Convert single string to list for aggregation function
                roi_agg_methods = (
                    [self.roi_aggregation_method]
                    if isinstance(self.roi_aggregation_method, str)
                    else self.roi_aggregation_method
                )

                results = apply_roi_trial_aggregation(
                    contrast_roi_data,
                    roi_aggregation_methods=roi_agg_methods,
                    trial_aggregation_methods=None,
                    marker_name="timelockedcontrast",
                )

            # Fix output format to match HDF5: transpose from (1, n_channels) to (n_channels, 1)
            if "timelockedcontrast" in results:
                data = results["timelockedcontrast"]["data"]
                if data.shape[0] == 1 and data.shape[1] > 1:
                    # Transpose to column format to match HDF5
                    results["timelockedcontrast"]["data"] = data.T

        return results
