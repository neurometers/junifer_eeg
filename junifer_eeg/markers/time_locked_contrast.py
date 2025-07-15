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

        # Get data and compute averages following NICE approach
        data_a = epochs_a.get_data()  # (n_epochs, n_channels, n_times)
        data_b = epochs_b.get_data()  # (n_epochs, n_channels, n_times)

        # Average across time and epochs for each condition (following NICE)
        # First average across epochs, then across time
        evoked_a = np.mean(data_a, axis=0)  # (n_channels, n_times)
        evoked_b = np.mean(data_b, axis=0)  # (n_channels, n_times)

        # Then average across time
        evoked_a_mean = np.mean(evoked_a, axis=1)  # (n_channels,)
        evoked_b_mean = np.mean(evoked_b, axis=1)  # (n_channels,)

        # Compute contrast (condition_a - condition_b)
        contrast = evoked_a_mean - evoked_b_mean

        # Handle ROI selection
        if self.rois is not None:
            contrast_roi_data = get_data_for_rois(
                contrast.reshape(1, -1).T,  # Transpose to (n_channels, 1)
                list(epochs.ch_names),
                self.rois,
                equipment=self.equipment,
            )
        else:
            # Use all channels as individual ROIs
            contrast_roi_data = {
                ch: contrast[i : i + 1].reshape(1, -1).T
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
