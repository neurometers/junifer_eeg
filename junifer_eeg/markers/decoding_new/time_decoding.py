"""Time decoding marker implementation."""

from typing import Any, ClassVar, Dict, Optional

import numpy as np
from junifer.api.decorators import register_marker

from ._decoding_base import DecodingBase


@register_marker
class TimeDecoding(DecodingBase):
    """Time decoding marker for temporal classification analysis.

    This marker performs decoding analysis across time using MNE's Sliding
    Estimator to classify between experimental conditions at each time point.
    Essential for analyzing the temporal dynamics of discriminative neural
    patterns.

    The marker applies optional ROI filtering BEFORE decoding to restrict
    classification to specific channels.
    """

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "timedecoding": "timeseries",  # 2D: (1, n_times) time series
        },
    }

    def __init__(
        self,
        condition_a: str | list[str],
        condition_b: str | list[str],
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        n_splits: int = 5,
        scoring: str = "roc_auc",
        random_state: Optional[int] = 42,
        comment: Optional[str] = None,
        rois: list[str] | list[int] | None = None,
        equipment: str = "egi256",
        on: Optional[str | list[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize TimeDecoding marker.

        Parameters
        ----------
        condition_a : str or list of str
            Condition(s) for class A.
        condition_b : str or list of str
            Condition(s) for class B.
        tmin : float, optional
            Start time for decoding window.
        tmax : float, optional
            End time for decoding window.
        n_splits : int, default=5
            Number of cross-validation folds.
        scoring : str, default='roc_auc'
            Scoring metric ('roc_auc' or 'accuracy').
        random_state : int, optional
            Random state for reproducibility.
        comment : str, optional
            Label for this decoding analysis.
        rois : list of str or int, optional
            Flat list of channel specifications for filtering BEFORE decoding.
            Each item can be:
            - int: channel index (e.g., 0, 1, 223)
            - str: channel name (e.g., 'E1') OR semantic ROI (e.g., 'scalp', 'frontal')

            Examples:
            - ['frontal'] - Decode from frontal channels only
            - list(range(32)) - Decode from first 32 channels
            - None - Use all channels (default)

            **NOTE:** ROI filtering restricts which channels contribute to decoding.
        equipment : str, default='egi256'
            Equipment type for electrode mapping.
        """
        # Set default comment if not provided
        if comment is None:
            comment = f"{'-'.join(condition_a if isinstance(condition_a, list) else [condition_a])}_vs_{'-'.join(condition_b if isinstance(condition_b, list) else [condition_b])}_time_decoding"

        super().__init__(
            condition_a=condition_a,
            condition_b=condition_b,
            tmin=tmin,
            tmax=tmax,
            n_splits=n_splits,
            scoring=scoring,
            random_state=random_state,
            comment=comment,
            rois=rois,
            equipment=equipment,
            on=on,
            name=name,
        )

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute time decoding between conditions.

        Process:
        1. Filter epochs by condition
        2. Apply ROI filtering (if specified) to select channels
        3. Crop to time window (if specified)
        4. Use SlidingEstimator to classify at each time point
        5. Return time series of scores (one per time point), averaged across folds
        """
        from mne.decoding import SlidingEstimator, cross_val_multiscore

        # Prepare epochs data using base class helper
        X, y, ch_names, valid_data = self._prepare_epochs_data(input)

        # Check if conditions were found
        if not valid_data:
            return {
                "timedecoding": {
                    "data": np.full((1, 100), 0.5),
                    "col_names": [f"t_{i}" for i in range(100)],
                }
            }

        # Apply ROI filtering using base class helper
        X, ch_names = self._apply_roi_filtering(X, ch_names)

        # Create classifier pipeline using base class helper
        clf = self._create_classifier_pipeline()

        # Set up cross-validation using base class helper
        cv = self._setup_cross_validation(y)

        # Create SlidingEstimator for time-resolved decoding
        time_decoder = SlidingEstimator(
            clf,
            scoring=self.scoring,
            n_jobs=1,  # Single job for consistency
        )

        # Perform cross-validation (returns: n_folds x n_times)
        scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)

        # Mean across folds: (n_times,)
        mean_scores = np.mean(scores, axis=0)

        # Return as row vector: (1, n_times)
        return {
            "timedecoding": {
                "data": mean_scores.reshape(1, -1),
                "col_names": [f"t_{i}" for i in range(len(mean_scores))],
            }
        }
