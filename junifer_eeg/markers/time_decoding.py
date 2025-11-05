"""Time decoding marker for EEG analysis."""

from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker


@register_marker
class TimeDecoding(BaseMarker):
    """Time decoding marker for temporal classification analysis.

    This marker performs decoding analysis across time using MNE's Sliding
    Estimator to classify between experimental conditions at each time point.
    Essential for analyzing the temporal dynamics of discriminative neural
    patterns.

    The marker applies optional ROI filtering BEFORE decoding to restrict
    classification to specific channels.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scikit-learn"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "timedecoding": "vector",
        },
    }

    def __init__(
        self,
        condition_a: str | List[str],
        condition_b: str | List[str],
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        n_splits: int = 5,
        scoring: str = "roc_auc",
        random_state: Optional[int] = 42,
        comment: Optional[str] = None,
        rois: Union[List[str], List[int], None] = None,
        equipment: str = "egi256",
        on: Optional[str | List[str]] = None,
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
        equipment : str, default='standard'
            Equipment type for electrode mapping.
        """
        self.condition_a = (
            condition_a if isinstance(condition_a, list) else [condition_a]
        )
        self.condition_b = (
            condition_b if isinstance(condition_b, list) else [condition_b]
        )
        self.tmin = tmin
        self.tmax = tmax
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        self.comment = (
            comment
            or f"{'-'.join(self.condition_a)}_vs_{'-'.join(self.condition_b)}_time_decoding"
        )
        self.rois = rois
        self.equipment = equipment

        super().__init__(on=on, name=name)

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
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        from .utils import filter_to_eeg_channels, get_data_for_rois

        epochs = input["data"]

        # Filter to EEG channels
        epochs, eeg_ch_names, eeg_indices = filter_to_eeg_channels(epochs)

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

        # Check for missing conditions
        if (
            epochs_a is None
            or epochs_b is None
            or len(epochs_a) == 0
            or len(epochs_b) == 0
        ):
            # Return chance level (0.5) time series if conditions not found
            # Use a reasonable default length (e.g., 100 time points)
            n_times = 100
            return {
                "timedecoding": {
                    "data": np.full((1, n_times), 0.5),
                    "col_names": [f"t_{i}" for i in range(n_times)],
                }
            }

        # Crop to time window if specified
        if self.tmin is not None or self.tmax is not None:
            epochs_a = epochs_a.copy().crop(tmin=self.tmin, tmax=self.tmax)
            epochs_b = epochs_b.copy().crop(tmin=self.tmin, tmax=self.tmax)

        # Combine epochs
        import mne

        combined_epochs = mne.concatenate_epochs([epochs_a, epochs_b])

        # Get data: (n_epochs, n_channels, n_times)
        X = combined_epochs.get_data()
        ch_names = list(combined_epochs.ch_names)
        n_epochs, n_channels, n_times = X.shape

        # Apply ROI filtering BEFORE decoding (if specified)
        if self.rois is not None:
            # Transpose to (n_channels, n_epochs, n_times)
            X_transposed = X.transpose(1, 0, 2)

            # Get ROI-filtered data
            roi_data_dict = get_data_for_rois(
                X_transposed,
                ch_names,
                self.rois,
                self.equipment,
            )

            # Extract the filtered data (returns {"selected_channels": data})
            if "selected_channels" in roi_data_dict:
                X_filtered = roi_data_dict["selected_channels"]
                # Transpose back to (n_epochs, n_channels, n_times)
                X = X_filtered.transpose(1, 0, 2)
                n_epochs, n_channels, n_times = X.shape

        # Create labels
        y = np.concatenate(
            [
                np.zeros(len(epochs_a)),  # Label 0 for condition A
                np.ones(len(epochs_b)),  # Label 1 for condition B
            ],
        )

        # Create classifier pipeline following NICE approach
        if self.scoring == "roc_auc":
            # Use SVM with probability for ROC AUC
            scaler = StandardScaler()
            transform = SelectPercentile(f_classif, percentile=10)
            svc = SVC(
                C=1,
                kernel="linear",
                probability=True,
                random_state=self.random_state,
            )
            clf = Pipeline(
                [("scaler", scaler), ("anova", transform), ("svc", svc)],
            )
        else:
            # Use LDA for accuracy
            from sklearn.discriminant_analysis import (
                LinearDiscriminantAnalysis,
            )

            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lda", LinearDiscriminantAnalysis()),
                ]
            )

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

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
