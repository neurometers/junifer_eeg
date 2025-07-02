"""Window decoding marker for EEG analysis."""

from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker

from .utils import apply_roi_trial_aggregation, get_data_for_rois


@register_marker
class WindowDecoding(BaseMarker):
    """Window decoding marker for condition classification.

    This marker performs decoding analysis within specified time windows
    to classify between experimental conditions, following the ICM Local
    Global paradigm patterns. Essential for analyzing discriminative
    neural patterns and condition-specific activity.

    Based on the NICE WindowDecoding implementation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scikit-learn"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "windowdecoding": "vector",
        },
    }

    def __init__(
        self,
        condition_a: str | List[str],
        condition_b: str | List[str],
        tmin: float,
        tmax: float,
        n_splits: int = 5,
        scoring: str = "roc_auc",
        random_state: Optional[int] = 42,
        comment: Optional[str] = None,
        rois: Optional[List[str]] = None,
        roi_aggregation_method: Optional[str | List[str]] = None,
        trial_aggregation_method: Optional[str | List[str]] = None,
        equipment: str = "standard",
        on: Optional[str | List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize WindowDecoding marker."""
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
            or f"{'-'.join(self.condition_a)}_vs_{'-'.join(self.condition_b)}_decoding"
        )
        self.rois = rois
        self.roi_aggregation_method = roi_aggregation_method
        self.trial_aggregation_method = trial_aggregation_method
        self.equipment = equipment

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute window decoding between conditions."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        epochs = input["data"]

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
            # Return chance level if conditions not found - use aggregation framework
            chance_data = np.array([[0.5]])

            if self.rois is not None:
                # Create dummy data for ROI processing
                dummy_data = np.full((len(epochs.ch_names), 1), 0.5)
                roi_data = get_data_for_rois(
                    dummy_data,
                    list(epochs.ch_names),
                    self.rois,
                    equipment=self.equipment,
                )
            else:
                roi_data = {"all_channels": chance_data.T}

            results = apply_roi_trial_aggregation(
                roi_data,
                roi_aggregation_methods=self.roi_aggregation_method,
                trial_aggregation_methods=self.trial_aggregation_method,
                marker_name="windowdecoding",
            )
            return results

        # Crop to time window
        epochs_a = epochs_a.crop(tmin=self.tmin, tmax=self.tmax)
        epochs_b = epochs_b.crop(tmin=self.tmin, tmax=self.tmax)

        # Reset baseline to None since we cropped the epochs
        epochs_a.baseline = None
        epochs_b.baseline = None

        # Combine epochs and create labels (following NICE approach)
        import mne

        combined_epochs = mne.concatenate_epochs([epochs_a, epochs_b])
        labels = np.concatenate(
            [
                np.zeros(len(epochs_a)),  # Label 0 for condition A
                np.ones(len(epochs_b)),  # Label 1 for condition B
            ]
        )

        # Get data: (n_epochs, n_channels, n_times)
        X = combined_epochs.get_data()

        # Flatten across time for window decoding: (n_epochs, n_channels * n_times)
        X_flat = X.reshape(X.shape[0], -1)

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
                [("scaler", scaler), ("anova", transform), ("svc", svc)]
            )
        else:
            # Use LDA for accuracy
            clf = LinearDiscriminantAnalysis()
            scaler = StandardScaler()
            X_flat = scaler.fit_transform(X_flat)

        # Cross-validation following NICE approach
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Perform cross-validation
        scores = cross_val_score(
            clf, X_flat, labels, cv=cv, scoring=self.scoring
        )

        # Mean score across folds
        mean_score = np.mean(scores)

        # Package results using aggregation framework
        score_data = np.array([[mean_score]])

        if self.rois is not None:
            # Create dummy data for ROI processing (single value repeated)
            dummy_data = np.full((len(epochs.ch_names), 1), mean_score)
            roi_data = get_data_for_rois(
                dummy_data,
                list(epochs.ch_names),
                self.rois,
                equipment=self.equipment,
            )
        else:
            roi_data = {"all_channels": score_data.T}

        results = apply_roi_trial_aggregation(
            roi_data,
            roi_aggregation_methods=self.roi_aggregation_method,
            trial_aggregation_methods=self.trial_aggregation_method,
            marker_name="windowdecoding",
        )

        return results
