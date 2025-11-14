"""Window decoding marker for EEG analysis."""

from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker


@register_marker
class WindowDecoding(BaseMarker):
    """Window decoding marker for condition classification.

    This marker performs decoding analysis within specified time windows
    to classify between experimental conditions. Essential for analyzing
    discriminative neural patterns and condition-specific activity.

    The marker applies optional ROI filtering BEFORE decoding to restrict
    classification to specific channels (e.g., frontal, parietal regions).
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
        n_splits: int = 10,
        scoring: str = "roc_auc",
        random_state: Optional[int] = None,
        comment: Optional[str] = None,
        rois: Union[List[str], List[int], None] = None,
        equipment: str = "egi256",
        on: Optional[str | List[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize WindowDecoding marker.

        Parameters
        ----------
        condition_a : str or list of str
            Condition(s) for class A.
        condition_b : str or list of str
            Condition(s) for class B.
        tmin : float
            Start time for decoding window.
        tmax : float
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
            or f"{'-'.join(self.condition_a)}_vs_{'-'.join(self.condition_b)}_decoding"
        )
        self.rois = rois
        self.equipment = equipment

        super().__init__(on=on, name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute window decoding between conditions.

        Process:
        1. Filter epochs by condition
        2. Apply ROI filtering (if specified) to select channels
        3. Crop to time window
        4. Flatten: (n_epochs, n_channels * n_times)
        5. Cross-validated classification using NICE's decode_window algorithm
        6. Return mean score across folds
        """
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
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
            # Return chance level (0.5) if conditions not found
            return {
                "windowdecoding": {
                    "data": np.array([[0.5]]),
                    "col_names": ["score"],
                }
            }

        # Crop to time window
        epochs_a = epochs_a.copy().crop(tmin=self.tmin, tmax=self.tmax)
        epochs_b = epochs_b.copy().crop(tmin=self.tmin, tmax=self.tmax)

        # Reset baseline to None since we cropped the epochs
        epochs_a.baseline = None
        epochs_b.baseline = None

        # Combine epochs
        import mne

        combined_epochs = mne.concatenate_epochs([epochs_a, epochs_b])

        # Get data: (n_epochs, n_channels, n_times)
        X = combined_epochs.get_data()
        ch_names = list(combined_epochs.ch_names)
        n_epochs, n_channels, n_times = X.shape

        # Apply ROI filtering BEFORE flattening (if specified)
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

        # Create labels using NICE's approach
        y = np.concatenate(
            [
                np.zeros(len(epochs_a)),  # Label 0 for condition A
                np.ones(len(epochs_b)),  # Label 1 for condition B
            ],
        )

        # Encode labels using NICE's approach
        y = LabelEncoder().fit_transform(y)

        # Flatten spatial-temporal features for window decoding (NICE approach)
        X_flat = X.reshape(len(X), np.prod(X.shape[1:]))

        # Create classifier pipeline
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

        # Set up cross-validation
        if self.n_splits is None or isinstance(self.n_splits, int):
            n_splits = self.n_splits if isinstance(self.n_splits, int) else 10
            cv = StratifiedKFold(
                n_splits=int(min(n_splits, len(y) / 2)),
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv = self.n_splits

        # Compute sample weights using NICE's approach
        sample_weight = np.zeros(len(y), dtype=float)
        for this_y in np.unique(y):
            this_mask = y == this_y
            sample_weight[this_mask] = 1.0 / np.sum(this_mask)

        # Perform cross-validation using NICE's decode_window logic
        scores = []
        for train_idx, test_idx in cv.split(X_flat, y):
            # Clone classifier for each fold
            from sklearn.base import clone

            clf_fold = clone(clf)

            try:
                # Fit with sample weights (NICE approach)
                clf_fold.fit(
                    X_flat[train_idx],
                    y[train_idx],
                    svc__sample_weight=sample_weight[train_idx],
                )

                # Predict and score
                prediction = clf_fold.predict(X_flat[test_idx])
                score = roc_auc_score(
                    y_true=y[test_idx],
                    y_score=prediction,
                    sample_weight=sample_weight[test_idx],
                    average="weighted",
                )
            except ValueError as e:
                # Handle case where no features are selected (perfect separation)
                if "0 feature(s)" in str(e):
                    # If no features selected, use simple decision based on mean
                    train_mean_A = np.mean(
                        X_flat[train_idx][y[train_idx] == 0]
                    )
                    train_mean_B = np.mean(
                        X_flat[train_idx][y[train_idx] == 1]
                    )

                    test_mean_A = np.mean(X_flat[test_idx][y[test_idx] == 0])
                    test_mean_B = np.mean(X_flat[test_idx][y[test_idx] == 1])

                    # Perfect classification if means are separable
                    if (
                        train_mean_A < train_mean_B
                        and test_mean_A < test_mean_B
                    ) or (
                        train_mean_A > train_mean_B
                        and test_mean_A > test_mean_B
                    ):
                        prediction = y[test_idx]  # Perfect prediction
                        score = 1.0
                    else:
                        prediction = np.zeros_like(y[test_idx])  # Chance level
                        score = 0.5
                else:
                    raise e

            scores.append(score)

        # Mean score across folds
        mean_score = np.mean(scores)

        # Return single scalar result
        return {
            "windowdecoding": {
                "data": np.array([[mean_score]]),
                "col_names": ["score"],
            }
        }
