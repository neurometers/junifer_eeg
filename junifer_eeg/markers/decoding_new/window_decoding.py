"""Window decoding marker implementation."""

from typing import Any, ClassVar, Dict, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker

from ._decoding_base import DecodingBase


@register_marker
class WindowDecoding(DecodingBase):
    """Window decoding marker for condition classification.

    This marker performs decoding analysis within specified time windows
    to classify between experimental conditions. Essential for analyzing
    discriminative neural patterns and condition-specific activity.

    The marker applies optional ROI filtering BEFORE decoding to restrict
    classification to specific channels (e.g., frontal, parietal regions).
    """

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "EEG": {
            "windowdecoding": "vector",
        },
    }

    def __init__(
        self,
        condition_a: str | list[str],
        condition_b: str | list[str],
        tmin: float,
        tmax: float,
        n_splits: int = 10,
        scoring: str = "roc_auc",
        random_state: Optional[int] = None,
        comment: Optional[str] = None,
        rois: Union[list[str], list[int], None] = None,
        equipment: str = "egi256",
        on: Optional[str | list[str]] = None,
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
        n_splits : int, default=10
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
        """Compute window decoding between conditions.

        Process:
        1. Filter epochs by condition
        2. Apply ROI filtering (if specified) to select channels
        3. Crop to time window (if specified)
        4. Use manual cross-validation with sample weighting
        5. Return single scalar score averaged across folds
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        # Prepare epochs data using base class helper
        X, y, ch_names, valid_data = self._prepare_epochs_data(input)

        # Check if conditions were found
        if not valid_data:
            # Return chance level (0.5) if conditions not found
            return {
                "windowdecoding": {
                    "data": np.array([0.5]),
                    "col_names": ["score"],
                }
            }

        # Apply ROI filtering using base class helper
        X, ch_names = self._apply_roi_filtering(X, ch_names)

        # Encode labels using NICE's approach
        y = LabelEncoder().fit_transform(y)

        # Flatten spatial-temporal features for window decoding (NICE approach)
        X_flat = X.reshape(len(X), np.prod(X.shape[1:]))

        # Create classifier pipeline using base class helper
        clf = self._create_classifier_pipeline()

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
                "data": np.array([mean_score]),
                "col_names": ["score"],
            }
        }
