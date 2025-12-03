"""Base class for Decoding markers.

This refactoring follows the conservative helper-based approach recommended
for decoding markers with different cross-validation strategies.

Structure:
- Base class: Shared data preparation helpers (condition handling, ROI filtering, time windowing)
- TimeDecoding: Inherits base, uses SlidingEstimator for temporal decoding
- WindowDecoding: Inherits base, uses manual CV for spatial-temporal decoding
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from junifer.markers.base import BaseMarker

from ..utils import filter_to_eeg_channels, get_data_for_rois


class DecodingBase(BaseMarker):
    """Base class for Decoding markers with common data preparation functionality.

    Provides shared functionality for data preparation including condition filtering,
    ROI filtering, time windowing, and classifier pipeline setup. Uses conservative
    helper-based approach to support different cross-validation strategies.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scikit-learn"}

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
        """Initialize the Decoding base marker.

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
        equipment : str, default='egi256'
            Equipment type for electrode mapping.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
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
        self.comment = comment
        self.rois = rois
        self.equipment = equipment
        super().__init__(on=on, name=name)

    def _prepare_epochs_data(
        self, input: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], bool]:
        """Prepare and filter epochs data for decoding.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.

        Returns
        -------
        tuple
            (X, y, ch_names, valid_data) where:
            - X: Data array (n_epochs, n_channels, n_times)
            - y: Labels array (n_epochs,)
            - ch_names: Channel names list
            - valid_data: bool indicating if conditions were found

        Notes
        -----
        Returns chance level data (0.5) if conditions not found to match
        original marker behavior.
        """
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
            # Return empty data to indicate invalid conditions
            return np.array([]), np.array([]), [], False

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

        # Create labels
        y = np.concatenate(
            [
                np.zeros(len(epochs_a)),  # Label 0 for condition A
                np.ones(len(epochs_b)),  # Label 1 for condition B
            ],
        )

        return X, y, ch_names, True

    def _apply_roi_filtering(
        self, X: np.ndarray, ch_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply ROI filtering to data before decoding.

        Parameters
        ----------
        X : np.ndarray
            Data array (n_epochs, n_channels, n_times)
        ch_names : List[str]
            Channel names

        Returns
        -------
        tuple
            (X_filtered, ch_names_filtered) where:
            - X_filtered: ROI-filtered data array
            - ch_names_filtered: Filtered channel names
        """
        if self.rois is None:
            return X, ch_names

        # Transpose to (n_channels, n_epochs, n_times) for ROI filtering
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
            X_filtered = X_filtered.transpose(1, 0, 2)

            # Update channel names to match ROI names
            ch_names_filtered = (
                self.rois if isinstance(self.rois, list) else [self.rois]
            )
        else:
            X_filtered = X
            ch_names_filtered = ch_names

        return X_filtered, ch_names_filtered

    def _create_classifier_pipeline(self) -> Any:
        """Create classifier pipeline following NICE approach.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Configured classifier pipeline
        """
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

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

        return clf

    def _setup_cross_validation(self, y: np.ndarray) -> Any:
        """Set up cross-validation strategy.

        Parameters
        ----------
        y : np.ndarray
            Labels array

        Returns
        -------
        sklearn.model_selection.StratifiedKFold
            Configured cross-validation object
        """
        from sklearn.model_selection import StratifiedKFold

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        return cv

    def _create_chance_level_output(
        self, n_times: int = 100
    ) -> Dict[str, Any]:
        """Create chance level output for missing conditions.

        Parameters
        ----------
        n_times : int, default=100
            Number of time points for output

        Returns
        -------
        dict
            Generic chance level output data (marker-specific format handled in subclasses)

        Notes
        -----
        This method provides a generic chance level output template.
        Subclasses should override this to return marker-specific output format
        (e.g., {"timedecoding": {...}} vs {"windowdecoding": {...}}).
        """
        return {
            "data": np.full((1, n_times), 0.5),
            "col_names": [f"t_{i}" for i in range(n_times)],
        }
