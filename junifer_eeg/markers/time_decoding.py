"""Time decoding marker for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


@register_marker
class TimeDecoding(BaseMarker):
    """Time decoding marker using MNE-Python's SlidingEstimator.

    This marker performs temporal decoding analysis by creating epochs from
    continuous data and using temporal segments as conditions for classification.

    Note: Adapted from NICE for continuous data. Uses temporal segments or
    frequency content as conditions for decoding.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scikit-learn"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"time_decoding_scores": "vector"}
    }

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 0.8,
        epoch_length: float = 2.0,
        overlap: float = 0.0,
        condition_method: str = "temporal_halves",
        n_splits: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the TimeDecoding marker.

        Parameters
        ----------
        tmin : float, default=-0.2
            Start time for decoding window in seconds.
        tmax : float, default=0.8
            End time for decoding window in seconds.
        epoch_length : float, default=2.0
            Length of epochs to create from continuous data.
        overlap : float, default=0.0
            Overlap between epochs (0.0 = no overlap, 0.9 = 90% overlap).
        condition_method : str, default="temporal_halves"
            Method to create conditions: "temporal_halves", "alpha_beta", "spectral_power".
        n_splits : int, default=5
            Number of cross-validation splits.
        scoring : str, default="roc_auc"
            Scoring metric for classification.
        random_state : int, default=42
            Random state for reproducibility.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.condition_method = condition_method
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute time decoding.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed time decoding scores.
        """
        from mne.decoding import SlidingEstimator, cross_val_multiscore
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        # Get the MNE Raw object
        raw = input["data"]

        # Create epochs from continuous data
        epochs = self._create_epochs(raw)

        # Create conditions based on chosen method
        X, y = self._create_conditions(epochs)

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Not enough conditions created for classification"
            )

        # Set up classifier pipeline
        scaler = StandardScaler()
        feature_select = SelectPercentile(f_classif, percentile=10)
        svc = SVC(
            C=1,
            kernel="linear",
            probability=True,
            random_state=self.random_state,
        )
        clf = Pipeline(
            [
                ("scaler", scaler),
                ("feature_select", feature_select),
                ("svc", svc),
            ]
        )

        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=min(self.n_splits, len(y) // 2),
            shuffle=True,
            random_state=self.random_state,
        )

        # Create SlidingEstimator
        time_decoder = SlidingEstimator(clf, scoring=self.scoring, n_jobs=1)

        # Perform cross-validation
        try:
            scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)
            # Average across CV folds
            mean_scores = np.mean(scores, axis=0)
        except Exception:
            # If decoding fails, return zeros
            n_times = X.shape[2]
            mean_scores = np.zeros(n_times)

        # Create time labels
        times = epochs.times
        time_labels = [f"decode_t_{t:.3f}s" for t in times]

        # Return data in junifer format
        return {
            "time_decoding_scores": {
                "data": mean_scores.reshape(1, -1),  # Shape: (1, n_times)
                "col_names": time_labels,
            }
        }

    def _create_epochs(self, raw):
        """Create epochs from continuous data."""
        import mne

        # Create epochs from continuous data
        duration = self.epoch_length
        overlap_samples = int(self.overlap * duration * raw.info["sfreq"])

        # Create events at regular intervals
        sfreq = raw.info["sfreq"]
        duration_samples = int(duration * sfreq)
        step_samples = duration_samples - overlap_samples

        # Calculate the number of epochs we can create
        n_samples = raw.n_times
        n_epochs = max(1, (n_samples - duration_samples) // step_samples + 1)

        # Create event array
        events = np.zeros((n_epochs, 3), dtype=int)
        for i in range(n_epochs):
            events[i, 0] = i * step_samples + duration_samples // 2
            events[i, 2] = 1  # Event ID

        # Make sure events don't exceed data length
        valid_events = events[events[:, 0] < n_samples - duration_samples // 2]

        if len(valid_events) == 0:
            raise ValueError("Data too short to create any epochs")

        # Create epochs
        epochs = mne.Epochs(
            raw,
            valid_events,
            event_id={"epoch": 1},
            tmin=-duration / 2,
            tmax=duration / 2,
            baseline=None,
            preload=True,
            verbose=False,
        )

        # Crop to analysis window if specified
        if self.tmin is not None or self.tmax is not None:
            epochs = epochs.copy().crop(tmin=self.tmin, tmax=self.tmax)

        return epochs

    def _create_conditions(self, epochs):
        """Create conditions for classification based on the chosen method."""
        X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        n_epochs = X.shape[0]

        if self.condition_method == "temporal_halves":
            # Split epochs into first and second half based on their temporal order
            y = np.zeros(n_epochs, dtype=int)
            y[n_epochs // 2 :] = 1

        elif self.condition_method == "alpha_beta":
            # Create conditions based on alpha vs beta power
            from scipy.signal import welch

            sfreq = epochs.info["sfreq"]
            alpha_power = []
            beta_power = []

            for epoch_data in X:
                # Compute power for each epoch
                epoch_alpha = 0
                epoch_beta = 0

                for ch_data in epoch_data:
                    # Use shorter nperseg for short data
                    nperseg = min(64, len(ch_data) // 2, len(ch_data))
                    if nperseg < 4:
                        nperseg = len(ch_data)

                    freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
                    alpha_mask = (freqs >= 8) & (freqs <= 13)
                    beta_mask = (freqs >= 13) & (freqs <= 30)

                    if np.any(alpha_mask):
                        epoch_alpha += np.mean(psd[alpha_mask])
                    if np.any(beta_mask):
                        epoch_beta += np.mean(psd[beta_mask])

                alpha_power.append(epoch_alpha)
                beta_power.append(epoch_beta)

            # Create binary labels based on alpha/beta ratio
            alpha_power = np.array(alpha_power)
            beta_power = np.array(beta_power)

            # Handle edge cases
            if np.all(alpha_power == 0) and np.all(beta_power == 0):
                # Fallback to temporal halves if no frequency content
                y = np.zeros(n_epochs, dtype=int)
                y[n_epochs // 2 :] = 1
            else:
                alpha_beta_ratio = alpha_power / (beta_power + 1e-10)
                # Use median split, but ensure we have at least one of each class
                median_ratio = np.median(alpha_beta_ratio)
                y = (alpha_beta_ratio > median_ratio).astype(int)

                # Ensure we have both classes
                if len(np.unique(y)) < 2:
                    # Fallback to temporal halves
                    y = np.zeros(n_epochs, dtype=int)
                    y[n_epochs // 2 :] = 1

        elif self.condition_method == "spectral_power":
            # Create conditions based on overall spectral power
            power_per_epoch = np.mean(
                np.var(X, axis=2), axis=1
            )  # Power per epoch
            y = (power_per_epoch > np.median(power_per_epoch)).astype(int)

        else:
            raise ValueError(
                f"Unknown condition method: {self.condition_method}"
            )

        # Final check to ensure we have both classes
        if len(np.unique(y)) < 2:
            # Ultimate fallback: alternate labels
            y = np.arange(n_epochs) % 2

        return X, y
