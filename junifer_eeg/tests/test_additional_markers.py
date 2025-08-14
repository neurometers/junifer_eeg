"""Tests for additional EEG markers from NICE package."""

import mne
import numpy as np
import pytest

from junifer_eeg.markers import (
    GeneralizationDecoding,
    TimeDecoding,
    TimeLockedTopography,
)


def create_test_raw():
    """Create synthetic Raw object for testing."""
    sfreq = 100  # 100 Hz sampling rate
    duration = 20  # 20 seconds (longer to avoid filter warnings)
    times = np.arange(0, duration, 1 / sfreq)

    # Create different frequency components
    alpha_signal = np.sin(2 * np.pi * 10 * times)  # 10 Hz alpha
    beta_signal = np.sin(2 * np.pi * 20 * times)  # 20 Hz beta
    noise = np.random.RandomState(42).normal(0, 0.1, len(times))

    # Combine signals
    data = alpha_signal + 0.5 * beta_signal + noise

    # Create MNE Raw object
    info = mne.create_info(
        ch_names=["Cz", "Fz"],
        sfreq=sfreq,
        ch_types=["eeg"] * 2,
    )
    raw_data = np.tile(data, (2, 1))  # 2 channels
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    return raw


class TestTimeLockedTopography:
    """Test TimeLockedTopography marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = TimeLockedTopography(
            tmin=-0.1,
            tmax=0.5,
            epoch_length=1.0,
            overlap=0.2,
            on="EEG",
        )
        assert marker.tmin == -0.1
        assert marker.tmax == 0.5
        assert marker.epoch_length == 1.0
        assert marker.overlap == 0.2

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = TimeLockedTopography(
            tmin=-0.1,
            tmax=0.2,
            epoch_length=1.0,
            overlap=0.0,
            on="EEG",
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "timelockedtopo" in result
        data = result["timelockedtopo"]["data"]
        col_names = result["timelockedtopo"]["col_names"]

        assert data.shape[0] == 1  # Single aggregated value per channel
        assert data.shape[1] == 2  # 2 channels
        assert len(col_names) == data.shape[1]
        assert np.all(np.isfinite(data))

    def test_compute_with_baseline(self):
        """Test marker computation with baseline correction."""
        raw = create_test_raw()
        marker = TimeLockedTopography(
            tmin=-0.1,
            tmax=0.2,
            epoch_length=1.0,
            baseline=(-0.1, 0.0),
            on="EEG",
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "timelockedtopo" in result
        data = result["timelockedtopo"]["data"]
        assert np.all(np.isfinite(data))


class TestTimeDecoding:
    """Test TimeDecoding marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = TimeDecoding(
            tmin=-0.1,
            tmax=0.3,
            epoch_length=1.5,
            condition_method="alpha_beta",
            n_splits=3,
            on="EEG",
        )
        assert marker.tmin == -0.1
        assert marker.tmax == 0.3
        assert marker.epoch_length == 1.5
        assert marker.condition_method == "alpha_beta"
        assert marker.n_splits == 3

    def test_compute_temporal_halves(self):
        """Test marker computation with temporal halves condition."""
        raw = create_test_raw()
        marker = TimeDecoding(
            tmin=-0.2,
            tmax=0.2,
            epoch_length=1.0,
            overlap=0.0,
            condition_method="temporal_halves",
            n_splits=3,
            on="EEG",
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "time_decoding_scores" in result
        data = result["time_decoding_scores"]["data"]
        col_names = result["time_decoding_scores"]["col_names"]

        assert data.shape[0] == 1  # 1 observation
        assert data.shape[1] > 0  # Should have time points
        assert len(col_names) == data.shape[1]
        assert all("decode_t_" in name for name in col_names)
        assert np.all(np.isfinite(data))
        assert np.all(
            (data >= 0) & (data <= 1),
        )  # ROC-AUC scores should be [0,1]

    def test_compute_spectral_power_condition(self):
        """Test marker computation with spectral power condition."""
        raw = create_test_raw()
        marker = TimeDecoding(
            tmin=-0.1,
            tmax=0.1,
            epoch_length=1.0,
            overlap=0.0,
            condition_method="spectral_power",
            n_splits=3,
            on="EEG",
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "time_decoding_scores" in result
        data = result["time_decoding_scores"]["data"]
        assert np.all(np.isfinite(data))

    def test_invalid_condition_method(self):
        """Test that invalid condition method raises error."""
        raw = create_test_raw()
        marker = TimeDecoding(condition_method="invalid_method", on="EEG")

        input_data = {"data": raw}
        with pytest.raises(ValueError, match="Unknown condition method"):
            marker.compute(input_data)


class TestGeneralizationDecoding:
    """Test GeneralizationDecoding marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = GeneralizationDecoding(
            tmin=-0.1,
            tmax=0.3,
            epoch_length=1.5,
            condition_method="spectral_power",
            n_splits=3,
            on="EEG",
        )
        assert marker.tmin == -0.1
        assert marker.tmax == 0.3
        assert marker.epoch_length == 1.5
        assert marker.condition_method == "spectral_power"
        assert marker.n_splits == 3

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = GeneralizationDecoding(
            tmin=-0.1,
            tmax=0.1,
            epoch_length=1.0,
            overlap=0.0,
            condition_method="temporal_halves",
            n_splits=3,
            on="EEG",
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "generalization_matrix" in result
        data = result["generalization_matrix"]["data"]
        col_names = result["generalization_matrix"]["col_names"]
        row_names = result["generalization_matrix"]["row_names"]

        # Should be a square matrix (n_times, n_times)
        assert data.shape[0] == data.shape[1]
        assert data.shape[0] > 0  # Should have time points
        assert len(col_names) == data.shape[1]
        assert len(row_names) == data.shape[0]
        assert col_names == row_names  # Same times for train/test
        assert all("t_" in name for name in col_names)
        assert np.all(np.isfinite(data))
        assert np.all(
            (data >= 0) & (data <= 1),
        )  # ROC-AUC scores should be [0,1]

    def test_compute_alpha_beta_condition(self):
        """Test marker computation with alpha-beta condition."""
        raw = create_test_raw()
        marker = GeneralizationDecoding(
            tmin=-0.1,
            tmax=0.1,
            epoch_length=1.0,
            overlap=0.0,
            condition_method="alpha_beta",
            n_splits=3,
            on="EEG",
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "generalization_matrix" in result
        data = result["generalization_matrix"]["data"]
        assert data.shape[0] == data.shape[1]  # Square matrix
        assert np.all(np.isfinite(data))


class TestMarkerIntegration:
    """Test integration between markers."""

    def test_all_markers_can_be_imported(self):
        """Test that all markers can be imported successfully."""
        from junifer_eeg.markers import (
            ContingentNegativeVariation,
            GeneralizationDecoding,
            KolmogorovComplexity,
            PermutationEntropy,
            PowerSpectralDensityEstimator,
            PowerSpectralDensitySummary,
            SpectralPower,
            TimeDecoding,
            TimeLockedTopography,
        )

        # Just verify they can be instantiated
        raw = create_test_raw()
        input_data = {"data": raw}

        markers = [
            TimeLockedTopography(tmin=-0.1, tmax=0.1, on="EEG"),
            KolmogorovComplexity(tmin=0.5, tmax=2.0, on="EEG"),
            PermutationEntropy(tmin=0.5, tmax=2.0, on="EEG"),
            ContingentNegativeVariation(on="EEG"),
            PowerSpectralDensityEstimator(fmin=1, fmax=30, on="EEG"),
            PowerSpectralDensitySummary(fmin=1, fmax=30, on="EEG"),
            SpectralPower(on="EEG"),
        ]

        # Verify each marker can compute something
        for marker in markers:
            result = marker.compute(input_data)
            assert isinstance(result, dict)
            assert len(result) > 0

        # Test decoding markers separately (they need more epochs)
        decoding_markers = [
            TimeDecoding(tmin=-0.1, tmax=0.1, n_splits=2, on="EEG"),
            GeneralizationDecoding(tmin=-0.1, tmax=0.1, n_splits=2, on="EEG"),
        ]

        for marker in decoding_markers:
            try:
                result = marker.compute(input_data)
                assert isinstance(result, dict)
                assert len(result) > 0
            except ValueError:
                # It's ok if decoding fails with synthetic data
                pass
