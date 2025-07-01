"""Tests for WindowDecoding marker."""

import mne
import numpy as np
import pytest

from junifer_eeg.markers import WindowDecoding


class TestWindowDecoding:
    """Test WindowDecoding marker."""

    def test_window_decoding_init(self):
        """Test WindowDecoding initialization."""
        marker = WindowDecoding(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.0,
            tmax=1.0,
            comment="test_decoding",
        )
        assert marker.condition_a == ["LSGS"]
        assert marker.condition_b == ["LDGD"]
        assert marker.tmin == 0.0
        assert marker.tmax == 1.0
        assert marker.comment == "test_decoding"

    def test_window_decoding_multiple_conditions(self):
        """Test WindowDecoding with multiple conditions."""
        marker = WindowDecoding(
            condition_a=["LSGS", "LSGD"],
            condition_b=["LDGS", "LDGD"],
            tmin=0.6,
            tmax=0.967,
            comment="local",
        )
        assert marker.condition_a == ["LSGS", "LSGD"]
        assert marker.condition_b == ["LDGS", "LDGD"]

    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs for testing."""
        # Create synthetic EEG data with discriminable patterns
        n_channels = 8
        n_times = 100
        sfreq = 250

        # Create info
        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create events for different conditions
        n_epochs_per_condition = 15  # More epochs for better decoding
        events = []
        event_id = {"LSGS": 1, "LDGD": 2}

        # Create discriminable data patterns
        np.random.seed(42)  # For reproducible tests
        data_parts = []

        for i, (condition, event_code) in enumerate(event_id.items()):
            for _j in range(n_epochs_per_condition):
                event_time = len(data_parts) * n_times
                events.append([event_time, 0, event_code])

                # Create slightly different patterns for each condition
                if condition == "LSGS":
                    epoch_data = (
                        np.random.randn(n_channels, n_times) * 1e-6
                        + i * 0.5e-6
                    )
                else:
                    epoch_data = (
                        np.random.randn(n_channels, n_times) * 1e-6
                        - i * 0.5e-6
                    )

                data_parts.append(epoch_data)

        events = np.array(events)

        # Concatenate all data
        data = np.concatenate(data_parts, axis=1)
        raw = mne.io.RawArray(data, info, verbose=False)

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=-0.2,
            tmax=0.8,
            baseline=None,
            preload=True,
            verbose=False,
        )

        return epochs

    def test_window_decoding_compute(self, sample_epochs):
        """Test WindowDecoding computation."""
        marker = WindowDecoding(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.0,
            tmax=0.5,
            comment="test",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "windowdecoding" in result
        assert "data" in result["windowdecoding"]
        assert "col_names" in result["windowdecoding"]

        # Check dimensions
        data = result["windowdecoding"]["data"]
        col_names = result["windowdecoding"]["col_names"]

        assert data.shape == (1, 1)  # One decoding score
        assert len(col_names) == 1

        # Score should be between 0 and 1
        score = data[0, 0]
        assert 0.0 <= score <= 1.0

    def test_window_decoding_missing_conditions(self, sample_epochs):
        """Test WindowDecoding with missing conditions."""
        marker = WindowDecoding(
            condition_a="missing_condition",
            condition_b="LDGD",
            tmin=0.0,
            tmax=0.5,
            comment="missing",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        # Should return chance level (0.5) when conditions are missing
        assert "windowdecoding" in result
        data = result["windowdecoding"]["data"]
        assert data[0, 0] == 0.5

    def test_window_decoding_parameters(self, sample_epochs):
        """Test WindowDecoding with different parameters."""
        marker = WindowDecoding(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.1,
            tmax=0.6,
            n_splits=3,
            scoring="roc_auc",
            random_state=123,
            comment="params",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "windowdecoding" in result
        data = result["windowdecoding"]["data"]
        assert data.shape == (1, 1)

        # For ROC AUC, score should be between 0 and 1
        score = data[0, 0]
        assert 0.0 <= score <= 1.0

    def test_window_decoding_cross_validation(self, sample_epochs):
        """Test WindowDecoding cross-validation stability."""
        # Run multiple times with same random state - should be identical
        marker1 = WindowDecoding(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.0,
            tmax=0.5,
            random_state=42,
            comment="cv1",
        )

        marker2 = WindowDecoding(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.0,
            tmax=0.5,
            random_state=42,
            comment="cv2",
        )

        input_data = {"data": sample_epochs}
        result1 = marker1.compute(input_data)
        result2 = marker2.compute(input_data)

        # Should be identical with same random state
        assert np.allclose(
            result1["windowdecoding"]["data"],
            result2["windowdecoding"]["data"],
        )
