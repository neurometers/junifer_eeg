"""Tests for TimeLockedContrast marker."""

import mne
import numpy as np
import pytest

from junifer_eeg.markers import TimeLockedContrast


class TestTimeLockedContrast:
    """Test TimeLockedContrast marker."""

    def test_time_locked_contrast_init(self):
        """Test TimeLockedContrast initialization."""
        marker = TimeLockedContrast(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.0,
            tmax=1.0,
            comment="test_contrast",
        )
        assert marker.condition_a == ["LSGS"]
        assert marker.condition_b == ["LDGD"]
        assert marker.tmin == 0.0
        assert marker.tmax == 1.0
        assert marker.comment == "test_contrast"

    def test_time_locked_contrast_multiple_conditions(self):
        """Test TimeLockedContrast with multiple conditions."""
        marker = TimeLockedContrast(
            condition_a=["LSGS", "LSGD"],
            condition_b=["LDGS", "LDGD"],
            comment="multi_condition",
        )
        assert marker.condition_a == ["LSGS", "LSGD"]
        assert marker.condition_b == ["LDGS", "LDGD"]

    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs for testing."""
        # Create synthetic EEG data
        n_channels = 8
        n_times = 100
        sfreq = 250

        # Create info
        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create events for different conditions
        n_epochs_per_condition = 10
        events = []
        event_id = {"LSGS": 1, "LDGD": 2, "LSGD": 3, "LDGS": 4}

        for i, (_condition, event_code) in enumerate(event_id.items()):
            for j in range(n_epochs_per_condition):
                event_time = i * n_epochs_per_condition * n_times + j * n_times
                events.append([event_time, 0, event_code])

        events = np.array(events)

        # Create raw data
        total_samples = len(events) * n_times
        data = np.random.randn(n_channels, total_samples) * 1e-6
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

    def test_time_locked_contrast_compute(self, sample_epochs):
        """Test TimeLockedContrast computation."""
        marker = TimeLockedContrast(
            condition_a="LSGS",
            condition_b="LDGD",
            comment="test",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "timelockedcontrast" in result
        assert "data" in result["timelockedcontrast"]
        assert "col_names" in result["timelockedcontrast"]

        # Check dimensions
        data = result["timelockedcontrast"]["data"]
        col_names = result["timelockedcontrast"]["col_names"]

        assert data.shape[0] == 1  # One contrast
        assert data.shape[1] == len(sample_epochs.ch_names)  # One per channel
        assert len(col_names) == len(sample_epochs.ch_names)

    def test_time_locked_contrast_time_window(self, sample_epochs):
        """Test TimeLockedContrast with time window."""
        marker = TimeLockedContrast(
            condition_a="LSGS",
            condition_b="LDGD",
            tmin=0.1,
            tmax=0.5,
            comment="windowed",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "timelockedcontrast" in result
        data = result["timelockedcontrast"]["data"]
        assert data.shape[0] == 1
        assert data.shape[1] == len(sample_epochs.ch_names)

    def test_time_locked_contrast_missing_conditions(self, sample_epochs):
        """Test TimeLockedContrast with missing conditions."""
        marker = TimeLockedContrast(
            condition_a="missing_condition",
            condition_b="LDGD",
            comment="missing",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        # Should return zeros when conditions are missing
        assert "timelockedcontrast" in result
        data = result["timelockedcontrast"]["data"]
        assert np.allclose(data, 0.0)

    def test_time_locked_contrast_baseline(self, sample_epochs):
        """Test TimeLockedContrast with baseline correction."""
        marker = TimeLockedContrast(
            condition_a="LSGS",
            condition_b="LDGD",
            baseline=(-0.2, 0.0),
            comment="baseline",
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "timelockedcontrast" in result
        data = result["timelockedcontrast"]["data"]
        assert data.shape[0] == 1
        assert data.shape[1] == len(sample_epochs.ch_names)
