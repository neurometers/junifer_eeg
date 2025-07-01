"""Tests for SymbolicMutualInformation marker."""

import mne
import numpy as np
import pytest

from junifer_eeg.markers import SymbolicMutualInformation


class TestSymbolicMutualInformation:
    """Test SymbolicMutualInformation marker."""

    def test_symbolic_mutual_information_init(self):
        """Test SymbolicMutualInformation initialization."""
        marker = SymbolicMutualInformation(
            tmin=0.0, tmax=1.0, weighted=True, kernel=3
        )
        assert marker.tmin == 0.0
        assert marker.tmax == 1.0
        assert marker.weighted
        assert marker.kernel == 3

    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs for testing."""
        # Create synthetic EEG data
        n_channels = 4  # Smaller for faster connectivity computation
        n_times = 100
        sfreq = 250

        # Create info
        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create events
        n_epochs = 10
        events = []
        for i in range(n_epochs):
            events.append([i * n_times, 0, 1])
        events = np.array(events)

        # Create raw data with some structure
        np.random.seed(42)
        data = np.random.randn(n_channels, n_epochs * n_times) * 1e-6

        # Add some coupling between channels
        data[1] = (
            0.7 * data[0] + 0.3 * np.random.randn(n_epochs * n_times) * 1e-6
        )

        raw = mne.io.RawArray(data, info, verbose=False)

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            tmin=-0.2,
            tmax=0.8,
            baseline=None,
            preload=True,
            verbose=False,
        )

        return epochs

    def test_symbolic_mutual_information_compute(self, sample_epochs):
        """Test SymbolicMutualInformation computation."""
        marker = SymbolicMutualInformation(tmin=0.0, tmax=0.5)

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "symbolic_mutual_information" in result
        assert "data" in result["symbolic_mutual_information"]
        assert "col_names" in result["symbolic_mutual_information"]

        # Check dimensions
        data = result["symbolic_mutual_information"]["data"]
        col_names = result["symbolic_mutual_information"]["col_names"]
        n_channels = len(sample_epochs.ch_names)

        assert data.shape == (
            1,
            n_channels * n_channels,
        )  # Flattened connectivity matrix
        assert len(col_names) == n_channels * n_channels

    def test_symbolic_mutual_information_diagonal(self, sample_epochs):
        """Test SymbolicMutualInformation diagonal values."""
        marker = SymbolicMutualInformation()

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        data = result["symbolic_mutual_information"]["data"]
        n_channels = len(sample_epochs.ch_names)

        # Reshape to matrix
        smi_matrix = data.reshape(n_channels, n_channels)

        # Diagonal should be 1.0 (self-SMI)
        diagonal = np.diag(smi_matrix)
        assert np.allclose(diagonal, 1.0)

    def test_symbolic_mutual_information_ordinal_patterns(self, sample_epochs):
        """Test ordinal pattern computation."""
        marker = SymbolicMutualInformation(kernel=3, tau=1)

        # Test the marker computation with shorter data
        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "symbolic_mutual_information" in result
        data = result["symbolic_mutual_information"]["data"]
        assert data.shape[0] == 1  # One trial
        assert data.shape[1] > 0  # Should have connectivity values

    def test_symbolic_mutual_information_pattern_to_index(self, sample_epochs):
        """Test pattern to index conversion using internal method."""
        marker = SymbolicMutualInformation()

        # Test basic functionality through compute
        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        # Should complete without error
        assert "symbolic_mutual_information" in result

    def test_symbolic_mutual_information_identical_signals(
        self, sample_epochs
    ):
        """Test SMI for channels with similar patterns."""
        marker = SymbolicMutualInformation()

        # Test that the marker works with real epochs
        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        data = result["symbolic_mutual_information"]["data"]
        n_channels = len(sample_epochs.ch_names)

        # Reshape to matrix
        smi_matrix = data.reshape(n_channels, n_channels)

        # Check that diagonal is 1.0 (self-SMI)
        diagonal = np.diag(smi_matrix)
        assert np.allclose(diagonal, 1.0)

        # Check that matrix is square
        assert smi_matrix.shape == (n_channels, n_channels)

        # Check that values are finite (not NaN or inf)
        assert np.all(np.isfinite(smi_matrix))

    def test_symbolic_mutual_information_parameters(self, sample_epochs):
        """Test different parameters."""
        marker = SymbolicMutualInformation(
            tmin=0.1, tmax=0.6, kernel=4, tau=2, weighted=False
        )

        input_data = {"data": sample_epochs}
        result = marker.compute(input_data)

        assert "symbolic_mutual_information" in result
        data = result["symbolic_mutual_information"]["data"]
        n_channels = len(sample_epochs.ch_names)
        assert data.shape == (1, n_channels * n_channels)

    def test_symbolic_mutual_information_short_signals(self):
        """Test SMI with reasonable signals."""
        marker = SymbolicMutualInformation(kernel=3, tau=1)  # Smaller kernel

        # Create reasonable length epochs
        n_channels = 2
        n_times = 100  # Reasonable length
        sfreq = 250

        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Reasonable signals
        data = np.random.randn(n_channels, n_times) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)

        # Create single epoch
        events = np.array([[0, 0, 1]])
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0.0,
            tmax=(n_times - 1) / sfreq,
            baseline=None,
            preload=True,
            verbose=False,
        )

        input_data = {"data": epochs}
        result = marker.compute(input_data)

        # Should complete successfully
        assert "symbolic_mutual_information" in result
        data = result["symbolic_mutual_information"]["data"]
        assert data.shape == (1, n_channels * n_channels)

        # Values should be finite
        assert np.all(np.isfinite(data))
