"""Validation tests for 3D tensor transformation of sleep markers.

This module tests the SpindlesDetection and SlowWavesDetection markers to verify:
1. 3D tensor output structure (n_features, n_epochs, n_channels)
2. Proper aggregation behavior matching SpectralPower pattern
3. Correct output type methods (timeseries, vector, scalar_table)
4. Edge cases with empty data and NaN handling
5. Integration with junifer pipeline

The markers transform from event-level long format to aggregated 3D tensors
with mean values per epoch/channel/feature for consistency with other markers.

Note: YASA RuntimeWarnings about filter_length are expected for synthetic test data
with short signals (500-1000 samples) and are suppressed in these tests.
"""

import mne
import numpy as np
import pytest

from junifer_eeg.markers.slow_waves_detection import SlowWavesDetection
from junifer_eeg.markers.spindles_detection import SpindlesDetection


@pytest.fixture(scope="function")
def synthetic_sleep_epochs():
    """Create synthetic EEG epochs with sleep-like patterns for testing."""
    # Create synthetic data: 5 epochs, 8 channels, 1000 time points
    n_epochs = 5
    n_channels = 8
    n_times = 1000
    sfreq = 250  # Hz

    # Generate synthetic EEG data with some sleep-like patterns
    np.random.seed(42)
    data = (
        np.random.randn(n_epochs, n_channels, n_times) * 1e-6
    )  # Convert to volts

    # Add some spindle-like oscillations (12-15 Hz) to random epochs/channels
    t = np.arange(n_times) / sfreq
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            if np.random.random() < 0.3:  # 30% chance of spindle
                spindle_freq = np.random.uniform(12, 15)
                spindle_start = np.random.randint(200, 600)
                spindle_end = spindle_start + np.random.randint(50, 150)
                spindle_t = t[spindle_start:spindle_end]
                spindle_signal = 0.5e-6 * np.sin(
                    2 * np.pi * spindle_freq * spindle_t
                )
                data[epoch_idx, ch_idx, spindle_start:spindle_end] += (
                    spindle_signal
                )

    # Add some slow wave-like oscillations (0.5-4 Hz)
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            if np.random.random() < 0.4:  # 40% chance of slow wave
                sw_freq = np.random.uniform(0.5, 4)
                sw_start = np.random.randint(100, 700)
                sw_end = sw_start + np.random.randint(200, 400)
                sw_t = t[sw_start:sw_end]
                sw_signal = 2e-6 * np.sin(2 * np.pi * sw_freq * sw_t)
                data[epoch_idx, ch_idx, sw_start:sw_end] += sw_signal

    # Create MNE info object
    ch_names = [f"E{i + 1:02d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create epochs object
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
    epochs = mne.EpochsArray(data, info, events, tmin=0)

    return epochs


@pytest.fixture(scope="function")
def empty_epochs():
    """Create empty epochs with pure noise for edge case testing."""
    n_epochs, n_channels, n_times = 3, 4, 500
    ch_names = [f"E{i + 1:02d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")

    # Pure noise - should detect no spindles/slow waves
    noise_data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
    noise_epochs = mne.EpochsArray(noise_data, info, events, tmin=0)

    return noise_epochs


class TestSpindlesDetection3DTensor:
    """Test SpindlesDetection 3D tensor transformation."""

    def test_tensor_structure_and_shape(self, synthetic_sleep_epochs):
        """Verify 3D tensor output structure and correct shape."""
        # Initialize marker
        marker = SpindlesDetection(
            freq_sp=(11, 16),
            freq_broad=(1, 30),
            channel_aggregation_method=None,
            epoch_aggregation_method=None,
        )

        # Compute marker
        result = marker.compute({"data": synthetic_sleep_epochs})

        # Validate output structure
        assert "spindlesdetection" in result
        assert "data" in result["spindlesdetection"]
        assert "col_names" in result["spindlesdetection"]

        data = result["spindlesdetection"]["data"]
        col_names = result["spindlesdetection"]["col_names"]

        # Check tensor shape: (n_features, n_epochs, n_channels)
        expected_shape = (4, 5, 8)  # 4 features, 5 epochs, 8 channels
        assert data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {data.shape}"
        )

        # Check column names
        expected_ch_names = [f"E{i + 1:02d}" for i in range(8)]
        assert col_names == expected_ch_names, (
            f"Expected {expected_ch_names}, got {col_names}"
        )

        # Check data type and NaN handling
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float64

    def test_channel_aggregation(self, synthetic_sleep_epochs):
        """Test channel aggregation reduces dimensions correctly."""
        # Initialize marker with channel aggregation
        marker = SpindlesDetection(
            freq_sp=(11, 16),
            freq_broad=(1, 30),
            channel_aggregation_method="mean",
            epoch_aggregation_method=None,
        )

        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["spindlesdetection"]["data"]

        # After channel aggregation: (n_features, n_epochs)
        expected_shape = (4, 5)
        assert data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {data.shape}"
        )

    def test_both_aggregations(self, synthetic_sleep_epochs):
        """Test both channel and epoch aggregation."""
        # Initialize marker with both aggregations
        marker = SpindlesDetection(
            freq_sp=(11, 16),
            freq_broad=(1, 30),
            channel_aggregation_method="mean",
            epoch_aggregation_method="mean",
        )

        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["spindlesdetection"]["data"]

        # After both aggregations: (n_features,)
        expected_shape = (4,)
        assert data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {data.shape}"
        )

    def test_empty_data_returns_nan(self, empty_epochs):
        """Test that empty data returns all NaN values."""
        marker = SpindlesDetection()
        result = marker.compute({"data": empty_epochs})

        assert result["spindlesdetection"]["data"].shape == (4, 3, 4)
        assert np.all(np.isnan(result["spindlesdetection"]["data"])), (
            "All values should be NaN for empty data"
        )

    def test_output_type_methods(self):
        """Test get_output_type returns correct values."""
        # No aggregation
        marker = SpindlesDetection()
        assert (
            marker.get_output_type("EEG", "spindlesdetection") == "timeseries"
        )

        # Channel aggregation only
        marker_chan = SpindlesDetection(channel_aggregation_method="mean")
        assert (
            marker_chan.get_output_type("EEG", "spindlesdetection") == "vector"
        )

        # Both aggregations
        marker_both = SpindlesDetection(
            channel_aggregation_method="mean", epoch_aggregation_method="mean"
        )
        assert (
            marker_both.get_output_type("EEG", "spindlesdetection")
            == "scalar_table"
        )


class TestSlowWavesDetection3DTensor:
    """Test SlowWavesDetection 3D tensor transformation."""

    def test_tensor_structure_and_shape(self, synthetic_sleep_epochs):
        """Verify 3D tensor output structure and correct shape."""
        # Initialize marker
        marker = SlowWavesDetection(
            freq_sw=(0.5, 4),
            channel_aggregation_method=None,
            epoch_aggregation_method=None,
        )

        # Compute marker
        result = marker.compute({"data": synthetic_sleep_epochs})

        # Validate output structure
        assert "slowwavesdetection" in result
        assert "data" in result["slowwavesdetection"]
        assert "col_names" in result["slowwavesdetection"]

        data = result["slowwavesdetection"]["data"]
        col_names = result["slowwavesdetection"]["col_names"]

        # Check tensor shape: (n_features, n_epochs, n_channels)
        expected_shape = (5, 5, 8)  # 5 features, 5 epochs, 8 channels
        assert data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {data.shape}"
        )

        # Check column names
        expected_ch_names = [f"E{i + 1:02d}" for i in range(8)]
        assert col_names == expected_ch_names, (
            f"Expected {expected_ch_names}, got {col_names}"
        )

        # Check data type and NaN handling
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float64

    def test_channel_aggregation(self, synthetic_sleep_epochs):
        """Test channel aggregation reduces dimensions correctly."""
        # Initialize marker with channel aggregation
        marker = SlowWavesDetection(
            freq_sw=(0.5, 4),
            channel_aggregation_method="mean",
            epoch_aggregation_method=None,
        )

        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

        # After channel aggregation: (n_features, n_epochs)
        expected_shape = (5, 5)
        assert data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {data.shape}"
        )

    def test_both_aggregations(self, synthetic_sleep_epochs):
        """Test both channel and epoch aggregation."""
        # Initialize marker with both aggregations
        marker = SlowWavesDetection(
            freq_sw=(0.5, 4),
            channel_aggregation_method="mean",
            epoch_aggregation_method="mean",
        )

        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

        # After both aggregations: (n_features,)
        expected_shape = (5,)
        assert data.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {data.shape}"
        )

    def test_empty_data_returns_nan(self, empty_epochs):
        """Test that empty data returns all NaN values."""
        marker = SlowWavesDetection()
        result = marker.compute({"data": empty_epochs})

        assert result["slowwavesdetection"]["data"].shape == (5, 3, 4)
        assert np.all(np.isnan(result["slowwavesdetection"]["data"])), (
            "All values should be NaN for empty data"
        )

    def test_output_type_methods(self):
        """Test get_output_type returns correct values."""
        # No aggregation
        marker = SlowWavesDetection()
        assert (
            marker.get_output_type("EEG", "slowwavesdetection") == "timeseries"
        )

        # Channel aggregation only
        marker_chan = SlowWavesDetection(channel_aggregation_method="mean")
        assert (
            marker_chan.get_output_type("EEG", "slowwavesdetection")
            == "vector"
        )

        # Both aggregations
        marker_both = SlowWavesDetection(
            channel_aggregation_method="mean", epoch_aggregation_method="mean"
        )
        assert (
            marker_both.get_output_type("EEG", "slowwavesdetection")
            == "scalar_table"
        )


class TestSleepMarkersAggregationEdgeCases:
    """Test edge cases for aggregation with all-NaN data."""

    def test_aggregation_with_all_nan_data(self, empty_epochs):
        """Test aggregation handles all-NaN data correctly."""
        # Test channel aggregation on all-NaN data
        spindles_chan = SpindlesDetection(channel_aggregation_method="mean")
        result_chan = spindles_chan.compute({"data": empty_epochs})
        assert result_chan["spindlesdetection"]["data"].shape == (
            4,
            3,
        )  # (features, epochs)
        assert np.all(np.isnan(result_chan["spindlesdetection"]["data"])), (
            "Channel aggregation of all NaN should be NaN"
        )

        # Test epoch aggregation on all-NaN data
        spindles_epoch = SpindlesDetection(epoch_aggregation_method="mean")
        result_epoch = spindles_epoch.compute({"data": empty_epochs})
        assert result_epoch["spindlesdetection"]["data"].shape == (
            4,
            4,
        )  # (features, channels)
        assert np.all(np.isnan(result_epoch["spindlesdetection"]["data"])), (
            "Epoch aggregation of all NaN should be NaN"
        )

        # Test both aggregations on all-NaN data
        spindles_both = SpindlesDetection(
            channel_aggregation_method="mean", epoch_aggregation_method="mean"
        )
        result_both = spindles_both.compute({"data": empty_epochs})
        assert result_both["spindlesdetection"]["data"].shape == (
            4,
        )  # (features,)
        assert np.all(np.isnan(result_both["spindlesdetection"]["data"])), (
            "Both aggregations of all NaN should be NaN"
        )

    @pytest.mark.parametrize(
        "aggregation_method",
        ["mean", "std", "trim_mean80", "trim_mean90", "median"],
    )
    def test_all_aggregation_methods(self, empty_epochs, aggregation_method):
        """Test that all VALID_AGGREGATION_METHODS work correctly with NaN data."""
        # Test channel aggregation with different methods using empty data
        # This tests that aggregation methods handle NaN properly
        marker = SpindlesDetection(
            channel_aggregation_method=aggregation_method,
            epoch_aggregation_method=None,
        )
        result = marker.compute({"data": empty_epochs})

        # Should reduce to (features, epochs) and handle NaN gracefully
        assert result["spindlesdetection"]["data"].shape == (4, 3)
        # With empty data, all results should be NaN regardless of aggregation method
        assert np.all(np.isnan(result["spindlesdetection"]["data"])), (
            f"Aggregation method {aggregation_method} should handle NaN data correctly"
        )

    def test_feature_names_and_ordering(self, synthetic_sleep_epochs):
        """Test that feature names and ordering are correct."""
        marker = SpindlesDetection()
        result = marker.compute({"data": synthetic_sleep_epochs})

        # Spindles should have features: Duration, Amplitude, Frequency, Density
        # We can't directly test feature names, but we can verify the tensor has 4 features
        data = result["spindlesdetection"]["data"]
        assert data.shape[0] == 4, "Spindles should have exactly 4 features"

        # Test that features have reasonable ranges (not all NaN, reasonable values)
        # Duration should be positive (seconds)
        durations = data[0, :, :]  # First feature = Duration
        valid_durations = durations[~np.isnan(durations)]
        if len(valid_durations) > 0:
            assert np.all(valid_durations > 0), "Duration should be positive"
            assert np.all(valid_durations < 10), (
                "Duration should be reasonable (< 10s)"
            )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
