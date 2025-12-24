"""Test script for SlowWavesDetection marker.

Tests the slow waves detection marker that:
1. Has a mandatory `feature` parameter
2. Uses singleton pattern with caching for efficiency
3. Returns only the requested feature per epoch/channel
"""

import numpy as np
import pytest
from mne import create_info
from mne.epochs import EpochsArray

from junifer_eeg.markers.slow_waves_detection import (
    SLOW_WAVE_FEATURES,
    SlowWavesDetection,
    SlowWavesDetectionBase,
)


@pytest.fixture(scope="function")
def synthetic_sleep_epochs():
    """Create synthetic EEG epochs with slow wave-like patterns for testing."""
    n_epochs = 5
    n_channels = 8
    n_times = 1000
    sfreq = 250.0

    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6

    # Add slow wave-like oscillations (0.5-4 Hz)
    t = np.arange(n_times) / sfreq
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            if np.random.random() < 0.4:
                sw_freq = np.random.uniform(0.5, 4)
                sw_start = np.random.randint(100, 700)
                sw_end = sw_start + np.random.randint(200, 400)
                sw_t = t[sw_start:sw_end]
                sw_signal = 2e-6 * np.sin(2 * np.pi * sw_freq * sw_t)
                data[epoch_idx, ch_idx, sw_start:sw_end] += sw_signal

    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
    epochs = EpochsArray(data, info, events=events, tmin=0.0)

    return epochs


class TestSlowWavesDetectionMandatoryFeature:
    """Test mandatory feature parameter validation."""

    def test_feature_is_mandatory(self):
        """Test that feature parameter is required."""
        with pytest.raises(TypeError):
            SlowWavesDetection()

    def test_invalid_feature_raises_error(self):
        """Test that invalid feature raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SlowWavesDetection(feature="InvalidFeature")

        assert "MANDATORY" in str(exc_info.value)
        assert "Duration" in str(exc_info.value)

    @pytest.mark.parametrize("feature", SLOW_WAVE_FEATURES)
    def test_valid_features_accepted(self, feature):
        """Test that all valid features are accepted."""
        marker = SlowWavesDetection(feature=feature)
        assert marker.feature == feature


class TestSlowWavesDetectionFeatureComputation:
    """Test that all features can be computed correctly."""

    @pytest.mark.parametrize("feature", SLOW_WAVE_FEATURES)
    def test_feature_computation(self, synthetic_sleep_epochs, feature):
        """Test that each feature can be computed."""
        marker = SlowWavesDetection(
            feature=feature,
            freq_sw=(0.5, 4),
            channel_method=None,
            trial_method=None,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

        # Should return 2D array (n_epochs, n_channels)
        assert data.shape == (5, 8), f"Expected (5, 8), got {data.shape}"
        assert isinstance(data, np.ndarray)


class TestSlowWavesDetectionCaching:
    """Test singleton caching behavior."""

    def test_singleton_pattern(self):
        """Test that singleton pattern works correctly."""
        # Get two instances - should be the same object
        base1 = SlowWavesDetectionBase()
        base2 = SlowWavesDetectionBase()
        assert base1 is base2, "Singleton should return same instance"

    def test_cache_stores_all_features(self, synthetic_sleep_epochs):
        """Test that cache stores all features from single computation."""
        SlowWavesDetectionBase._cache.clear()

        # Compute one feature
        marker = SlowWavesDetection(feature="Duration")
        marker.compute({"data": synthetic_sleep_epochs})

        # Cache should have at least one entry
        assert len(SlowWavesDetectionBase._cache) >= 1

        # Check that cached entry contains all features
        cache_values = list(SlowWavesDetectionBase._cache.values())
        assert len(cache_values) >= 1
        cached_features = cache_values[0]
        assert "Duration" in cached_features
        assert "PTP" in cached_features
        assert "Frequency" in cached_features
        assert "Slope" in cached_features
        assert "Density" in cached_features

    def test_multiple_features_same_marker_params(
        self, synthetic_sleep_epochs
    ):
        """Test that different features with same params work correctly."""
        # Test that we can extract different features
        marker_dur = SlowWavesDetection(feature="Duration")
        marker_ptp = SlowWavesDetection(feature="PTP")

        result_dur = marker_dur.compute({"data": synthetic_sleep_epochs})
        result_ptp = marker_ptp.compute({"data": synthetic_sleep_epochs})

        # Both should produce valid results with same shape
        assert result_dur["slowwavesdetection"]["data"].shape == (5, 8)
        assert result_ptp["slowwavesdetection"]["data"].shape == (5, 8)

    def test_different_params_new_cache_entry(self, synthetic_sleep_epochs):
        """Test that different parameters create new cache entries."""
        SlowWavesDetectionBase._cache.clear()

        # First marker with default params
        marker1 = SlowWavesDetection(feature="Duration", freq_sw=(0.3, 1.5))
        marker1.compute({"data": synthetic_sleep_epochs})

        initial_cache_size = len(SlowWavesDetectionBase._cache)

        # Second marker with different freq_sw
        marker2 = SlowWavesDetection(feature="Duration", freq_sw=(0.5, 4.0))
        marker2.compute({"data": synthetic_sleep_epochs})

        # Cache should have grown
        assert len(SlowWavesDetectionBase._cache) > initial_cache_size


class TestSlowWavesDetectionAggregation:
    """Test aggregation methods work correctly."""

    def test_channel_aggregation(self, synthetic_sleep_epochs):
        """Test channel aggregation preserves 2D with size 1."""
        marker = SlowWavesDetection(
            feature="Duration",
            channel_method="mean",
            trial_method=None,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

        # After channel aggregation: (n_epochs, 1) - preserved dimensions
        assert data.shape == (5, 1), f"Expected (5, 1), got {data.shape}"

    def test_trial_aggregation(self, synthetic_sleep_epochs):
        """Test trial aggregation preserves 2D with size 1."""
        marker = SlowWavesDetection(
            feature="Duration",
            channel_method=None,
            trial_method="mean",
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

        # After trial aggregation: (1, n_channels) - preserved dimensions
        assert data.shape == (1, 8), f"Expected (1, 8), got {data.shape}"

    def test_both_aggregations(self, synthetic_sleep_epochs):
        """Test both aggregations preserve 2D with size (1, 1)."""
        marker = SlowWavesDetection(
            feature="Duration",
            channel_method="mean",
            trial_method="mean",
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

        # After both aggregations: (1, 1) - preserved dimensions
        assert data.shape == (1, 1), f"Expected (1, 1), got {data.shape}"

    @pytest.mark.parametrize(
        "method", ["mean", "std", "median", "trim_mean80", "trim_mean90"]
    )
    def test_aggregation_methods(self, synthetic_sleep_epochs, method):
        """Test various aggregation methods work."""
        marker = SlowWavesDetection(
            feature="Density",
            channel_method=method,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        assert "data" in result["slowwavesdetection"]


class TestSlowWavesDetectionOutputType:
    """Test get_output_type method."""

    def test_no_aggregation_returns_timeseries(self):
        """Test no aggregation returns timeseries."""
        marker = SlowWavesDetection(feature="Duration")
        assert (
            marker.get_output_type("EEG", "slowwavesdetection") == "timeseries"
        )

    def test_channel_aggregation_returns_timeseries(self):
        """Test channel aggregation returns timeseries (preserved dimensions)."""
        marker = SlowWavesDetection(feature="Duration", channel_method="mean")
        assert (
            marker.get_output_type("EEG", "slowwavesdetection") == "timeseries"
        )

    def test_trial_aggregation_returns_timeseries(self):
        """Test trial aggregation returns timeseries (preserved dimensions)."""
        marker = SlowWavesDetection(feature="Duration", trial_method="mean")
        assert (
            marker.get_output_type("EEG", "slowwavesdetection") == "timeseries"
        )

    def test_both_aggregations_returns_timeseries(self):
        """Test both aggregations returns timeseries (preserved dimensions)."""
        marker = SlowWavesDetection(
            feature="Duration",
            channel_method="mean",
            trial_method="mean",
        )
        assert (
            marker.get_output_type("EEG", "slowwavesdetection") == "timeseries"
        )


class TestSlowWavesDetectionFeatures:
    """Test specific feature characteristics."""

    @pytest.mark.parametrize("feature", SLOW_WAVE_FEATURES)
    def test_all_features_compute(self, synthetic_sleep_epochs, feature):
        """Test that all features can be computed."""
        marker = SlowWavesDetection(feature=feature)
        result = marker.compute({"data": synthetic_sleep_epochs})

        assert "slowwavesdetection" in result
        assert "data" in result["slowwavesdetection"]
        data = result["slowwavesdetection"]["data"]
        assert data.shape == (5, 8)  # (n_epochs, n_channels)

    def test_feature_values_reasonable(self, synthetic_sleep_epochs):
        """Test that feature values are in reasonable ranges."""
        # Duration should be positive
        marker_dur = SlowWavesDetection(feature="Duration")
        result_dur = marker_dur.compute({"data": synthetic_sleep_epochs})
        dur_data = result_dur["slowwavesdetection"]["data"]
        valid_dur = dur_data[~np.isnan(dur_data)]
        if len(valid_dur) > 0:
            assert np.all(valid_dur > 0), "Duration should be positive"

        # Frequency should be positive
        marker_freq = SlowWavesDetection(feature="Frequency")
        result_freq = marker_freq.compute({"data": synthetic_sleep_epochs})
        freq_data = result_freq["slowwavesdetection"]["data"]
        valid_freq = freq_data[~np.isnan(freq_data)]
        if len(valid_freq) > 0:
            assert np.all(valid_freq > 0), "Frequency should be positive"

        # Density should be non-negative integers
        marker_dens = SlowWavesDetection(feature="Density")
        result_dens = marker_dens.compute({"data": synthetic_sleep_epochs})
        dens_data = result_dens["slowwavesdetection"]["data"]
        valid_dens = dens_data[~np.isnan(dens_data)]
        if len(valid_dens) > 0:
            assert np.all(valid_dens >= 0), "Density should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
