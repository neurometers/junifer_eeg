"""Test script for SpindlesDetection marker.

Tests the spindles detection marker that:
1. Has a mandatory `feature` parameter
2. Uses singleton pattern with caching for efficiency
3. Returns only the requested feature per epoch/channel
"""

import numpy as np
import pytest
from mne import create_info
from mne.epochs import EpochsArray

from junifer_eeg.markers.spindles_detection import (
    SPINDLE_FEATURES,
    SpindlesDetection,
    SpindlesDetectionBase,
)


@pytest.fixture(scope="function")
def synthetic_sleep_epochs():
    """Create synthetic EEG epochs with spindle-like patterns for testing."""
    n_epochs = 5
    n_channels = 8
    n_times = 1000
    sfreq = 250.0

    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6

    # Add spindle-like oscillations (12-15 Hz)
    t = np.arange(n_times) / sfreq
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            if np.random.random() < 0.3:
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

    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
    epochs = EpochsArray(data, info, events=events, tmin=0.0)

    return epochs


class TestSpindlesDetectionMandatoryFeature:
    """Test mandatory feature parameter validation."""

    def test_feature_is_mandatory(self):
        """Test that feature parameter is required."""
        with pytest.raises(TypeError):
            SpindlesDetection()

    def test_invalid_feature_raises_error(self):
        """Test that invalid feature raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SpindlesDetection(feature="InvalidFeature")

        assert "MANDATORY" in str(exc_info.value)
        assert "Duration" in str(exc_info.value)

    @pytest.mark.parametrize("feature", SPINDLE_FEATURES)
    def test_valid_features_accepted(self, feature):
        """Test that all valid features are accepted."""
        marker = SpindlesDetection(feature=feature)
        assert marker.feature == feature


class TestSpindlesDetectionFeatureComputation:
    """Test that all features can be computed correctly."""

    @pytest.mark.parametrize("feature", SPINDLE_FEATURES)
    def test_feature_computation(self, synthetic_sleep_epochs, feature):
        """Test that each feature can be computed."""
        marker = SpindlesDetection(
            feature=feature,
            freq_sp=(11, 16),
            freq_broad=(1, 30),
            channel_method=None,
            trial_method=None,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["spindlesdetection"]["data"]

        # Should return 2D array (n_epochs, n_channels)
        assert data.shape == (5, 8), f"Expected (5, 8), got {data.shape}"
        assert isinstance(data, np.ndarray)


class TestSpindlesDetectionCaching:
    """Test singleton caching behavior."""

    def test_singleton_pattern(self):
        """Test that singleton pattern works correctly."""
        # Get two instances - should be the same object
        base1 = SpindlesDetectionBase()
        base2 = SpindlesDetectionBase()
        assert base1 is base2, "Singleton should return same instance"

    def test_cache_stores_all_features(self, synthetic_sleep_epochs):
        """Test that cache stores all features from single computation."""
        SpindlesDetectionBase._cache.clear()

        # Compute one feature
        marker = SpindlesDetection(feature="Duration")
        marker.compute({"data": synthetic_sleep_epochs})

        # Cache should have at least one entry
        assert len(SpindlesDetectionBase._cache) >= 1

        # Check that cached entry contains all features
        cache_values = list(SpindlesDetectionBase._cache.values())
        assert len(cache_values) >= 1
        cached_features = cache_values[0]
        assert "Duration" in cached_features
        assert "Amplitude" in cached_features
        assert "Frequency" in cached_features
        assert "Density" in cached_features

    def test_multiple_features_same_marker_params(
        self, synthetic_sleep_epochs
    ):
        """Test that different features with same params work correctly."""
        # Test that we can extract different features
        marker_dur = SpindlesDetection(feature="Duration")
        marker_amp = SpindlesDetection(feature="Amplitude")

        result_dur = marker_dur.compute({"data": synthetic_sleep_epochs})
        result_amp = marker_amp.compute({"data": synthetic_sleep_epochs})

        # Both should produce valid results with same shape
        assert result_dur["spindlesdetection"]["data"].shape == (5, 8)
        assert result_amp["spindlesdetection"]["data"].shape == (5, 8)

    def test_different_params_new_cache_entry(self, synthetic_sleep_epochs):
        """Test that different parameters create new cache entries."""
        SpindlesDetectionBase._cache.clear()

        # First marker with default params
        marker1 = SpindlesDetection(feature="Duration", freq_sp=(12, 15))
        marker1.compute({"data": synthetic_sleep_epochs})

        initial_cache_size = len(SpindlesDetectionBase._cache)

        # Second marker with different freq_sp
        marker2 = SpindlesDetection(feature="Duration", freq_sp=(11, 16))
        marker2.compute({"data": synthetic_sleep_epochs})

        # Cache should have grown
        assert len(SpindlesDetectionBase._cache) > initial_cache_size


class TestSpindlesDetectionAggregation:
    """Test aggregation methods work correctly."""

    def test_channel_aggregation(self, synthetic_sleep_epochs):
        """Test channel aggregation preserves 2D with size 1."""
        marker = SpindlesDetection(
            feature="Duration",
            channel_method="mean",
            trial_method=None,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["spindlesdetection"]["data"]

        # After channel aggregation: (n_epochs, 1) - preserved dimensions
        assert data.shape == (5, 1), f"Expected (5, 1), got {data.shape}"

    def test_trial_aggregation(self, synthetic_sleep_epochs):
        """Test trial aggregation preserves 2D with size 1."""
        marker = SpindlesDetection(
            feature="Duration",
            channel_method=None,
            trial_method="mean",
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["spindlesdetection"]["data"]

        # After trial aggregation: (1, n_channels) - preserved dimensions
        assert data.shape == (1, 8), f"Expected (1, 8), got {data.shape}"

    def test_both_aggregations(self, synthetic_sleep_epochs):
        """Test both aggregations preserve 2D with size (1, 1)."""
        marker = SpindlesDetection(
            feature="Duration",
            channel_method="mean",
            trial_method="mean",
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["spindlesdetection"]["data"]

        # After both aggregations: (1, 1) - preserved dimensions
        assert data.shape == (1, 1), f"Expected (1, 1), got {data.shape}"

    @pytest.mark.parametrize(
        "method", ["mean", "std", "median", "trim_mean80", "trim_mean90"]
    )
    def test_aggregation_methods(self, synthetic_sleep_epochs, method):
        """Test various aggregation methods work."""
        marker = SpindlesDetection(
            feature="Density",
            channel_method=method,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        assert "data" in result["spindlesdetection"]


class TestSpindlesDetectionOutputType:
    """Test get_output_type method."""

    def test_no_aggregation_returns_timeseries(self):
        """Test no aggregation returns timeseries."""
        marker = SpindlesDetection(feature="Duration")
        assert (
            marker.get_output_type("EEG", "spindlesdetection") == "timeseries"
        )

    def test_channel_aggregation_returns_timeseries(self):
        """Test channel aggregation returns timeseries (preserved dimensions)."""
        marker = SpindlesDetection(feature="Duration", channel_method="mean")
        assert (
            marker.get_output_type("EEG", "spindlesdetection") == "timeseries"
        )

    def test_trial_aggregation_returns_timeseries(self):
        """Test trial aggregation returns timeseries (preserved dimensions)."""
        marker = SpindlesDetection(feature="Duration", trial_method="mean")
        assert (
            marker.get_output_type("EEG", "spindlesdetection") == "timeseries"
        )

    def test_both_aggregations_returns_timeseries(self):
        """Test both aggregations returns timeseries (preserved dimensions)."""
        marker = SpindlesDetection(
            feature="Duration",
            channel_method="mean",
            trial_method="mean",
        )
        assert (
            marker.get_output_type("EEG", "spindlesdetection") == "timeseries"
        )


class TestSpindlesDetectionDensityZeroSemantics:
    """Density is a count: cells with no detected spindles must be 0, not NaN."""

    @staticmethod
    def _flat_epochs():
        """Epochs with no detectable spindles on any (epoch, channel) cell."""
        n_epochs, n_channels, n_times, sfreq = 3, 4, 1000, 250.0
        data = np.zeros((n_epochs, n_channels, n_times))
        ch_names = [f"E{i + 1}" for i in range(n_channels)]
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
        return EpochsArray(data, info, events=events, tmin=0.0)

    def test_density_is_zero_not_nan_when_no_spindles(self):
        """Flat signal yields zero detections → Density=0 everywhere."""
        SpindlesDetectionBase._cache.clear()
        marker = SpindlesDetection(
            feature="Density",
            freq_sp=(11, 16),
            freq_broad=(1, 30),
            channel_method=None,
            trial_method=None,
        )
        result = marker.compute({"data": self._flat_epochs()})
        data = result["spindlesdetection"]["data"]

        assert not np.any(np.isnan(data)), (
            "Density returned NaN; should be 0 for cells with no detected spindles"
        )
        assert (data >= 0).all()
        assert np.array_equal(data, data.astype(int))
        assert np.all(data == 0)

    def test_mean_features_remain_nan_when_no_spindles(self):
        """Duration/Amplitude/Frequency are means; with no spindles they stay NaN."""
        SpindlesDetectionBase._cache.clear()
        epochs = self._flat_epochs()
        for feature in ("Duration", "Amplitude", "Frequency"):
            marker = SpindlesDetection(
                feature=feature,
                freq_sp=(11, 16),
                freq_broad=(1, 30),
                channel_method=None,
                trial_method=None,
            )
            data = marker.compute({"data": epochs})["spindlesdetection"][
                "data"
            ]
            assert np.all(np.isnan(data)), (
                f"{feature}: expected all NaN with no detections, got {data}"
            )

    def test_density_no_nan_when_spindles_exist(self, synthetic_sleep_epochs):
        """Where detections do exist, output must still be NaN-free integer counts."""
        SpindlesDetectionBase._cache.clear()
        marker = SpindlesDetection(
            feature="Density",
            freq_sp=(11, 16),
            freq_broad=(1, 30),
            channel_method=None,
            trial_method=None,
        )
        data = marker.compute({"data": synthetic_sleep_epochs})[
            "spindlesdetection"
        ]["data"]
        assert not np.any(np.isnan(data))
        assert (data >= 0).all()
        assert np.array_equal(data, data.astype(int))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
