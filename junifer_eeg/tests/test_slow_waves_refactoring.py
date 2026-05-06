"""Test script for SlowWavesDetection marker.

Tests the slow waves detection marker that:
1. Has a mandatory `feature` parameter
2. Uses singleton pattern with caching for efficiency
3. Returns only the requested feature per epoch/channel
"""

import pandas as pd
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

    def test_invalid_detection_method_raises_error(self):
        """Only the supported slow-wave backends should be accepted."""
        with pytest.raises(ValueError, match="detection_method"):
            SlowWavesDetection(
                feature="Duration", detection_method="not_a_method"
            )

    def test_invalid_ptp_threshold_mode_raises_error(self):
        """Only supported PTP threshold modes should be accepted."""
        with pytest.raises(ValueError, match="ptp_threshold_mode"):
            SlowWavesDetection(
                feature="Duration", ptp_threshold_mode="not_a_mode"
            )


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

    def test_custom_method_computation(self, synthetic_sleep_epochs):
        """The custom zero-crossing backend should also return 2D output."""
        marker = SlowWavesDetection(
            feature="Density",
            detection_method="custom",
            freq_sw=(1.0, 10.0),
            channel_method=None,
            trial_method=None,
        )
        result = marker.compute({"data": synthetic_sleep_epochs})
        data = result["slowwavesdetection"]["data"]

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

    def test_different_detection_method_new_cache_entry(
        self, synthetic_sleep_epochs
    ):
        """Cache key should distinguish YASA from the custom backend."""
        SlowWavesDetectionBase._cache.clear()

        SlowWavesDetection(
            feature="Duration",
            detection_method="yasa",
            freq_sw=(0.5, 4.0),
        ).compute({"data": synthetic_sleep_epochs})
        initial_cache_size = len(SlowWavesDetectionBase._cache)

        SlowWavesDetection(
            feature="Duration",
            detection_method="custom",
            freq_sw=(1.0, 10.0),
        ).compute({"data": synthetic_sleep_epochs})

        assert len(SlowWavesDetectionBase._cache) > initial_cache_size


class TestSlowWavesDetectionFiltering:
    """Test post-detection filtering criteria."""

    def test_freq_threshold_removes_fast_waves(self):
        """Waves above freq_threshold should be discarded before aggregation."""
        sw_base = SlowWavesDetectionBase()
        sw_df = pd.DataFrame(
            {
                "Epoch": [0, 0, 0],
                "ChanIdx": [0, 0, 0],
                "Channel": ["E1", "E1", "E1"],
                "PTP": [10.0, 20.0, 30.0],
                "Frequency": [6.5, 7.0, 8.5],
                "Slope": [1.0, 2.0, 3.0],
            }
        )

        filtered = sw_base._apply_dynamic_threshold(
            sw_df,
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="adaptive",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
        )

        assert not filtered.empty
        assert filtered["Frequency"].max() <= 7.0
        assert 8.5 not in filtered["Frequency"].tolist()

    def test_precomputed_thresholds_are_used(self, tmp_path):
        """Fixed per-subject thresholds should bypass percentile computation."""
        sw_base = SlowWavesDetectionBase()
        sw_df = pd.DataFrame(
            {
                "Epoch": [0, 0, 0, 0],
                "ChanIdx": [0, 0, 1, 1],
                "Channel": ["E1", "E1", "E2", "E2"],
                "PTP": [30.0, 50.0, 20.0, 45.0],
                "Frequency": [6.0, 6.0, 6.0, 6.0],
                "Slope": [1.0, 2.0, 1.0, 2.0],
            }
        )
        thr_path = tmp_path / "thresholds.csv"
        pd.DataFrame(
            {
                "subject": ["03", "03"],
                "channel": ["E1", "E2"],
                "ptp_threshold": [45.0, 40.0],
            }
        ).to_csv(thr_path, index=False)

        filtered = sw_base._apply_dynamic_threshold(
            sw_df,
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="adaptive",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=str(thr_path),
            subject_id="03",
        )

        assert sorted(filtered["Channel"].tolist()) == ["E1", "E2"]
        assert sorted(filtered["PTP"].tolist()) == [45.0, 50.0]

    def test_missing_subject_thresholds_raise(self, tmp_path):
        """Providing a thresholds CSV without the current subject should fail."""
        sw_base = SlowWavesDetectionBase()
        sw_df = pd.DataFrame(
            {
                "Epoch": [0],
                "ChanIdx": [0],
                "Channel": ["E1"],
                "PTP": [40.0],
                "Frequency": [6.0],
                "Slope": [1.0],
            }
        )
        thr_path = tmp_path / "thresholds.csv"
        pd.DataFrame(
            {
                "subject": ["99"],
                "channel": ["E1"],
                "ptp_threshold": [35.0],
            }
        ).to_csv(thr_path, index=False)

        with pytest.raises(ValueError, match="No PTP thresholds found"):
            sw_base._apply_dynamic_threshold(
                sw_df,
                freq_threshold=7.0,
                artifact_threshold=75.0,
                ptp_threshold_mode="adaptive",
                ptp_percentile=90.0,
                max_ptp_amplitude=150.0,
                ptp_thresholds_path=str(thr_path),
                subject_id="03",
            )


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


class TestSlowWavesDetectionOptInExtensions:
    """Tests for opt-in extensions: proximity rule, strict flag, Le Coz support."""

    @staticmethod
    def _build_minimal_df():
        """Two waves on the same epoch/channel at known centers."""
        return pd.DataFrame(
            {
                "Epoch": [0, 0],
                "ChanIdx": [0, 0],
                "Channel": ["E1", "E1"],
                "Start": [0.10, 1.20],
                "End": [0.40, 1.50],
                "MidCrossing": [0.20, 1.35],
                "PTP": [30.0, 35.0],
                "Frequency": [3.0, 3.0],
                "Slope": [100.0, 110.0],
                "AscendingSlope": [80.0, 60.0],
                "DescendingSlope": [120.0, 90.0],
            }
        )

    def test_default_behavior_unchanged_when_optins_absent(self):
        """All new opt-in params default to None/False/identity values."""
        sw_base = SlowWavesDetectionBase()
        sw_df = self._build_minimal_df()

        # Old call signature (no opt-in kwargs).
        old = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="adaptive",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
        )
        # Equivalent call passing all new kwargs at their documented defaults.
        new = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="adaptive",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
            proximity_amplitude=None,
            proximity_window=1.0,
            ptp_thresholds_strict=True,
            slope_uv_per_s_range=None,
            filtered_per_epoch=None,
            sf=None,
        )
        pd.testing.assert_frame_equal(
            old.reset_index(drop=True), new.reset_index(drop=True)
        )

    def test_proximity_rule_drops_waves_near_artifact(self):
        """A wave whose center sits inside the artifact ±window is dropped."""
        sw_base = SlowWavesDetectionBase()
        sw_df = self._build_minimal_df()
        sf = 100.0
        n_samples = 200  # 0..2.0 s

        # Inject a 200 µV spike at t=0.50 s (sample 50). Wave 1 (center=0.20 s)
        # is >0.25 s away, so survives. Wave 2 (center=1.35 s) is also >0.25 s
        # from the spike, so survives too. Use proximity_window=0.6 s to catch
        # neither — then narrow the window and verify only the close wave goes.
        sig = np.zeros((1, n_samples), dtype=float)
        sig[0, 50] = 250.0  # large positive spike at 0.50 s
        filtered_per_epoch = {0: sig}

        # Window = 0.10 s → 0.50 ± 0.10 covers nothing near our centers.
        keep_all = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="fixed",  # skip percentile to isolate effect
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
            proximity_amplitude=150.0,
            proximity_window=0.10,
            filtered_per_epoch=filtered_per_epoch,
            sf=sf,
        )
        assert len(keep_all) == 2

        # Window = 0.40 s → 0.50 ± 0.40 = [0.10, 0.90] covers wave 1 (0.20 s).
        keep_one = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="fixed",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
            proximity_amplitude=150.0,
            proximity_window=0.40,
            filtered_per_epoch=filtered_per_epoch,
            sf=sf,
        )
        assert len(keep_one) == 1
        assert float(keep_one["MidCrossing"].iloc[0]) == pytest.approx(1.35)

    def test_proximity_disabled_by_default(self):
        """Without proximity_amplitude no waves are dropped even with artifacts."""
        sw_base = SlowWavesDetectionBase()
        sw_df = self._build_minimal_df()
        sig = np.full((1, 200), 1000.0)  # huge artifact everywhere
        out = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="fixed",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
            filtered_per_epoch={0: sig},
            sf=100.0,
        )
        assert len(out) == 2

    def test_ptp_thresholds_strict_false_returns_empty(self, tmp_path):
        """Non-strict mode warns and returns empty instead of raising."""
        sw_base = SlowWavesDetectionBase()
        sw_df = self._build_minimal_df()
        thr_path = tmp_path / "thresholds.csv"
        pd.DataFrame(
            {
                "subject": ["99"],
                "channel": ["E1"],
                "ptp_threshold": [35.0],
            }
        ).to_csv(thr_path, index=False)

        out = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="adaptive",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=str(thr_path),
            subject_id="03",
            ptp_thresholds_strict=False,
        )
        assert out.empty
        assert list(out.columns) == list(sw_df.columns)

    def test_slope_range_filter_drops_out_of_range_waves(self):
        """slope_uv_per_s_range drops waves whose slopes fall outside."""
        sw_base = SlowWavesDetectionBase()
        sw_df = self._build_minimal_df()
        # AscendingSlope: [80, 60]; DescendingSlope: [120, 90]
        # Range (70, 100) → wave 0 fails (Desc=120 > 100), wave 1 fails (Asc=60 < 70)
        out = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="fixed",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
            slope_uv_per_s_range=(70.0, 100.0),
        )
        assert out.empty

        # Range (50, 130) → both waves pass
        out2 = sw_base._apply_dynamic_threshold(
            sw_df.copy(),
            freq_threshold=7.0,
            artifact_threshold=75.0,
            ptp_threshold_mode="fixed",
            ptp_percentile=90.0,
            max_ptp_amplitude=150.0,
            ptp_thresholds_path=None,
            subject_id=None,
            slope_uv_per_s_range=(50.0, 130.0),
        )
        assert len(out2) == 2

    def test_custom_backend_emits_lecoz_columns(self, synthetic_sleep_epochs):
        """The custom detector must emit MidCrossing + Asc/Desc slope columns."""
        sw_base = SlowWavesDetectionBase()
        epochs = synthetic_sleep_epochs.copy()
        epochs._data -= epochs._data.mean(axis=1, keepdims=True)
        all_events, filtered = sw_base._detect_with_custom_method(
            epochs_copy=epochs,
            sf=float(epochs.info["sfreq"]),
            ch_names=list(epochs.ch_names),
            chan2idx={ch: i for i, ch in enumerate(epochs.ch_names)},
            freq_sw=(0.5, 4.0),
            amp_ptp_initial=1.0,
            filter_design="chebyshev2",
        )
        assert isinstance(filtered, dict)
        assert all(
            isinstance(v, np.ndarray) and v.ndim == 2
            for v in filtered.values()
        )
        if all_events:
            df = all_events[0]
            for col in ("MidCrossing", "AscendingSlope", "DescendingSlope"):
                assert col in df.columns

    def test_filter_design_fir_changes_filtered_signal(
        self, synthetic_sleep_epochs
    ):
        """FIR vs Chebyshev II should produce numerically different filtered signals."""
        sw_base = SlowWavesDetectionBase()
        sample = synthetic_sleep_epochs.get_data()[0] * 1e6
        sf = float(synthetic_sleep_epochs.info["sfreq"])

        cheby = sw_base._bandpass_uv(sample, sf, (0.5, 4.0), "chebyshev2")
        fir = sw_base._bandpass_uv(sample, sf, (0.5, 4.0), "fir")
        assert cheby.shape == fir.shape == sample.shape
        # Different filter designs must not produce bit-identical outputs.
        assert not np.allclose(cheby, fir)

    def test_invalid_filter_design_raises(self):
        with pytest.raises(ValueError, match="filter_design"):
            SlowWavesDetection(feature="Density", filter_design="butter")

    def test_invalid_slope_range_raises(self):
        with pytest.raises(ValueError, match="slope_uv_per_s_range"):
            SlowWavesDetection(
                feature="Density", slope_uv_per_s_range=(2.0, 1.0)
            )


class TestSlowWavesDetectionSubjectIdExtraction:
    """Subject id is read from Junifer's canonical input['meta']['element']."""

    def test_subject_id_read_from_input_meta_element(self):
        """Regression: real datagrabbers populate input['meta']['element']."""
        sid = SlowWavesDetection._extract_subject_id(
            input={"data": object(), "meta": {"element": {"subject": "03"}}},
            extra_input={},
        )
        assert sid == "03"

    def test_subject_id_extra_input_fallback_still_works(self):
        """Legacy/test code paths that pass element via extra_input."""
        sid = SlowWavesDetection._extract_subject_id(
            input={"data": object()},
            extra_input={"element": {"subject": "07"}},
        )
        assert sid == "07"

    def test_subject_id_input_meta_takes_precedence_over_extra_input(self):
        sid = SlowWavesDetection._extract_subject_id(
            input={"meta": {"element": {"subject": "from_meta"}}},
            extra_input={"element": {"subject": "from_extra"}},
        )
        assert sid == "from_meta"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
