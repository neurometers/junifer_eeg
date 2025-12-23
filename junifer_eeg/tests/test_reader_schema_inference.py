"""Tests for reader schema inference correctness.

This module tests that the reader correctly identifies dimensions for all
marker types with different aggregation configurations.
"""

import numpy as np
import pytest

from junifer_eeg.reader import (
    DimensionType,
    MarkerRegistry,
    MarkerType,
    infer_schema,
)


class TestMarkerTypeDetection:
    """Test marker type detection from names."""

    def test_spectral_detection(self):
        """Test detection of spectral markers."""
        registry = MarkerRegistry()
        assert registry.detect_marker_type("psd_alpha") == MarkerType.SPECTRAL
        assert (
            registry.detect_marker_type("spectral_power_db")
            == MarkerType.SPECTRAL
        )
        assert (
            registry.detect_marker_type("SpectralPowerBands")
            == MarkerType.SPECTRAL
        )

    def test_connectivity_detection(self):
        """Test detection of connectivity markers."""
        registry = MarkerRegistry()
        assert (
            registry.detect_marker_type("wsmi_multiscale")
            == MarkerType.CONNECTIVITY
        )
        assert (
            registry.detect_marker_type("smi_theta") == MarkerType.CONNECTIVITY
        )
        assert (
            registry.detect_marker_type("SymbolicMutualInformation")
            == MarkerType.CONNECTIVITY
        )

    def test_entropy_detection(self):
        """Test detection of entropy markers."""
        registry = MarkerRegistry()
        assert registry.detect_marker_type("pe_alpha") == MarkerType.ENTROPY
        assert (
            registry.detect_marker_type("permutation_entropy")
            == MarkerType.ENTROPY
        )
        assert (
            registry.detect_marker_type("PermutationEntropy")
            == MarkerType.ENTROPY
        )

    def test_complexity_detection(self):
        """Test detection of complexity markers."""
        registry = MarkerRegistry()
        assert (
            registry.detect_marker_type("kolmogorov_complexity")
            == MarkerType.COMPLEXITY
        )
        assert (
            registry.detect_marker_type("KolmogorovComplexity")
            == MarkerType.COMPLEXITY
        )

    def test_sleep_detection(self):
        """Test detection of sleep markers."""
        registry = MarkerRegistry()
        assert (
            registry.detect_marker_type("spindles_detection")
            == MarkerType.SLEEP
        )
        assert registry.detect_marker_type("slow_waves") == MarkerType.SLEEP
        assert (
            registry.detect_marker_type("SlowWavesDetection")
            == MarkerType.SLEEP
        )


class TestSpectralSchemaInference:
    """Test schema inference for spectral markers."""

    def test_2d_epochs_channels(self):
        """Test 2D spectral data (epochs, channels)."""
        data = np.random.randn(100, 64)  # 100 epochs, 64 channels
        schema = infer_schema("psd_alpha", data)

        assert len(schema.dimensions) == 2
        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[0].size == 100
        assert schema.dimensions[1].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[1].size == 64

    def test_3d_bands_epochs_channels(self):
        """Test 3D spectral data (bands, epochs, channels)."""
        data = np.random.randn(5, 100, 64)  # 5 bands, 100 epochs, 64 channels
        schema = infer_schema("spectral_power_bands", data)

        assert len(schema.dimensions) == 3
        assert schema.dimensions[0].dim_type == DimensionType.BANDS
        assert schema.dimensions[0].size == 5
        assert schema.dimensions[0].labels == [
            "delta",
            "theta",
            "alpha",
            "beta",
            "gamma",
        ]
        assert schema.dimensions[1].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[1].size == 100
        assert schema.dimensions[2].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[2].size == 64

    def test_1d_aggregated_channels(self):
        """Test 1D spectral data with channel names (aggregated epochs)."""
        data = np.random.randn(64)
        col_names = [f"E{i}" for i in range(64)]
        schema = infer_schema("psd_alpha", data, col_names=col_names)

        assert len(schema.dimensions) == 1
        assert schema.dimensions[0].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[0].size == 64
        assert schema.dimensions[0].labels == col_names

    def test_1d_aggregated_epochs(self):
        """Test 1D spectral data without channel names (aggregated channels)."""
        data = np.random.randn(100)
        schema = infer_schema("psd_alpha", data)

        assert len(schema.dimensions) == 1
        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[0].size == 100


class TestConnectivitySchemaInference:
    """Test schema inference for connectivity markers (WSMI, SMI)."""

    def test_2d_epochs_channel_pairs(self):
        """Test 2D connectivity (epochs, channel_pairs) - single tau."""
        data = np.random.randn(100, 2080)  # 100 epochs, 64x65/2 pairs
        schema = infer_schema("smi_theta", data)

        assert len(schema.dimensions) == 2
        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[0].size == 100
        assert schema.dimensions[1].dim_type == DimensionType.CHANNEL_PAIRS
        assert schema.dimensions[1].size == 2080

    def test_3d_epochs_channels_channels(self):
        """Test 3D connectivity (epochs, ch_i, ch_j) - square matrix."""
        data = np.random.randn(100, 64, 64)  # 100 epochs, 64x64 matrix
        schema = infer_schema("wsmi_matrix", data)

        assert len(schema.dimensions) == 3
        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[0].size == 100
        assert schema.dimensions[1].dim_type == DimensionType.CHANNELS_I
        assert schema.dimensions[1].size == 64
        assert schema.dimensions[2].dim_type == DimensionType.CHANNELS_J
        assert schema.dimensions[2].size == 64


class TestEntropyComplexitySchemaInference:
    """Test schema inference for entropy and complexity markers."""

    def test_2d_epochs_channels(self):
        """Test 2D entropy/complexity data (epochs, channels)."""
        data = np.random.randn(100, 64)

        # Test entropy
        schema = infer_schema("permutation_entropy", data)
        assert len(schema.dimensions) == 2
        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[1].dim_type == DimensionType.CHANNELS

        # Test complexity
        schema = infer_schema("kolmogorov_complexity", data)
        assert len(schema.dimensions) == 2
        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[1].dim_type == DimensionType.CHANNELS

    def test_1d_aggregated(self):
        """Test 1D aggregated entropy/complexity."""
        data = np.random.randn(64)
        col_names = [f"E{i}" for i in range(64)]

        schema = infer_schema("pe_alpha", data, col_names=col_names)
        assert len(schema.dimensions) == 1
        assert schema.dimensions[0].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[0].labels == col_names


class TestSleepSchemaInference:
    """Test schema inference for sleep markers (spindles, slow waves)."""

    def test_3d_features_epochs_channels_spindles(self):
        """Test 3D sleep data for spindles (4 features)."""
        # Spindles: Duration, Amplitude, Frequency, Density
        data = np.random.randn(4, 100, 64)
        schema = infer_schema("spindles_detection", data)

        assert len(schema.dimensions) == 3
        assert schema.dimensions[0].dim_type == DimensionType.FEATURES
        assert schema.dimensions[0].size == 4
        assert schema.dimensions[0].labels == [
            "Duration",
            "Amplitude",
            "Frequency",
            "Density",
        ]
        assert schema.dimensions[1].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[1].size == 100
        assert schema.dimensions[2].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[2].size == 64

    def test_3d_features_epochs_channels_slow_waves(self):
        """Test 3D sleep data for slow waves (5 features)."""
        # Slow waves: Duration, PTP, Frequency, Slope, Density
        data = np.random.randn(5, 100, 64)
        schema = infer_schema("slow_waves_detection", data)

        assert len(schema.dimensions) == 3
        assert schema.dimensions[0].dim_type == DimensionType.FEATURES
        assert schema.dimensions[0].size == 5
        assert schema.dimensions[0].labels == [
            "Duration",
            "PTP",
            "Frequency",
            "Slope",
            "Density",
        ]
        assert schema.dimensions[1].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[1].size == 100
        assert schema.dimensions[2].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[2].size == 64


class TestSchemaDescriptions:
    """Test that schema descriptions are human-readable."""

    def test_spectral_description(self):
        """Test spectral marker description."""
        data = np.random.randn(5, 100, 64)
        schema = infer_schema("spectral_power", data)
        desc = schema.describe()

        assert "spectral_power" in desc
        assert "BANDS" in desc
        assert "EPOCHS" in desc
        assert "CHANNELS" in desc

    def test_connectivity_description(self):
        """Test connectivity marker description."""
        data = np.random.randn(4, 1193, 32640)
        schema = infer_schema("wsmi_multiscale", data)
        desc = schema.describe()

        assert "wsmi_multiscale" in desc
        assert "BANDS" in desc or "tau" in desc.lower()
        assert "EPOCHS" in desc
        assert "CHANNEL_PAIRS" in desc


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_connectivity_shape(self):
        """Test connectivity with small number of channels."""
        # 10 channels -> 10*9/2 = 45 pairs
        data = np.random.randn(100, 45)
        schema = infer_schema("wsmi_small", data)

        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[1].dim_type == DimensionType.CHANNEL_PAIRS

    def test_single_epoch(self):
        """Test single epoch data."""
        data = np.random.randn(1, 64)
        schema = infer_schema("psd_alpha", data)

        assert schema.dimensions[0].dim_type == DimensionType.EPOCHS
        assert schema.dimensions[0].size == 1

    def test_single_channel(self):
        """Test single channel data."""
        data = np.random.randn(100, 1)
        schema = infer_schema("psd_alpha", data)

        assert schema.dimensions[1].dim_type == DimensionType.CHANNELS
        assert schema.dimensions[1].size == 1

    def test_unknown_marker_type(self):
        """Test unknown marker type falls back gracefully."""
        data = np.random.randn(100, 64)
        schema = infer_schema("unknown_marker_xyz", data)

        # Should still create valid dimensions
        assert len(schema.dimensions) == 2
        assert schema.marker_type == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
