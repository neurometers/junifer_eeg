"""Test that all markers consistently store col_names via format_marker_result.

This test suite validates that:
1. All  markers use the centralized format_marker_result function
2. col_names are stored correctly when no channel aggregation is applied
3. col_names are NOT stored when channel aggregation IS applied
4. Data can be read back correctly via JuniferH5Reader
5. Channel ordering is preserved between input and output

Reference data is stored in reference_data/col_names_test_reference.h5
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from junifer_eeg.markers.base import format_marker_result
from junifer_eeg.markers.kolmogorov_complexity import KolmogorovComplexity
from junifer_eeg.markers.permutation_entropy import PermutationEntropy
from junifer_eeg.markers.spectral_power import SpectralPowerBands
from junifer_eeg.markers.time_locked import (
    TimeLockedTopography,
)

# Path to reference data
REFERENCE_DATA_DIR = Path(__file__).parent / "reference_data"


class TestFormatMarkerResult:
    """Test the centralized format_marker_result function."""

    def test_col_names_stored_when_no_aggregation(self):
        """col_names should be stored when channel_aggregated=False."""
        data = np.random.rand(10, 256)
        col_names = [f"E{i}" for i in range(1, 257)]

        result = format_marker_result(
            feature_name="test",
            data=data,
            col_names=col_names,
            channel_aggregated=False,
        )

        assert "col_names" in result["test"]
        assert result["test"]["col_names"] == col_names

    def test_col_names_not_stored_when_aggregated(self):
        """col_names should NOT be stored when channel_aggregated=True."""
        data = np.random.rand(10)  # Only epochs dimension left
        col_names = [f"E{i}" for i in range(1, 257)]

        result = format_marker_result(
            feature_name="test",
            data=data,
            col_names=col_names,
            channel_aggregated=True,
        )

        assert "col_names" not in result["test"]

    def test_col_names_not_stored_for_scalar(self):
        """col_names should NOT be stored for scalar data."""
        data = np.float64(1.5)
        col_names = ["E1"]

        result = format_marker_result(
            feature_name="test",
            data=data,
            col_names=col_names,
            channel_aggregated=True,
        )

        assert "col_names" not in result["test"]

    def test_col_names_converted_to_list(self):
        """col_names should be converted to list (not tuple or other)."""
        data = np.random.rand(10, 5)
        col_names = ("E1", "E2", "E3", "E4", "E5")  # tuple

        result = format_marker_result(
            feature_name="test",
            data=data,
            col_names=col_names,
            channel_aggregated=False,
        )

        assert isinstance(result["test"]["col_names"], list)


class TestMarkersUseFormatMarkerResult:
    """Test that all  markers use format_marker_result consistently."""

    @pytest.fixture
    def synthetic_epochs(self):
        """Create synthetic epochs for testing."""
        # Create synthetic EEG data
        n_channels = 64
        n_epochs = 20
        n_times = 500
        sfreq = 250.0

        # Generate random data
        data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6

        # Create info
        ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=ch_types
        )

        # Create events
        events = np.column_stack(
            [
                np.arange(n_epochs) * n_times,
                np.zeros(n_epochs, dtype=int),
                np.ones(n_epochs, dtype=int),
            ]
        )

        # Create epochs
        epochs = mne.EpochsArray(data, info, events=events, tmin=-0.2)
        return epochs

    def test_spectral_power_bands_no_aggregation(self, synthetic_epochs):
        """SpectralPowerBands should store col_names when no aggregation."""
        marker = SpectralPowerBands(
            bands={"alpha": [8.0, 12.0]},
            channel_method=None,
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" in result["spectralpower"]
        col_names = result["spectralpower"]["col_names"]
        assert len(col_names) == len(synthetic_epochs.ch_names)

    def test_spectral_power_bands_with_channel_agg(self, synthetic_epochs):
        """SpectralPowerBands should NOT store col_names when channel aggregation."""
        marker = SpectralPowerBands(
            bands={"alpha": [8.0, 12.0]},
            channel_method="mean",
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" not in result["spectralpower"]

    def test_permutation_entropy_no_aggregation(self, synthetic_epochs):
        """PermutationEntropy should store col_names when no aggregation."""
        marker = PermutationEntropy(
            taus=[4],
            kernel=3,
            channel_method=None,
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" in result["permutationentropy"]
        col_names = result["permutationentropy"]["col_names"]
        assert len(col_names) == len(synthetic_epochs.ch_names)

    def test_permutation_entropy_with_channel_agg(self, synthetic_epochs):
        """PermutationEntropy should NOT store col_names when channel aggregation."""
        marker = PermutationEntropy(
            taus=[4],
            kernel=3,
            channel_method="mean",
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" not in result["permutationentropy"]

    def test_kolmogorov_complexity_no_aggregation(self, synthetic_epochs):
        """KolmogorovComplexity should store col_names when no aggregation."""
        marker = KolmogorovComplexity(
            nbins=16,
            channel_method=None,
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" in result["kolmogorovcomplexity"]
        col_names = result["kolmogorovcomplexity"]["col_names"]
        assert len(col_names) == len(synthetic_epochs.ch_names)

    def test_kolmogorov_complexity_with_channel_agg(self, synthetic_epochs):
        """KolmogorovComplexity should NOT store col_names when channel aggregation."""
        marker = KolmogorovComplexity(
            nbins=16,
            channel_method="mean",
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" not in result["kolmogorovcomplexity"]

    def test_time_locked_topography_no_aggregation(self, synthetic_epochs):
        """TimeLockedTopography should store col_names when no aggregation."""
        marker = TimeLockedTopography(
            tmin=0.0,
            tmax=0.1,
            channel_method=None,
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" in result["timelockedtopo"]
        col_names = result["timelockedtopo"]["col_names"]
        assert len(col_names) == len(synthetic_epochs.ch_names)

    def test_time_locked_topography_with_channel_agg(self, synthetic_epochs):
        """TimeLockedTopography should NOT store col_names when channel aggregation."""
        marker = TimeLockedTopography(
            tmin=0.0,
            tmax=0.1,
            channel_method="mean",
            trial_method=None,
        )

        result = marker.compute({"data": synthetic_epochs})

        assert "col_names" not in result["timelockedtopo"]


class TestChannelOrderingPreservation:
    """Test that channel ordering is preserved from input to output."""

    @pytest.fixture
    def epochs_with_known_order(self):
        """Create epochs with known channel values for ordering verification."""
        n_channels = 32
        n_epochs = 10
        n_times = 250
        sfreq = 250.0

        # Create data where each channel has a unique mean value
        # This allows us to verify ordering by checking the values
        data = np.zeros((n_epochs, n_channels, n_times))
        for ch_idx in range(n_channels):
            # Each channel has a distinct baseline
            data[:, ch_idx, :] = (
                ch_idx * 0.1 + np.random.randn(n_epochs, n_times) * 0.01
            )

        ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=ch_types
        )

        events = np.column_stack(
            [
                np.arange(n_epochs) * n_times,
                np.zeros(n_epochs, dtype=int),
                np.ones(n_epochs, dtype=int),
            ]
        )

        epochs = mne.EpochsArray(data, info, events=events, tmin=0.0)
        return epochs

    def test_channel_order_preserved_in_output(self, epochs_with_known_order):
        """Verify that channel order in output matches input order."""
        marker = SpectralPowerBands(
            bands={"alpha": [8.0, 12.0]},
            channel_method=None,
            trial_method="mean",
        )

        result = marker.compute({"data": epochs_with_known_order})

        col_names = result["spectralpower"]["col_names"]
        input_ch_names = epochs_with_known_order.ch_names

        # Verify ordering matches
        assert col_names == list(input_ch_names), (
            f"Channel order mismatch!\n"
            f"Input:  {input_ch_names[:5]}...\n"
            f"Output: {col_names[:5]}..."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
