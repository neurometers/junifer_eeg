"""Tests for new EEG markers."""

import mne
import numpy as np

from junifer_eeg.markers import (
    ContingentNegativeVariation,
    KolmogorovComplexity,
    PermutationEntropy,
    PowerSpectralDensityEstimator,
    PowerSpectralDensitySummary,
)


def create_test_raw():
    """Create synthetic Raw object for testing."""
    sfreq = 100  # 100 Hz sampling rate
    duration = 2  # 2 seconds
    times = np.arange(0, duration, 1 / sfreq)

    # Create different frequency components
    alpha_signal = np.sin(2 * np.pi * 10 * times)  # 10 Hz alpha
    beta_signal = np.sin(2 * np.pi * 20 * times)  # 20 Hz beta
    noise = np.random.RandomState(42).normal(0, 0.1, len(times))

    # Combine signals
    data = alpha_signal + 0.5 * beta_signal + noise

    # Create MNE Raw object
    info = mne.create_info(
        ch_names=["Cz", "Fz"], sfreq=sfreq, ch_types=["eeg"] * 2
    )
    raw_data = np.tile(data, (2, 1))  # 2 channels
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    return raw


class TestKolmogorovComplexity:
    """Test KolmogorovComplexity marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = KolmogorovComplexity(tmin=0.5, tmax=1.5, nbins=16, on="EEG")
        assert marker.tmin == 0.5
        assert marker.tmax == 1.5
        assert marker.nbins == 16

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = KolmogorovComplexity(on="EEG")

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "kolmogorov_complexity" in result
        data = result["kolmogorov_complexity"]["data"]
        col_names = result["kolmogorov_complexity"]["col_names"]

        assert data.shape == (1, 2)  # 1 observation, 2 channels
        assert len(col_names) == 2
        assert all("kolmogorov" in name for name in col_names)
        assert np.all(np.isfinite(data))
        assert np.all(data > 0)  # Complexity should be positive


class TestPermutationEntropy:
    """Test PermutationEntropy marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = PermutationEntropy(
            tmin=0.5, tmax=1.5, kernel=4, tau=10, on="EEG"
        )
        assert marker.tmin == 0.5
        assert marker.tmax == 1.5
        assert marker.kernel == 4
        assert marker.tau == 10

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = PermutationEntropy(kernel=3, tau=5, on="EEG")

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "permutation_entropy" in result
        data = result["permutation_entropy"]["data"]
        col_names = result["permutation_entropy"]["col_names"]

        assert data.shape == (1, 2)  # 1 observation, 2 channels
        assert len(col_names) == 2
        assert all("permutation_entropy" in name for name in col_names)
        assert np.all(np.isfinite(data))
        assert np.all((data >= 0) & (data <= 1))  # PE should be normalized


class TestContingentNegativeVariation:
    """Test ContingentNegativeVariation marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = ContingentNegativeVariation(tmin=0.5, tmax=1.5, on="EEG")
        assert marker.tmin == 0.5
        assert marker.tmax == 1.5

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = ContingentNegativeVariation(on="EEG")

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "cnv_slope" in result
        assert "cnv_intercept" in result

        slope_data = result["cnv_slope"]["data"]
        intercept_data = result["cnv_intercept"]["data"]
        slope_names = result["cnv_slope"]["col_names"]
        intercept_names = result["cnv_intercept"]["col_names"]

        assert slope_data.shape == (1, 2)  # 1 observation, 2 channels
        assert intercept_data.shape == (1, 2)
        assert len(slope_names) == 2
        assert len(intercept_names) == 2
        assert all("cnv_slope" in name for name in slope_names)
        assert all("cnv_intercept" in name for name in intercept_names)
        assert np.all(np.isfinite(slope_data))
        assert np.all(np.isfinite(intercept_data))


class TestPowerSpectralDensityEstimator:
    """Test PowerSpectralDensityEstimator marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = PowerSpectralDensityEstimator(
            tmin=0.5, tmax=1.5, fmin=1, fmax=30, on="EEG"
        )
        assert marker.tmin == 0.5
        assert marker.tmax == 1.5
        assert marker.fmin == 1
        assert marker.fmax == 30

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = PowerSpectralDensityEstimator(fmin=1, fmax=30, on="EEG")

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "psd_data" in result
        assert "psd_freqs" in result
        assert "psd_data_norm" in result

        psd_data = result["psd_data"]["data"]
        psd_freqs = result["psd_freqs"]["data"]
        psd_norm = result["psd_data_norm"]["data"]

        assert psd_data.shape[0] == 2  # 2 channels
        assert psd_freqs.shape[0] == 1  # 1 observation
        assert psd_data.shape == psd_norm.shape
        assert psd_data.shape[1] == psd_freqs.shape[1]  # Same frequency bins
        assert np.all(np.isfinite(psd_data))
        assert np.all(psd_data >= 0)  # PSD should be non-negative


class TestPowerSpectralDensitySummary:
    """Test PowerSpectralDensitySummary marker."""

    def test_initialization(self):
        """Test marker initialization."""
        marker = PowerSpectralDensitySummary(
            percentile=75, fmin=1, fmax=30, on="EEG"
        )
        assert marker.percentile == 75
        assert marker.fmin == 1
        assert marker.fmax == 30

    def test_compute(self):
        """Test marker computation."""
        raw = create_test_raw()
        marker = PowerSpectralDensitySummary(
            percentile=50, fmin=1, fmax=30, on="EEG"
        )

        input_data = {"data": raw}
        result = marker.compute(input_data)

        assert "psd_summary" in result
        data = result["psd_summary"]["data"]
        col_names = result["psd_summary"]["col_names"]

        assert data.shape == (1, 2)  # 1 observation, 2 channels
        assert len(col_names) == 2
        assert all("psd_p50" in name for name in col_names)
        assert np.all(np.isfinite(data))
        assert np.all(data >= 0)  # PSD summary should be non-negative
