"""Simple tests for SpectralPower marker."""

import numpy as np

from junifer_eeg.markers import SpectralPower


def test_spectral_power_basic():
    """Test basic SpectralPower functionality."""
    import mne

    # Create synthetic data
    sfreq = 100
    times = np.arange(0, 2, 1 / sfreq)  # 2 seconds
    data = np.sin(2 * np.pi * 10 * times)  # 10 Hz signal

    # Create MNE Raw object
    info = mne.create_info(ch_names=["Cz"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(data[np.newaxis, :], info)

    # Test marker with explicit bands parameter to ensure all 5 bands
    # Use aggregation methods to avoid the raw data return path
    marker = SpectralPower(
        bands={
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 45),
        },
        roi_aggregation_method=["mean"],
        trial_aggregation_method=["mean"],
    )
    input_data = {"data": raw}
    result = marker.compute(input_data)

    # Check output
    assert "spectralpower" in result
    assert "data" in result["spectralpower"]
    assert "col_names" in result["spectralpower"]
    assert isinstance(result["spectralpower"]["data"], np.ndarray)
    assert len(result["spectralpower"]["col_names"]) == 5  # 5 bands

    # Check that alpha power is higher (since we have 10Hz signal)
    col_names = result["spectralpower"]["col_names"]
    data = result["spectralpower"]["data"]

    # Find alpha and other band indices
    alpha_idx = next(i for i, name in enumerate(col_names) if "alpha" in name)
    delta_idx = next(i for i, name in enumerate(col_names) if "delta" in name)

    alpha_power = data[0, alpha_idx]
    delta_power = data[0, delta_idx]

    # Alpha should have higher power than delta for 10Hz signal (even in dB)
    # Since we're using dB conversion, values can be negative, but alpha should still be higher
    assert alpha_power > delta_power


def test_spectral_power_initialization():
    """Test SpectralPower initialization."""
    marker = SpectralPower()
    assert marker is not None

    # Test with parameters
    marker = SpectralPower(on="EEG", name="test_marker")
    assert marker is not None
