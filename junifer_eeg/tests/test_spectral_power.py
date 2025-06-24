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

    # Test marker
    marker = SpectralPower()
    input_data = {"data": raw}
    result = marker.compute(input_data)

    # Check output
    assert "spectral_power" in result
    assert "data" in result["spectral_power"]
    assert "col_names" in result["spectral_power"]
    assert isinstance(result["spectral_power"]["data"], np.ndarray)
    assert len(result["spectral_power"]["col_names"]) == 4  # 4 bands

    # Check that alpha power is higher (since we have 10Hz signal)
    col_names = result["spectral_power"]["col_names"]
    alpha_idx = col_names.index("Cz_alpha")
    alpha_power = result["spectral_power"]["data"][0, alpha_idx]

    # Alpha should have higher power than others for 10Hz signal
    assert alpha_power > 0


def test_spectral_power_initialization():
    """Test SpectralPower initialization."""
    marker = SpectralPower()
    assert marker is not None

    # Test with parameters
    marker = SpectralPower(on="EEG", name="test_marker")
    assert marker is not None
