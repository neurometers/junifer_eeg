"""Tests for EEGLoader preprocessor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from junifer_eeg.preprocessors import EEGLoader


def test_eeg_loader_initialization():
    """Test EEGLoader initialization."""
    loader = EEGLoader()
    assert loader is not None

    # Test with parameters
    loader = EEGLoader(on="BOLD")
    assert loader is not None


def test_eeg_loader_methods():
    """Test EEGLoader methods."""
    loader = EEGLoader()

    # Test valid inputs
    assert loader.get_valid_inputs() == ["BOLD"]
    assert loader.get_output_type("BOLD") == "BOLD"


@patch("mne.io.read_raw_edf")
def test_eeg_loader_preprocess(mock_read_edf):
    """Test EEGLoader preprocessing with mocked MNE."""
    # Setup mock
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 100}
    mock_raw.ch_names = ["CH1", "CH2"]
    mock_read_edf.return_value = mock_raw

    # Test the loader
    loader = EEGLoader()

    # Test preprocessing
    input_data = {"path": "/fake/path/test.edf"}
    processed_input, extra_output = loader.preprocess(input_data)

    # Check that read_raw_edf was called correctly
    mock_read_edf.assert_called_once_with(
        Path("/fake/path/test.edf"), preload=True, verbose=False
    )

    # Check that raw_object was added
    assert "raw_object" in processed_input
    assert processed_input["raw_object"] == mock_raw
    assert extra_output is None
