"""Tests for EEGFilter preprocessor."""

from unittest.mock import MagicMock

from junifer_eeg.preprocessors import EEGFilter


def test_eeg_filter_initialization():
    """Test EEGFilter initialization."""
    filter_obj = EEGFilter()
    assert filter_obj is not None

    # Test with parameters
    filter_obj = EEGFilter(low_freq=1.0, high_freq=40.0, on="EEG")
    assert filter_obj is not None
    assert filter_obj.low_freq == 1.0
    assert filter_obj.high_freq == 40.0


def test_eeg_filter_methods():
    """Test EEGFilter methods."""
    filter_obj = EEGFilter()

    # Test valid inputs
    assert filter_obj.get_valid_inputs() == ["EEG"]
    assert filter_obj.get_output_type("EEG") == "EEG"


def test_eeg_filter_preprocess():
    """Test EEGFilter preprocessing with mocked MNE."""
    # Mock the MNE Raw object
    mock_raw = MagicMock()
    # Configure the filter method to return the same object
    mock_raw.filter.return_value = mock_raw

    # Create filter
    filter_obj = EEGFilter(low_freq=1.0, high_freq=40.0)

    # Test input
    input_data = {"data": mock_raw}

    # Run preprocessing
    result, extra = filter_obj.preprocess(input_data)

    # Check that MNE filter was called
    mock_raw.filter.assert_called_once_with(
        l_freq=1.0,
        h_freq=40.0,
        verbose=False,
    )

    # Check output
    assert "data" in result
    assert result["data"] == mock_raw
    assert extra is None
