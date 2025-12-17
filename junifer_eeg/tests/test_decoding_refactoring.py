"""Test script to validate Decoding markers refactoring."""

import numpy as np
from mne import create_info
from mne.epochs import EpochsArray

from junifer_eeg.markers.decoding_new.time_decoding import (
    TimeDecoding as RefactoredTimeDecoding,
)
from junifer_eeg.markers.decoding_new.window_decoding import (
    WindowDecoding as RefactoredWindowDecoding,
)

# Import both implementations
from junifer_eeg.markers.time_decoding import (
    TimeDecoding as OriginalTimeDecoding,
)
from junifer_eeg.markers.window_decoding import (
    WindowDecoding as OriginalWindowDecoding,
)


def create_test_epochs():
    """Create synthetic EEG epochs data for testing."""
    _ = 40
    n_channels = 16
    n_times = 100
    sfreq = 250.0

    # Create synthetic EEG data with distinguishable patterns for conditions A and B
    np.random.seed(42)

    # Create data for condition A (20 epochs)
    data_a = np.random.randn(20, n_channels, n_times) * 0.5
    # Add structured signal for condition A
    t = np.linspace(0, n_times / sfreq, n_times)
    for i in range(n_channels):
        freq = 8 + i * 2  # Different frequencies per channel
        data_a[:, i, :] += 0.3 * np.sin(2 * np.pi * freq * t)

    # Create data for condition B (20 epochs) - different pattern
    data_b = np.random.randn(20, n_channels, n_times) * 0.5
    # Add different structured signal for condition B
    for i in range(n_channels):
        freq = 12 + i * 2  # Higher frequencies for condition B
        data_b[:, i, :] += 0.4 * np.sin(2 * np.pi * freq * t)

    # Combine data
    data = np.vstack([data_a, data_b])

    # Create MNE info
    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create epochs with events
    events_a = np.array([[i * 1000, 0, 1] for i in range(20)])  # Condition A
    events_b = np.array(
        [[i * 1000, 0, 2] for i in range(20, 40)]
    )  # Condition B
    events = np.vstack([events_a, events_b])

    event_id = {"condition_a": 1, "condition_b": 2}
    epochs = EpochsArray(
        data, info, events=events, event_id=event_id, tmin=0.0
    )

    return epochs


def test_time_decoding_equivalence():
    """Test that refactored TimeDecoding produces identical results."""
    print("Testing TimeDecoding equivalence...")
    epochs = create_test_epochs()

    # Test parameters
    params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.0,
        "tmax": 0.3,
        "n_splits": 5,
        "scoring": "roc_auc",
        "random_state": 42,
    }

    original_marker = OriginalTimeDecoding(**params)
    refactored_marker = RefactoredTimeDecoding(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    # Compare time series results
    orig_scores = orig_result["timedecoding"]["data"]
    refact_scores = refact_result["timedecoding"]["data"]

    print(f"Original scores shape: {orig_scores.shape}")
    print(f"Refactored scores shape: {refact_scores.shape}")
    print(
        f"Original scores range: [{np.min(orig_scores):.6f}, {np.max(orig_scores):.6f}]"
    )
    print(
        f"Refactored scores range: [{np.min(refact_scores):.6f}, {np.max(refact_scores):.6f}]"
    )

    # Check numerical equivalence
    abs_diff = np.abs(orig_scores - refact_scores)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"Max absolute difference: {max_abs_diff:.10f}")
    print(f"Mean absolute difference: {mean_abs_diff:.10f}")

    # Check if results are identical (within floating point precision)
    tolerance = 1e-10
    assert max_abs_diff < tolerance, (
        f"TimeDecoding: MISMATCH (tolerance {tolerance})"
    )
    print("✅ TimeDecoding: PERFECT MATCH")


def test_window_decoding_equivalence():
    """Test that refactored WindowDecoding produces identical results."""
    print("\nTesting WindowDecoding equivalence...")
    epochs = create_test_epochs()

    # Test parameters
    params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.1,
        "tmax": 0.3,
        "n_splits": 5,
        "scoring": "roc_auc",
        "random_state": 42,
    }

    original_marker = OriginalWindowDecoding(**params)
    refactored_marker = RefactoredWindowDecoding(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    # Compare scalar results
    orig_score = orig_result["windowdecoding"]["data"]
    refact_score = refact_result["windowdecoding"]["data"]

    # Handle both scalar and array formats - extract scalar values
    if isinstance(orig_score, np.ndarray) and orig_score.ndim > 0:
        orig_score = orig_score.item() if orig_score.size == 1 else orig_score
    if isinstance(refact_score, np.ndarray) and refact_score.ndim > 0:
        refact_score = (
            refact_score.item() if refact_score.size == 1 else refact_score
        )

    # Ensure we have scalar values
    orig_scalar = (
        float(orig_score)
        if not isinstance(orig_score, np.ndarray)
        else orig_score.item()
    )
    refact_scalar = (
        float(refact_score)
        if not isinstance(refact_score, np.ndarray)
        else refact_score.item()
    )

    print(f"Original score: {orig_scalar:.6f}")
    print(f"Refactored score: {refact_scalar:.6f}")

    # Check numerical equivalence
    abs_diff = abs(orig_scalar - refact_scalar)

    print(f"Absolute difference: {abs_diff:.10f}")
    print(f"Relative difference: {(abs_diff / orig_scalar) * 100:.6f}%")

    # Check if results are identical (within floating point precision)
    tolerance = 1e-10
    assert abs_diff < tolerance, (
        f"WindowDecoding: MISMATCH (tolerance {tolerance})"
    )
    print("✅ WindowDecoding: PERFECT MATCH")


def test_time_decoding_roi():
    """Test that TimeDecoding ROI filtering works correctly."""
    print("\nTesting TimeDecoding ROI filtering...")
    epochs = create_test_epochs()

    # Test with ROI filtering
    params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.0,
        "tmax": 0.3,
        "rois": ["E1", "E2", "E3", "E4"],  # First 4 channels
        "n_splits": 5,
        "scoring": "roc_auc",
        "random_state": 42,
    }

    original_marker = OriginalTimeDecoding(**params)
    refactored_marker = RefactoredTimeDecoding(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_scores = orig_result["timedecoding"]["data"]
    refact_scores = refact_result["timedecoding"]["data"]

    max_diff = np.max(np.abs(orig_scores - refact_scores))
    assert max_diff < 1e-10, (
        f"TimeDecoding ROI filtering failed - Max diff: {max_diff:.10f}"
    )
    print("✅ TimeDecoding ROI filtering works correctly!")


def test_window_decoding_roi():
    """Test that WindowDecoding ROI filtering works correctly."""
    print("\nTesting WindowDecoding ROI filtering...")
    epochs = create_test_epochs()

    # Test with ROI filtering
    params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": ["E1", "E2", "E3", "E4"],  # First 4 channels
        "n_splits": 5,
        "scoring": "roc_auc",
        "random_state": 42,
    }

    original_marker = OriginalWindowDecoding(**params)
    refactored_marker = RefactoredWindowDecoding(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_score = orig_result["windowdecoding"]["data"][0]
    refact_score = refact_result["windowdecoding"]["data"][0]

    diff = abs(orig_score - refact_score)
    assert diff < 1e-10, (
        f"WindowDecoding ROI filtering failed - Diff: {diff:.10f}"
    )
    print("✅ WindowDecoding ROI filtering works correctly!")


def test_missing_conditions():
    """Test that markers handle missing conditions correctly."""
    print("\nTesting missing conditions handling...")
    epochs = create_test_epochs()

    # Test with non-existent conditions
    params = {
        "condition_a": "non_existent_a",
        "condition_b": "non_existent_b",
        "tmin": 0.0,
        "tmax": 0.3,
        "n_splits": 5,
        "scoring": "roc_auc",
        "random_state": 42,
    }

    original_time = OriginalTimeDecoding(**params)
    refactored_time = RefactoredTimeDecoding(**params)

    original_window = OriginalWindowDecoding(**params)
    refactored_window = RefactoredWindowDecoding(**params)

    orig_time_result = original_time.compute({"data": epochs})
    refact_time_result = refactored_time.compute({"data": epochs})

    orig_window_result = original_window.compute({"data": epochs})
    refact_window_result = refactored_window.compute({"data": epochs})

    # Check TimeDecoding chance level output
    orig_time_scores = orig_time_result["timedecoding"]["data"]
    refact_time_scores = refact_time_result["timedecoding"]["data"]

    if np.allclose(orig_time_scores, refact_time_scores, atol=1e-10):
        print("✅ TimeDecoding missing conditions handling works correctly!")
        time_success = True
    else:
        print("❌ TimeDecoding missing conditions handling failed")
        time_success = False

    # Check WindowDecoding chance level output
    orig_window_score = orig_window_result["windowdecoding"]["data"]
    refact_window_score = refact_window_result["windowdecoding"]["data"]

    # Handle both scalar and array formats for comparison
    if (
        isinstance(orig_window_score, np.ndarray)
        and orig_window_score.size == 1
    ):
        orig_window_score = orig_window_score.item()
    if (
        isinstance(refact_window_score, np.ndarray)
        and refact_window_score.size == 1
    ):
        refact_window_score = refact_window_score.item()

    if abs(orig_window_score - refact_window_score) < 1e-10:
        print("✅ WindowDecoding missing conditions handling works correctly!")
        window_success = True
    else:
        print("❌ WindowDecoding missing conditions handling failed")
        window_success = False

    return time_success and window_success


if __name__ == "__main__":
    print("=" * 70)
    print("DECODING MARKERS REFACTORING VALIDATION")
    print("=" * 70)

    success = True

    # Run all tests
    success &= test_time_decoding_equivalence()
    success &= test_window_decoding_equivalence()
    success &= test_time_decoding_roi()
    success &= test_window_decoding_roi()
    success &= test_missing_conditions()

    print("\n" + "=" * 70)
    if success:
        print(
            "🎉 ALL TESTS PASSED - Decoding markers refactoring is successful!"
        )
    else:
        print(
            "❌ SOME TESTS FAILED - Decoding markers refactoring needs fixes!"
        )
    print("=" * 70)
