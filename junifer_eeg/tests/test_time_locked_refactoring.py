"""Comprehensive validation tests for time-locked markers refactoring.

This script validates that refactored TimeLockedTopography and TimeLockedContrast
markers produce numerically identical results to the original implementations.

Following the same rigorous validation approach as decoding markers refactoring.
"""

import numpy as np
from mne import create_info
from mne.epochs import EpochsArray

from junifer_eeg.markers.time_locked_contrast import (
    TimeLockedContrast as OriginalTimeLockedContrast,
)
from junifer_eeg.markers.time_locked_new.time_locked_contrast import (
    TimeLockedContrast as RefactoredTimeLockedContrast,
)

# Import refactored implementations
from junifer_eeg.markers.time_locked_new.time_locked_topography import (
    TimeLockedTopography as RefactoredTimeLockedTopography,
)

# Import original implementations
from junifer_eeg.markers.time_locked_topography import (
    TimeLockedTopography as OriginalTimeLockedTopography,
)


def create_synthetic_epochs():
    """Create synthetic EEG epochs for testing.

    Returns
    -------
    epochs : mne.EpochsArray
        Synthetic epochs with distinguishable patterns between conditions.
    """
    # Create synthetic data with clear temporal patterns
    n_epochs = 40
    n_channels = 16
    n_times = 100
    sfreq = 250.0

    # Generate synthetic EEG data
    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times) * 0.1

    # Add temporal patterns (early vs late responses)
    time_pattern = np.linspace(0, 1, n_times)
    for i in range(n_epochs):
        # Early response pattern for first half
        if i < n_epochs // 2:
            data[i, :, :30] += 0.5 * np.exp(-time_pattern[:30] * 5)
        # Late response pattern for second half
        else:
            data[i, :, 70:] += 0.5 * np.exp(-(time_pattern[70:] - 0.7) * 5)

    # Create channel names
    ch_names = [f"E{i + 1}" for i in range(n_channels)]

    # Create info
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create events with two conditions
    events = []
    for i in range(n_epochs):
        condition = 1 if i < n_epochs // 2 else 2
        events.append([i * 1000, 0, condition])
    events = np.array(events)

    # Create event_id
    event_id = {"condition_a": 1, "condition_b": 2}

    # Create epochs
    epochs = EpochsArray(
        data, info, events=events, event_id=event_id, tmin=0.0
    )

    return epochs


def test_time_locked_topography_equivalence():
    """Test that refactored TimeLockedTopography produces identical results."""
    print("Testing TimeLockedTopography equivalence...")

    # Create test data
    epochs = create_synthetic_epochs()

    # Test parameters
    params = {
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": ["scalp"],
        "channel_aggregation_method": "mean",
        "trial_aggregation_method": "mean",
        "equipment": "egi256",
    }

    # Create markers
    original_marker = OriginalTimeLockedTopography(**params)
    refactored_marker = RefactoredTimeLockedTopography(**params)

    # Compute results
    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    # Compare results
    orig_data = orig_result["timelockedtopo"]["data"]
    refact_data = refact_result["timelockedtopo"]["data"]

    print(f"Original result shape: {orig_data.shape}")
    print(f"Refactored result shape: {refact_data.shape}")
    print(
        f"Original result range: [{np.min(orig_data):.6f}, {np.max(orig_data):.6f}]"
    )
    print(
        f"Refactored result range: [{np.min(refact_data):.6f}, {np.max(refact_data):.6f}]"
    )

    # Check numerical equivalence
    if np.allclose(orig_data, refact_data, atol=1e-10):
        print("✅ TimeLockedTopography: PERFECT MATCH")
        return True
    else:
        max_diff = np.max(np.abs(orig_data - refact_data))
        print(f"❌ TimeLockedTopography: MISMATCH (max diff: {max_diff:.10f})")
        return False


def test_time_locked_contrast_equivalence():
    """Test that refactored TimeLockedContrast produces identical results."""
    print("Testing TimeLockedContrast equivalence...")

    # Create test data
    epochs = create_synthetic_epochs()

    # Test parameters
    params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": ["scalp"],
        "channel_aggregation_method": "mean",
        "trial_aggregation_method": "mean",
        "equipment": "egi256",
    }

    # Create markers
    original_marker = OriginalTimeLockedContrast(**params)
    refactored_marker = RefactoredTimeLockedContrast(**params)

    # Compute results
    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    # Compare results
    orig_data = orig_result["timelockedcontrast"]["data"]
    refact_data = refact_result["timelockedcontrast"]["data"]

    print(f"Original result shape: {orig_data.shape}")
    print(f"Refactored result shape: {refact_data.shape}")
    print(
        f"Original result range: [{np.min(orig_data):.6f}, {np.max(orig_data):.6f}]"
    )
    print(
        f"Refactored result range: [{np.min(refact_data):.6f}, {np.max(refact_data):.6f}]"
    )

    # Check numerical equivalence
    if np.allclose(orig_data, refact_data, atol=1e-10):
        print("✅ TimeLockedContrast: PERFECT MATCH")
        return True
    else:
        max_diff = np.max(np.abs(orig_data - refact_data))
        print(f"❌ TimeLockedContrast: MISMATCH (max diff: {max_diff:.10f})")
        return False


def test_time_locked_topography_roi_filtering():
    """Test TimeLockedTopography ROI filtering functionality."""
    print("Testing TimeLockedTopography ROI filtering...")

    # Create test data
    epochs = create_synthetic_epochs()

    # Test with specific ROI
    params = {
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": [0, 1, 2, 3, 4],  # First 5 channels
        "channel_aggregation_method": None,
        "trial_aggregation_method": None,
        "equipment": "egi256",
    }

    original_marker = OriginalTimeLockedTopography(**params)
    refactored_marker = RefactoredTimeLockedTopography(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_data = orig_result["timelockedtopo"]["data"]
    refact_data = refact_result["timelockedtopo"]["data"]

    print(f"Original ROI result shape: {orig_data.shape}")
    print(f"Refactored ROI result shape: {refact_data.shape}")

    if np.allclose(orig_data, refact_data, atol=1e-10):
        print("✅ TimeLockedTopography ROI filtering: PERFECT MATCH")
        return True
    else:
        max_diff = np.max(np.abs(orig_data - refact_data))
        print(
            f"❌ TimeLockedTopography ROI filtering: MISMATCH (max diff: {max_diff:.10f})"
        )
        return False


def test_time_locked_contrast_roi_filtering():
    """Test TimeLockedContrast ROI filtering functionality."""
    print("Testing TimeLockedContrast ROI filtering...")

    # Create test data
    epochs = create_synthetic_epochs()

    # Test with specific ROI
    params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": [0, 1, 2, 3, 4],  # First 5 channels
        "channel_aggregation_method": None,
        "trial_aggregation_method": None,
        "equipment": "egi256",
    }

    original_marker = OriginalTimeLockedContrast(**params)
    refactored_marker = RefactoredTimeLockedContrast(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_data = orig_result["timelockedcontrast"]["data"]
    refact_data = refact_result["timelockedcontrast"]["data"]

    print(f"Original ROI result shape: {orig_data.shape}")
    print(f"Refactored ROI result shape: {refact_data.shape}")

    if np.allclose(orig_data, refact_data, atol=1e-10):
        print("✅ TimeLockedContrast ROI filtering: PERFECT MATCH")
        return True
    else:
        max_diff = np.max(np.abs(orig_data - refact_data))
        print(
            f"❌ TimeLockedContrast ROI filtering: MISMATCH (max diff: {max_diff:.10f})"
        )
        return False


def test_time_locked_topography_missing_conditions():
    """Test TimeLockedTopography with missing conditions (should work normally)."""
    print("Testing TimeLockedTopography missing conditions handling...")

    # Create test data
    epochs = create_synthetic_epochs()

    # TimeLockedTopography doesn't depend on conditions, should work normally
    params = {
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": None,
        "channel_aggregation_method": "mean",
        "trial_aggregation_method": "mean",
        "equipment": "egi256",
    }

    original_marker = OriginalTimeLockedTopography(**params)
    refactored_marker = RefactoredTimeLockedTopography(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_data = orig_result["timelockedtopo"]["data"]
    refact_data = refact_result["timelockedtopo"]["data"]

    print(f"Original result shape: {orig_data.shape}")
    print(f"Refactored result shape: {refact_data.shape}")

    if np.allclose(orig_data, refact_data, atol=1e-10):
        print(
            "✅ TimeLockedTopography missing conditions handling: PERFECT MATCH"
        )
        return True
    else:
        max_diff = np.max(np.abs(orig_data - refact_data))
        print(
            f"❌ TimeLockedTopography missing conditions handling: MISMATCH (max diff: {max_diff:.10f})"
        )
        return False


def test_time_locked_contrast_missing_conditions():
    """Test TimeLockedContrast with missing conditions."""
    print("Testing TimeLockedContrast missing conditions handling...")

    # Create test data
    epochs = create_synthetic_epochs()

    # Test with non-existent conditions
    params = {
        "condition_a": "non_existent_a",
        "condition_b": "non_existent_b",
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": None,
        "channel_aggregation_method": "mean",
        "trial_aggregation_method": "mean",
        "equipment": "egi256",
    }

    original_marker = OriginalTimeLockedContrast(**params)
    refactored_marker = RefactoredTimeLockedContrast(**params)

    # Both should raise ValueError for missing conditions
    orig_error = None
    refact_error = None

    try:
        original_marker.compute({"data": epochs})
    except Exception as e:
        orig_error = e

    try:
        refactored_marker.compute({"data": epochs})
    except Exception as e:
        refact_error = e

    # Check that both raise the same type of error
    if (
        type(orig_error) is type(refact_error)
        and orig_error is not None
        and refact_error is not None
    ):
        print(
            "✅ TimeLockedContrast missing conditions handling: PERFECT MATCH"
        )
        return True
    else:
        print("❌ TimeLockedContrast missing conditions handling: MISMATCH")
        print(f"Original error: {orig_error}")
        print(f"Refactored error: {refact_error}")
        return False


def main():
    """Run all validation tests for time-locked markers refactoring."""
    print("=" * 70)
    print("TIME-LOCKED MARKERS REFACTORING VALIDATION")
    print("=" * 70)

    success = True

    # Test TimeLockedTopography equivalence
    success &= test_time_locked_topography_equivalence()

    # Test TimeLockedContrast equivalence
    success &= test_time_locked_contrast_equivalence()

    # Test ROI filtering
    success &= test_time_locked_topography_roi_filtering()
    success &= test_time_locked_contrast_roi_filtering()

    # Test missing conditions
    success &= test_time_locked_topography_missing_conditions()
    success &= test_time_locked_contrast_missing_conditions()

    print("\n" + "=" * 70)
    if success:
        print(
            "🎉 ALL TESTS PASSED - Time-locked markers refactoring is successful!"
        )
        print(
            "Refactored markers produce numerically identical results to originals."
        )
    else:
        print(
            "❌ SOME TESTS FAILED - Fix issues before deploying refactored markers!"
        )
    print("=" * 70)

    return success


if __name__ == "__main__":
    main()
