"""Test script to validate KolmogorovComplexity refactoring."""

import numpy as np
from mne import create_info
from mne.epochs import EpochsArray

# Import both implementations
from junifer_eeg.markers.kolmogorov_complexity import (
    KolmogorovComplexity as OriginalKC,
)
from junifer_eeg.markers.kolmogorov_complexity_new.kolmogorov_complexity import (
    KolmogorovComplexity as RefactoredKC,
)


def create_test_data():
    """Create synthetic EEG data for testing."""
    n_epochs = 10
    n_channels = 8
    n_times = 200
    sfreq = 250.0

    # Create synthetic EEG data with mixed frequency patterns
    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times)

    # Add some structured signals
    for i in range(n_channels):
        freq = 5 + i * 2  # Different frequencies per channel
        t = np.linspace(0, n_times / sfreq, n_times)
        data[:, i, :] += 0.5 * np.sin(2 * np.pi * freq * t)

    # Create MNE info
    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create epochs
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
    epochs = EpochsArray(data, info, events=events, tmin=0.0)

    return epochs


def test_numerical_equivalence():
    """Test that refactored implementation produces identical results."""
    print("Creating test data...")
    epochs = create_test_data()

    # Test parameters
    params = {
        "tmin": 0.0,
        "tmax": 0.6,
        "nbins": 16,
        "channel_method": None,
        "trial_method": None,
    }

    print("Testing original implementation...")
    original_marker = OriginalKC(**params)
    original_result = original_marker.compute({"data": epochs})

    print("Testing refactored implementation...")
    refactored_marker = RefactoredKC(**params)
    refactored_result = refactored_marker.compute({"data": epochs})

    # Compare results
    orig_data = original_result["kolmogorovcomplexity"]["data"]
    refact_data = refactored_result["kolmogorovcomplexity"]["data"]

    print(f"Original shape: {orig_data.shape}")
    print(f"Refactored shape: {refact_data.shape}")
    print(
        f"Original range: [{np.min(orig_data):.6f}, {np.max(orig_data):.6f}]"
    )
    print(
        f"Refactored range: [{np.min(refact_data):.6f}, {np.max(refact_data):.6f}]"
    )

    # Check numerical equivalence
    abs_diff = np.abs(orig_data - refact_data)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"Max absolute difference: {max_abs_diff:.10f}")
    print(f"Mean absolute difference: {mean_abs_diff:.10f}")

    # Check if results are identical (within floating point precision)
    tolerance = 1e-10
    assert max_abs_diff < tolerance, (
        f"MISMATCH: Results differ by more than {tolerance}"
    )
    print(
        f"✅ Numerical equivalence: PERFECT MATCH (max diff: {max_abs_diff:.10f})"
    )


def test_aggregation_methods():
    """Test that aggregation methods work correctly."""
    print("\nTesting aggregation methods...")
    epochs = create_test_data()

    # Test different aggregation combinations
    test_cases = [
        {
            "channel_method": "mean",
            "trial_method": None,
        },
        {
            "channel_method": None,
            "trial_method": "mean",
        },
        {
            "channel_method": "mean",
            "trial_method": "mean",
        },
        {
            "channel_method": "std",
            "trial_method": "median",
        },
    ]

    for i, params in enumerate(test_cases):
        print(f"  Test case {i + 1}: {params}")

        base_params = {"tmin": 0.0, "tmax": 0.6, "nbins": 16}
        base_params.update(params)

        original_marker = OriginalKC(**base_params)
        refactored_marker = RefactoredKC(**base_params)

        orig_result = original_marker.compute({"data": epochs})
        refact_result = refactored_marker.compute({"data": epochs})

        assert np.allclose(
            orig_result["kolmogorovcomplexity"]["data"],
            refact_result["kolmogorovcomplexity"]["data"],
            rtol=1e-10,
            atol=1e-10,
        ), f"{params} aggregation failed - Results don't match"


def test_roi_filtering():
    """Test that ROI filtering works correctly."""
    print("\nTesting ROI filtering...")
    epochs = create_test_data()

    # Test ROI filtering
    params = {
        "tmin": 0.0,
        "tmax": 0.6,
        "nbins": 16,
        "rois": ["E1", "E2", "E3"],  # Select first 3 channels
        "channel_method": None,
        "trial_method": None,
    }

    original_marker = OriginalKC(**params)
    refactored_marker = RefactoredKC(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_data = orig_result["kolmogorovcomplexity"]["data"]
    refact_data = refact_result["kolmogorovcomplexity"]["data"]

    print(f"Original shape with ROI: {orig_data.shape}")
    print(f"Refactored shape with ROI: {refact_data.shape}")

    assert np.allclose(orig_data, refact_data, atol=1e-10), (
        f"ROI filtering failed - Max diff: {np.max(np.abs(orig_data - refact_data)):.10f}"
    )
    print(
        f"✅ ROI filtering: PERFECT MATCH (max diff: {np.max(np.abs(orig_data - refact_data)):.10f})"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("KOLMOGOROV COMPLEXITY REFACTORING VALIDATION")
    print("=" * 60)

    success = True

    # Run all tests
    try:
        test_numerical_equivalence()
    except AssertionError as e:
        print(f"❌ Numerical equivalence test failed: {e}")
        success = False

    try:
        test_aggregation_methods()
    except AssertionError as e:
        print(f"❌ Aggregation methods test failed: {e}")
        success = False

    try:
        test_roi_filtering()
    except AssertionError as e:
        print(f"❌ ROI filtering test failed: {e}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Refactoring is successful!")
    else:
        print("❌ SOME TESTS FAILED - Refactoring needs fixes!")
    print("=" * 60)
