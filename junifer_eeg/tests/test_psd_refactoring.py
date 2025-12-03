"""Test script to validate Power Spectral Density refactoring."""

import numpy as np
from mne import create_info
from mne.epochs import EpochsArray
from mne.io import RawArray

# Import both implementations
from junifer_eeg.markers.power_spectral_density import (
    PowerSpectralDensityEstimator as OriginalEstimator,
)
from junifer_eeg.markers.power_spectral_density import (
    PowerSpectralDensitySummary as OriginalSummary,
)
from junifer_eeg.markers.power_spectral_density_new.psd_estimator import (
    PowerSpectralDensityEstimator as RefactoredEstimator,
)
from junifer_eeg.markers.power_spectral_density_new.psd_summary import (
    PowerSpectralDensitySummary as RefactoredSummary,
)


def create_test_epochs():
    """Create synthetic EEG epochs data for testing."""
    n_epochs = 15
    n_channels = 8
    n_times = 250
    sfreq = 250.0

    # Create synthetic EEG data with mixed frequency patterns
    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times)

    # Add structured signals with different frequencies per channel
    t = np.linspace(0, n_times / sfreq, n_times)
    for i in range(n_channels):
        freq = (
            5 + i * 3
        )  # Different frequencies per channel (5, 8, 11, 14, 17, 20, 23, 26 Hz)
        data[:, i, :] += 0.5 * np.sin(2 * np.pi * freq * t)
        # Add some harmonics
        data[:, i, :] += 0.2 * np.sin(2 * np.pi * freq * 2 * t)

    # Create MNE info
    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create epochs
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
    epochs = EpochsArray(data, info, events=events, tmin=0.0)

    return epochs


def create_test_raw():
    """Create synthetic EEG raw data for testing."""
    n_channels = 8
    n_times = 1000
    sfreq = 250.0

    # Create synthetic EEG data with mixed frequency patterns
    np.random.seed(42)
    data = np.random.randn(n_channels, n_times)

    # Add structured signals with different frequencies per channel
    t = np.linspace(0, n_times / sfreq, n_times)
    for i in range(n_channels):
        freq = 5 + i * 3  # Different frequencies per channel
        data[i, :] += 0.5 * np.sin(2 * np.pi * freq * t)
        data[i, :] += 0.2 * np.sin(2 * np.pi * freq * 2 * t)

    # Create MNE info
    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create raw
    raw = RawArray(data, info)

    return raw


def test_estimator_equivalence_epochs():
    """Test that refactored Estimator produces identical results for epochs data."""
    print("Testing Estimator with epochs data...")
    epochs = create_test_epochs()

    # Test parameters
    params = {
        "tmin": 0.0,
        "tmax": 0.8,
        "fmin": 1.0,
        "fmax": 50.0,
        "n_fft": 256,
        "n_per_seg": 128,
        "n_overlap": 64,
    }

    original_marker = OriginalEstimator(**params)
    refactored_marker = RefactoredEstimator(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    # Compare PSD data
    orig_psd = orig_result["psd_data"]["data"]
    refact_psd = refact_result["psd_data"]["data"]

    print(f"Original PSD shape: {orig_psd.shape}")
    print(f"Refactored PSD shape: {refact_psd.shape}")
    print(
        f"Original PSD range: [{np.min(orig_psd):.6e}, {np.max(orig_psd):.6e}]"
    )
    print(
        f"Refactored PSD range: [{np.min(refact_psd):.6e}, {np.max(refact_psd):.6e}]"
    )

    # Check numerical equivalence
    abs_diff = np.abs(orig_psd - refact_psd)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"PSD Max absolute difference: {max_abs_diff:.10e}")
    print(f"PSD Mean absolute difference: {mean_abs_diff:.10e}")

    # Check if results are identical (within floating point precision)
    tolerance = 1e-12
    if max_abs_diff < tolerance:
        print("✅ PSD data: PERFECT MATCH")
        psd_match = True
    else:
        print(f"❌ PSD data: MISMATCH (tolerance {tolerance})")
        psd_match = False

    # Compare frequencies
    orig_freqs = orig_result["psd_freqs"]["data"]
    refact_freqs = refact_result["psd_freqs"]["data"]

    freq_diff = np.max(np.abs(orig_freqs - refact_freqs))
    print(f"Frequencies max difference: {freq_diff:.10e}")

    if freq_diff < tolerance:
        print("✅ Frequencies: PERFECT MATCH")
        freq_match = True
    else:
        print("❌ Frequencies: MISMATCH")
        freq_match = False

    # Compare normalized PSD
    orig_norm = orig_result["psd_data_norm"]["data"]
    refact_norm = refact_result["psd_data_norm"]["data"]

    norm_diff = np.max(np.abs(orig_norm - refact_norm))
    print(f"Normalized PSD max difference: {norm_diff:.10e}")

    if norm_diff < tolerance:
        print("✅ Normalized PSD: PERFECT MATCH")
        norm_match = True
    else:
        print("❌ Normalized PSD: MISMATCH")
        norm_match = False

    return psd_match and freq_match and norm_match


def test_estimator_equivalence_raw():
    """Test that refactored Estimator produces identical results for raw data."""
    print("\nTesting Estimator with raw data...")
    raw = create_test_raw()

    # Test parameters
    params = {
        "tmin": 0.0,
        "tmax": 2.0,
        "fmin": 1.0,
        "fmax": 50.0,
        "n_fft": 512,
        "n_per_seg": 256,
        "n_overlap": 128,
    }

    original_marker = OriginalEstimator(**params)
    refactored_marker = RefactoredEstimator(**params)

    orig_result = original_marker.compute({"data": raw})
    refact_result = refactored_marker.compute({"data": raw})

    # Compare PSD data
    orig_psd = orig_result["psd_data"]["data"]
    refact_psd = refact_result["psd_data"]["data"]

    print(f"Original PSD shape: {orig_psd.shape}")
    print(f"Refactored PSD shape: {refact_psd.shape}")

    # Check numerical equivalence
    abs_diff = np.abs(orig_psd - refact_psd)
    max_abs_diff = np.max(abs_diff)

    print(f"PSD Max absolute difference: {max_abs_diff:.10e}")

    # Check if results are identical (within floating point precision)
    tolerance = 1e-12
    if max_abs_diff < tolerance:
        print("✅ PSD data: PERFECT MATCH")
        return True
    else:
        print(f"❌ PSD data: MISMATCH (tolerance {tolerance})")
        return False


def test_summary_equivalence():
    """Test that refactored Summary produces identical results."""
    print("\nTesting Summary marker...")
    epochs = create_test_epochs()

    # Test parameters
    params = {
        "percentile": 50.0,  # Median frequency
        "tmin": 0.0,
        "tmax": 0.8,
        "fmin": 1.0,
        "fmax": 50.0,
        "n_fft": 256,
        "n_per_seg": 128,
        "n_overlap": 64,
        "channel_aggregation_method": None,
        "trial_aggregation_method": None,
    }

    original_marker = OriginalSummary(**params)
    refactored_marker = RefactoredSummary(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    # Compare SEF values
    orig_sef = orig_result["psdsummary"]["data"]
    refact_sef = refact_result["psdsummary"]["data"]

    print(f"Original SEF shape: {orig_sef.shape}")
    print(f"Refactored SEF shape: {refact_sef.shape}")
    print(
        f"Original SEF range: [{np.min(orig_sef):.6f}, {np.max(orig_sef):.6f}]"
    )
    print(
        f"Refactored SEF range: [{np.min(refact_sef):.6f}, {np.max(refact_sef):.6f}]"
    )

    # Check numerical equivalence
    abs_diff = np.abs(orig_sef - refact_sef)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"SEF Max absolute difference: {max_abs_diff:.10f}")
    print(f"SEF Mean absolute difference: {mean_abs_diff:.10f}")

    # Check if results are identical (within floating point precision)
    tolerance = 1e-10
    if max_abs_diff < tolerance:
        print("✅ SEF data: PERFECT MATCH")
        return True
    else:
        print(f"❌ SEF data: MISMATCH (tolerance {tolerance})")
        return False


def test_summary_aggregation():
    """Test that Summary aggregation methods work correctly."""
    print("\nTesting Summary aggregation methods...")
    epochs = create_test_epochs()

    # Test different aggregation combinations
    test_cases = [
        {
            "percentile": 0.9,
            "channel_aggregation_method": "mean",
            "trial_aggregation_method": None,
        },
        {
            "percentile": 0.25,
            "channel_aggregation_method": None,
            "trial_aggregation_method": "mean",
        },
        {
            "percentile": 0.75,
            "channel_aggregation_method": "mean",
            "trial_aggregation_method": "mean",
        },
        {
            "percentile": 0.5,
            "channel_aggregation_method": "std",
            "trial_aggregation_method": "median",
        },
    ]

    success = True
    for i, params in enumerate(test_cases):
        print(
            f"  Test case {i + 1}: percentile={params['percentile']}, agg={params.get('channel_aggregation_method', 'None')}/{params.get('trial_aggregation_method', 'None')}"
        )

        base_params = {
            "tmin": 0.0,
            "tmax": 0.8,
            "fmin": 1.0,
            "fmax": 50.0,
            "n_fft": 256,
            "n_per_seg": 128,
            "n_overlap": 64,
        }
        base_params.update(params)

        original_marker = OriginalSummary(**base_params)
        refactored_marker = RefactoredSummary(**base_params)

        orig_result = original_marker.compute({"data": epochs})
        refact_result = refactored_marker.compute({"data": epochs})

        orig_sef = orig_result["psdsummary"]["data"]
        refact_sef = refact_result["psdsummary"]["data"]

        if np.allclose(orig_sef, refact_sef, atol=1e-10):
            print("    ✅ PASS")
        else:
            print(
                f"    ❌ FAIL - Max diff: {np.max(np.abs(orig_sef - refact_sef)):.10f}"
            )
            success = False

    return success


def test_summary_roi():
    """Test that Summary ROI filtering works correctly."""
    print("\nTesting Summary ROI filtering...")
    epochs = create_test_epochs()

    # Test ROI filtering
    params = {
        "percentile": 50.0,
        "tmin": 0.0,
        "tmax": 0.8,
        "fmin": 1.0,
        "fmax": 50.0,
        "rois": ["E1", "E2", "E3"],  # Select first 3 channels
        "channel_aggregation_method": None,
        "trial_aggregation_method": None,
    }

    original_marker = OriginalSummary(**params)
    refactored_marker = RefactoredSummary(**params)

    orig_result = original_marker.compute({"data": epochs})
    refact_result = refactored_marker.compute({"data": epochs})

    orig_sef = orig_result["psdsummary"]["data"]
    refact_sef = refact_result["psdsummary"]["data"]

    print(f"Original SEF shape with ROI: {orig_sef.shape}")
    print(f"Refactored SEF shape with ROI: {refact_sef.shape}")

    if np.allclose(orig_sef, refact_sef, atol=1e-10):
        print("✅ ROI filtering works correctly!")
        return True
    else:
        print(
            f"❌ ROI filtering failed - Max diff: {np.max(np.abs(orig_sef - refact_sef)):.10f}"
        )
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("POWER SPECTRAL DENSITY REFACTORING VALIDATION")
    print("=" * 70)

    success = True

    # Run all tests
    success &= test_estimator_equivalence_epochs()
    success &= test_estimator_equivalence_raw()
    success &= test_summary_equivalence()
    success &= test_summary_aggregation()
    success &= test_summary_roi()

    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED - PSD refactoring is successful!")
    else:
        print("❌ SOME TESTS FAILED - PSD refactoring needs fixes!")
    print("=" * 70)
