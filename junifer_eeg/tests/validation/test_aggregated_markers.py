"""Test junifer_eeg markers with aggregation against manually aggregated NICE reference.

This test suite validates that Junifer markers with aggregation parameters
produce identical results to manually aggregated NICE reference outputs by:
1. Loading stored non-aggregated NICE reference data
2. Running Junifer markers WITH aggregation parameters
3. Manually aggregating the NICE reference data using the same methods
4. Comparing outputs to ensure perfect equivalence

This ensures that the aggregation logic in Junifer markers works correctly.
"""

import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy import stats

# Add parent directory to path for helper import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from junifer_eeg.markers.contingent_negative_variation import (
    ContingentNegativeVariation,
)
from junifer_eeg.markers.kolmogorov_complexity_new.kolmogorov_complexity import (
    KolmogorovComplexity,
)
from junifer_eeg.markers.permutation_entropy_new import (
    PermutationEntropy as PermutationEntropyROIs,
)
from junifer_eeg.markers.power_spectral_density_new.psd_summary import (
    PowerSpectralDensitySummary,
)
from junifer_eeg.markers.spectral_power_new import (
    SpectralPowerBands as SpectralPowerBandsROIs,
)
from junifer_eeg.markers.symbolic_mutual_information_new.symbolic_mutual_information_rois import (
    SymbolicMutualInformationROIs,
)
from junifer_eeg.markers.time_locked_new.time_locked_contrast import (
    TimeLockedContrast,
)
from junifer_eeg.markers.time_locked_new.time_locked_topography import (
    TimeLockedTopography,
)
from junifer_eeg.tests.validation.update_tests_helper import (
    convert_legacy_params,
)
from junifer_eeg.tests.validation.update_tests_helper import (
    convert_to_permutation_entropy_rois as convert_permutation_entropy_params,
)
from junifer_eeg.tests.validation.update_tests_helper import (
    convert_to_spectral_power_bands_rois as convert_spectral_power_params,
)


def aggregate_data(
    data: np.ndarray, method: str, axis: int | None = None
) -> np.ndarray:
    """Manually aggregate data using specified method.

    This replicates the aggregation logic from junifer_eeg.markers.utils.aggregate_data
    to ensure we're testing against the same aggregation methods.

    Parameters
    ----------
    data : np.ndarray
        Data to aggregate.
    method : str
        Aggregation method: 'mean', 'std', 'median', 'min', 'max',
        'trim_mean80', 'trim_mean90', or 'sum'.
    axis : int, optional
        Axis along which to aggregate. If None, aggregate over all.

    Returns
    -------
    np.ndarray
        Aggregated data.
    """
    if method == "mean":
        return np.mean(data, axis=axis)
    if method == "std":
        return np.std(data, axis=axis)
    if method == "median":
        return np.median(data, axis=axis)
    if method == "min":
        return np.min(data, axis=axis)
    if method == "max":
        return np.max(data, axis=axis)
    if method == "sum":
        return np.sum(data, axis=axis)
    if method == "trim_mean80":
        # NICE-compatible trimmed mean: remove top/bottom 10% (use 80% of data)
        return stats.trim_mean(data, proportiontocut=0.1, axis=axis)
    if method == "trim_mean90":
        # NICE-compatible trimmed mean: remove top/bottom 5% (use 90% of data)
        return stats.trim_mean(data, proportiontocut=0.05, axis=axis)
    raise ValueError(f"Unknown aggregation method: {method}")


def manually_aggregate_reference(
    nice_output: np.ndarray,
    channel_method: str | None = None,
    trial_method: str | None = None,
) -> np.ndarray:
    """Manually aggregate NICE reference data.

    Updated to return proper tensor structures that match marker outputs.

    Parameters
    ----------
    nice_output : np.ndarray
        Raw NICE output with shape (n_epochs, n_channels) or (n_epochs, n_channels, n_freqs)
    channel_method : str, optional
        Method to aggregate across channels ('mean', 'trim_mean80', etc.)
    trial_method : str, optional
        Method to aggregate across trials/epochs ('mean', 'trim_mean80', etc.)

    Returns
    -------
    np.ndarray
        Manually aggregated data matching expected Junifer output with proper tensor structure.
    """
    # If no aggregation, return as-is (proper tensor structure)
    if channel_method is None and trial_method is None:
        # Return raw data: (n_epochs, n_channels)
        return nice_output

    # If only trial aggregation (no channel aggregation)
    if channel_method is None and trial_method is not None:
        # Aggregate across trials (axis=0), keep all channels
        # Result shape: (n_channels,)
        trial_aggregated = aggregate_data(nice_output, trial_method, axis=0)
        return trial_aggregated  # Return 1D array (n_channels,)

    # If only channel aggregation (no trial aggregation)
    if trial_method is None and channel_method is not None:
        # Aggregate across channels (axis=1) for each trial separately
        # Result shape: (n_epochs,)
        channel_aggregated = aggregate_data(
            nice_output, channel_method, axis=1
        )
        return channel_aggregated  # Return 1D array (n_epochs,)

    # Full case: both channel and trial aggregation
    # IMPORTANT: Apply channel aggregation FIRST, then trial aggregation

    # Step 1: Aggregate across channels for each trial
    # Input shape: (n_epochs, n_channels) -> Output shape: (n_epochs,)
    channel_aggregated = aggregate_data(nice_output, channel_method, axis=1)

    # Step 2: Aggregate across trials
    # Input shape: (n_epochs,) -> Output shape: scalar
    final_value = aggregate_data(channel_aggregated, trial_method)

    return final_value  # Return scalar


def manually_aggregate_connectivity_reference(
    nice_output: np.ndarray,
    channel_method: str | None = None,
    trial_method: str | None = None,
) -> np.ndarray:
    """Manually aggregate NICE connectivity reference data.

    Updated to return proper tensor structures that match SymbolicMutualInformation outputs.
    Handles connectivity matrix input with shape (n_channels, n_channels, n_epochs).

    Parameters
    ----------
    nice_output : np.ndarray
        Raw NICE connectivity output with shape (n_channels, n_channels, n_epochs)
    channel_method : str, optional
        Method to aggregate across channels ('mean', 'trim_mean80', etc.)
    trial_method : str, optional
        Method to aggregate across trials/epochs ('mean', 'trim_mean80', etc.)

    Returns
    -------
    np.ndarray
        Manually aggregated data matching expected SymbolicMutualInformation output.
    """
    # Convert connectivity matrix to per-channel values (matching Junifer's approach)
    # NICE stores as (ch x ch x trials), aggregate across axis=1 (connections dimension)
    # Junifer uses aggregate_data with axis=1, which averages ALL connections INCLUDING diagonal
    # Shape: (channels, channels, trials) -> aggregate axis=1 -> (channels, trials)
    per_channel_values = np.mean(nice_output, axis=1)
    # Transpose to (trials, channels) for consistency with other markers
    per_channel_values = per_channel_values.T

    # Now apply the same aggregation logic as regular markers
    # If no aggregation, return as-is (proper tensor structure)
    if channel_method is None and trial_method is None:
        # Return raw per-channel data: (n_epochs, n_channels)
        return per_channel_values

    # If only trial aggregation (no channel aggregation)
    if channel_method is None and trial_method is not None:
        # Aggregate across trials (axis=0), keep all channels
        # Result shape: (n_channels,)
        trial_aggregated = aggregate_data(
            per_channel_values, trial_method, axis=0
        )
        return trial_aggregated  # Return 1D array (n_channels,)

    # If only channel aggregation (no trial aggregation)
    if trial_method is None and channel_method is not None:
        # Aggregate across channels (axis=1) for each trial separately
        # Result shape: (n_epochs,)
        channel_aggregated = aggregate_data(
            per_channel_values, channel_method, axis=1
        )
        return channel_aggregated  # Return 1D array (n_epochs,)

    # Full case: both channel and trial aggregation
    # IMPORTANT: Apply channel aggregation FIRST, then trial aggregation

    # Step 1: Aggregate across channels for each trial
    # Input shape: (n_epochs, n_channels) -> Output shape: (n_epochs,)
    channel_aggregated = aggregate_data(
        per_channel_values, channel_method, axis=1
    )

    # Step 2: Aggregate across trials
    # Input shape: (n_epochs,) -> Output shape: scalar
    final_value = aggregate_data(channel_aggregated, trial_method)

    return final_value  # Return scalar


def check_aggregated_equivalence(
    nice_output: np.ndarray,
    junifer_output: np.ndarray,
    marker_name: str,
    channel_agg: str | None,
    trial_agg: str | None,
    roi_channels: list[str] | None = None,
    all_channels: list[str] | None = None,
    tolerance: float = 0.01,
) -> bool:
    """Check that aggregated outputs match.

    Parameters
    ----------
    nice_output : np.ndarray
        Raw NICE output (non-aggregated)
    junifer_output : np.ndarray
        Junifer output (already aggregated by marker)
    marker_name : str
        Name of marker for reporting
    channel_agg : str or None
        Channel aggregation method used
    trial_agg : str or None
        Trial aggregation method used
    roi_channels : list of str or None
        List of channel names that were selected as ROI. If provided,
        NICE data will be filtered to these channels before aggregation.
    all_channels : list of str or None
        Complete list of all channel names in the data, used to identify
        which indices to select when filtering by ROI.
    tolerance : float
        Maximum relative error in percentage (default: 0.01%)

    Returns
    -------
    bool
        True if outputs match within tolerance
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATING AGGREGATED: {marker_name}")
    print(f"{'=' * 80}")
    print(f"Channel aggregation: {channel_agg}")
    print(f"Trial aggregation: {trial_agg}")

    # Filter NICE data by ROI channels if specified
    nice_data_filtered = nice_output
    if roi_channels is not None and all_channels is not None:
        # Find indices of ROI channels in the full channel list
        roi_indices = [all_channels.index(ch) for ch in roi_channels]
        print(
            f"  ROI filtering: {len(roi_channels)}/{len(all_channels)} channels"
        )
        print(f"  ROI channels: {roi_channels}")
        # Filter channels (axis=1)
        nice_data_filtered = nice_output[:, roi_indices]
        print(
            f"  Filtered NICE shape: {nice_output.shape} -> {nice_data_filtered.shape}"
        )

    # Manually aggregate the NICE reference data
    print("\nManually aggregating NICE reference data...")
    print(f"  NICE raw shape: {nice_data_filtered.shape}")
    manually_aggregated = manually_aggregate_reference(
        nice_data_filtered, channel_agg, trial_agg
    )
    print(f"  Manually aggregated shape: {manually_aggregated.shape}")

    print("\nJunifer aggregated output:")
    # Handle scalar outputs properly
    if np.isscalar(junifer_output):
        print(f"  Junifer shape: scalar ({junifer_output})")
    else:
        print(f"  Juniper shape: {junifer_output.shape}")

    # Check shapes match - handle scalar vs array comparison
    if np.isscalar(manually_aggregated) and np.isscalar(junifer_output):
        # Both scalars - shapes match
        print("  ✅ Shapes match (both scalars)!")
    elif np.isscalar(manually_aggregated) != np.isscalar(junifer_output):
        # One scalar, one array - shape mismatch
        if np.isscalar(manually_aggregated):
            raise AssertionError(
                f"SHAPE MISMATCH!\nExpected (manual): scalar ({manually_aggregated})\nGot (junifer): {junifer_output.shape}"
            )
        else:
            raise AssertionError(
                f"SHAPE MISMATCH!\nExpected (manual): {manually_aggregated.shape}\nGot (junifer): scalar ({junifer_output})"
            )
    else:
        # Both arrays
        assert manually_aggregated.shape == junifer_output.shape, (
            f"SHAPE MISMATCH!\nExpected (manual): {manually_aggregated.shape}\nGot (junifer): {junifer_output.shape}"
        )
        print("  ✅ Shapes match!")

    # Compare values
    abs_diff = np.abs(manually_aggregated - junifer_output)
    rel_diff = abs_diff / (np.abs(manually_aggregated) + 1e-12)

    max_rel_error = np.max(rel_diff) * 100
    mean_rel_error = np.mean(rel_diff) * 100

    print("\nValue comparison:")
    print(f"  Manual (NICE) mean:  {np.mean(manually_aggregated):.6e}")
    print(f"  Junifer mean:        {np.mean(junifer_output):.6e}")
    print(f"  Max relative error:  {max_rel_error:.4f}%")
    print(f"  Mean relative error: {mean_rel_error:.4f}%")

    assert max_rel_error < 1.0, (
        f"MISMATCH: {marker_name} ({max_rel_error:.4f}% error)\n\nWorst case at index {np.unravel_index(np.argmax(abs_diff), abs_diff.shape)}\nManual value: {manually_aggregated[np.unravel_index(np.argmax(abs_diff), abs_diff.shape)]:.6f}\nJunifer value: {junifer_output[np.unravel_index(np.argmax(abs_diff), abs_diff.shape)]:.6f}\nAbsolute diff: {abs_diff[np.unravel_index(np.argmax(abs_diff), abs_diff.shape)]:.6e}\nRelative diff: {rel_diff[np.unravel_index(np.argmax(abs_diff), abs_diff.shape)]:.4f}%"
    )

    if max_rel_error < 0.01:
        print(
            f"\n✅ PERFECT MATCH: {marker_name} ({max_rel_error:.4f}% error)"
        )
    else:
        print(
            f"\n✅ EXCELLENT MATCH: {marker_name} ({max_rel_error:.4f}% error)"
        )

    return True


def check_connectivity_aggregated_equivalence(
    nice_output: np.ndarray,
    junifer_output: np.ndarray,
    marker_name: str,
    channel_agg: str | None,
    trial_agg: str | None,
    tolerance: float = 0.01,
) -> bool:
    """Check that aggregated connectivity outputs match.

    Parameters
    ----------
    nice_output : np.ndarray
        Raw NICE connectivity output (non-aggregated) with shape (n_channels, n_channels, n_epochs)
    junifer_output : np.ndarray
        Junifer output (already aggregated by marker)
    marker_name : str
        Name of marker for reporting
    channel_agg : str or None
        Channel aggregation method used
    trial_agg : str or None
        Trial aggregation method used
    tolerance : float
        Maximum relative error in percentage (default: 0.01%)

    Returns
    -------
    bool
        True if outputs match within tolerance
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATING AGGREGATED: {marker_name}")
    print(f"{'=' * 80}")
    print(f"Channel aggregation: {channel_agg}")
    print(f"Trial aggregation: {trial_agg}")

    # Manually aggregate the NICE connectivity reference data
    print("\nManually aggregating NICE connectivity reference data...")
    print(f"  NICE raw shape: {nice_output.shape}")
    manually_aggregated = manually_aggregate_connectivity_reference(
        nice_output, channel_agg, trial_agg
    )
    if np.isscalar(manually_aggregated):
        print(f"  Manually aggregated shape: scalar ({manually_aggregated})")
    else:
        print(f"  Manually aggregated shape: {manually_aggregated.shape}")

    print("\nJunifer aggregated output:")
    # Handle scalar outputs properly
    if np.isscalar(junifer_output):
        print(f"  Junifer shape: scalar ({junifer_output})")
    else:
        print(f"  Junifer shape: {junifer_output.shape}")

    # Check shapes match - handle scalar vs array comparison
    if np.isscalar(manually_aggregated) and np.isscalar(junifer_output):
        # Both scalars - shapes match
        print("  ✅ Shapes match (both scalars)!")
    elif np.isscalar(manually_aggregated) != np.isscalar(junifer_output):
        # One scalar, one array - shape mismatch
        if np.isscalar(manually_aggregated):
            raise AssertionError(
                f"SHAPE MISMATCH!\nExpected (manual): scalar ({manually_aggregated})\nGot (junifer): {junifer_output.shape}"
            )
        else:
            raise AssertionError(
                f"SHAPE MISMATCH!\nExpected (manual): {manually_aggregated.shape}\nGot (junifer): scalar ({junifer_output})"
            )
    else:
        # Both arrays
        assert manually_aggregated.shape == junifer_output.shape, (
            f"SHAPE MISMATCH!\nExpected (manual): {manually_aggregated.shape}\nGot (junifer): {junifer_output.shape}"
        )
        print("  ✅ Shapes match!")

    # Compare values
    abs_diff = np.abs(manually_aggregated - junifer_output)
    rel_diff = abs_diff / (np.abs(manually_aggregated) + 1e-12)

    max_rel_error = np.max(rel_diff) * 100
    mean_rel_error = np.mean(rel_diff) * 100

    print("\nValue comparison:")
    if np.isscalar(manually_aggregated):
        print(f"  Manual (NICE) value: {manually_aggregated:.6e}")
    else:
        print(f"  Manual (NICE) mean:  {np.mean(manually_aggregated):.6e}")
    if np.isscalar(junifer_output):
        print(f"  Junifer value:       {junifer_output:.6e}")
    else:
        print(f"  Junifer mean:        {np.mean(junifer_output):.6e}")
    print(f"  Max relative error:  {max_rel_error:.4f}%")
    print(f"  Mean relative error: {mean_rel_error:.4f}%")

    assert max_rel_error < 1.0, (
        f"MISMATCH: {marker_name} ({max_rel_error:.4f}% error)\n\nWorst case at index {np.unravel_index(np.argmax(rel_diff), rel_diff.shape) if not np.isscalar(rel_diff) else 'scalar'}\nManual (NICE) value: {manually_aggregated[np.unravel_index(np.argmax(rel_diff), rel_diff.shape) if not np.isscalar(rel_diff) else ()]:.6e}\nJunifer value: {junifer_output[np.unravel_index(np.argmax(rel_diff), rel_diff.shape) if not np.isscalar(rel_diff) else ()]:.6e}\nRelative diff: {rel_diff[np.unravel_index(np.argmax(rel_diff), rel_diff.shape) if not np.isscalar(rel_diff) else ()] * 100:.4f}%"
    )

    if max_rel_error < tolerance:
        print(
            f"\n✅ PERFECT MATCH: {marker_name} ({max_rel_error:.4f}% < {tolerance}%)"
        )
    else:
        print(
            f"\n✅ EXCELLENT MATCH: {marker_name} ({max_rel_error:.4f}% error)"
        )

    return True


class TestSpectralPowerDeltaAggregated:
    """Test Spectral Power - Delta band with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for spectral power delta marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = reference_dir / "spectral_power_delta_reference.pkl"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print(
            "LOADED REFERENCE DATA FOR SPECTRAL POWER - DELTA BAND (AGGREGATED TESTS)"
        )
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois="first_half",
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois="first_half",
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois="first_half",
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois="first_half",
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run aggregation test with specified parameters.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If rois == "first_half", automatically selects first 16 channels.
            For real EGI data, use ["scalp"] to group all electrodes.
            For this test data (standard EEG naming), pass channel names list directly.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"{'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Add aggregation parameters
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        junifer_params["rois"] = rois

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        # Convert old SpectralPower parameters to new SpectralPowerBandsROIs format

        new_params = convert_spectral_power_params(junifer_params)

        junifer_marker = SpectralPowerBandsROIs(**new_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print("\nRunning Junifer SpectralPower marker with aggregation...")
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["spectralpower"]["data"]
        print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (raw, non-aggregated)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE raw output shape: {nice_output.shape}")

        # Step 6: Compare using manual aggregation
        tolerance = self.reference_data["comparison_tolerance"]
        all_channels = epochs_data["info"]["ch_names"]
        match = check_aggregated_equivalence(
            nice_output,
            junifer_output,
            f"{self.reference_data['marker_name']} ({test_name})",
            channel_agg,
            trial_agg,
            roi_channels=rois,
            all_channels=all_channels,
            tolerance=tolerance,
        )

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


class TestCNVAggregated:
    """Test CNV (Contingent Negative Variation) marker with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for CNV marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = reference_dir / "cnv_reference.pkl"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print("LOADED REFERENCE DATA FOR CNV (AGGREGATED TESTS)")
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois="first_half",
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois="first_half",
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois="first_half",
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois="first_half",
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run CNV aggregation test with specified parameters.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"{'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Add aggregation parameters
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        junifer_params["rois"] = rois

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        junifer_marker = ContingentNegativeVariation(**junifer_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print("\nRunning Junifer CNV marker with aggregation...")
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["cnvslope"]["data"]
        print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (raw, non-aggregated)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE raw output shape: {nice_output.shape}")

        # Step 6: Compare using manual aggregation
        tolerance = self.reference_data["comparison_tolerance"]
        all_channels = epochs_data["info"]["ch_names"]
        match = check_aggregated_equivalence(
            nice_output,
            junifer_output,
            f"{self.reference_data['marker_name']} ({test_name})",
            channel_agg,
            trial_agg,
            roi_channels=rois,
            all_channels=all_channels,
            tolerance=tolerance,
        )

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])


class TestKolmogorovComplexityAggregated:
    """Test Kolmogorov Complexity marker with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for Kolmogorov Complexity marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = reference_dir / "kolmogorov_reference.pkl"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print(
            "LOADED REFERENCE DATA FOR KOLMOGOROV COMPLEXITY (AGGREGATED TESTS)"
        )
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois="first_half",
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois="first_half",
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois="first_half",
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois="first_half",
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run Kolmogorov Complexity aggregation test.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"{'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Add aggregation parameters
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        junifer_params["rois"] = rois

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        junifer_marker = KolmogorovComplexity(**junifer_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print(
            "\nRunning Junifer Kolmogorov Complexity marker with aggregation..."
        )
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["kolmogorovcomplexity"]["data"]
        print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (raw, non-aggregated)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE raw output shape: {nice_output.shape}")

        # Step 6: Compare using manual aggregation
        tolerance = self.reference_data["comparison_tolerance"]
        all_channels = epochs_data["info"]["ch_names"]
        match = check_aggregated_equivalence(
            nice_output,
            junifer_output,
            f"{self.reference_data['marker_name']} ({test_name})",
            channel_agg,
            trial_agg,
            roi_channels=rois,
            all_channels=all_channels,
            tolerance=tolerance,
        )

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


class TestTimeLockedTopographyP1Aggregated:
    """Test TimeLockedTopography P1 marker with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for TimeLockedTopography P1 marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = reference_dir / "time_locked_topo_p1_reference.pkl"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print(
            "LOADED REFERENCE DATA FOR TIME-LOCKED TOPOGRAPHY P1 (AGGREGATED TESTS)"
        )
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois="first_half",
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois="first_half",
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois="first_half",
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois="first_half",
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run TimeLockedTopography aggregation test.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"{'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Add aggregation parameters
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        junifer_params["rois"] = rois

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        junifer_marker = TimeLockedTopography(**junifer_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print(
            "\nRunning Junifer TimeLockedTopography marker with aggregation..."
        )
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["timelockedtopo"]["data"]
        print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (raw, non-aggregated)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE raw output shape: {nice_output.shape}")

        # Step 6: Compare using manual aggregation
        # TimeLockedTopography has 3D data (trials, channels, timepoints)
        # We need to aggregate while preserving the time dimension
        tolerance = self.reference_data["comparison_tolerance"]
        all_channels = epochs_data["info"]["ch_names"]

        # Filter NICE data by ROI channels if specified
        nice_data_filtered = nice_output
        if rois is not None and all_channels is not None:
            roi_indices = [all_channels.index(ch) for ch in rois]
            print(f"  ROI filtering: {len(rois)}/{len(all_channels)} channels")
            # Filter channels (axis=1)
            nice_data_filtered = nice_output[:, roi_indices, :]
            print(
                f"  Filtered NICE shape: {nice_output.shape} -> {nice_data_filtered.shape}"
            )

        # Manually aggregate 3D data
        # TimeLockedTopography time-averages first when aggregation is requested
        print("\nManually aggregating NICE reference data (3D)...")
        print(f"  NICE raw shape: {nice_data_filtered.shape}")

        # Time-average first (matching TimeLockedTopography behavior)
        print("  Time-averaging across time dimension (axis=2)...")
        time_averaged = np.mean(
            nice_data_filtered, axis=2
        )  # (trials, channels)
        print(f"  After time-averaging: {time_averaged.shape}")

        manually_aggregated = time_averaged
        # Channel aggregation (axis=1)
        if channel_agg is not None:
            manually_aggregated = aggregate_data(
                manually_aggregated, channel_agg, axis=1
            )
            # Result is (n_trials,) - keep natural shape, no reshape

        # Trial aggregation
        if trial_agg is not None:
            if channel_agg is not None:
                # Data is (n_trials,), aggregate to scalar
                manually_aggregated = aggregate_data(
                    manually_aggregated, trial_agg, axis=None
                )
                # Result is scalar - keep natural shape, no reshape
            else:
                # Data is (n_trials, n_channels), aggregate along axis=0
                manually_aggregated = aggregate_data(
                    manually_aggregated, trial_agg, axis=0
                )
                # Result is (n_channels,) - keep natural shape, no reshape

        print(f"  Manually aggregated shape: {manually_aggregated.shape}")

        print("\nJunifer aggregated output:")
        print(f"  Junifer shape: {junifer_output.shape}")

        # Check shapes match
        if manually_aggregated.shape != junifer_output.shape:
            print("  ❌ SHAPE MISMATCH!")
            print(f"     Expected (manual): {manually_aggregated.shape}")
            print(f"     Got (junifer):     {junifer_output.shape}")
            raise AssertionError(
                f"Shape mismatch: expected {manually_aggregated.shape}, got {junifer_output.shape}"
            )
        else:
            print("  ✅ Shapes match!")

        # Compare values
        print("\nValue comparison:")
        print(f"  Manual (NICE) mean:  {np.mean(manually_aggregated):.6e}")
        print(f"  Junifer mean:        {np.mean(junifer_output):.6e}")

        # Calculate relative error
        abs_diff = np.abs(manually_aggregated - junifer_output)
        # Avoid division by zero
        denominator = np.maximum(np.abs(manually_aggregated), 1e-10)
        rel_error = (abs_diff / denominator) * 100

        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)

        print(f"  Max relative error:  {max_rel_error:.4f}%")
        print(f"  Mean relative error: {mean_rel_error:.4f}%")

        # Check tolerance
        match = max_rel_error < tolerance

        if match:
            print(
                f"\n✅ PERFECT MATCH: {self.reference_data['marker_name']} ({test_name}) ({max_rel_error:.4f}% < {tolerance}%)"
            )
        else:
            print(
                f"\n❌ MISMATCH: {self.reference_data['marker_name']} ({test_name}) ({max_rel_error:.4f}% error)"
            )
            # Show worst case
            worst_idx = np.unravel_index(np.argmax(rel_error), rel_error.shape)
            print(f"\nWorst case at index {worst_idx}:")
            print(
                f"  Manual (NICE) value: {manually_aggregated[worst_idx]:.6e}"
            )
            print(f"  Junifer value:       {junifer_output[worst_idx]:.6e}")
            print(f"  Relative diff:       {rel_error[worst_idx]:.4f}%")

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


class TestPermutationEntropyAggregated:
    """Test Permutation Entropy marker with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for Permutation Entropy marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = reference_dir / "permutation_entropy_reference.pkl"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print(
            "LOADED REFERENCE DATA FOR PERMUTATION ENTROPY (AGGREGATED TESTS)"
        )
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois="first_half",
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois="first_half",
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois="first_half",
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois="first_half",
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run Permutation Entropy aggregation test.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"={'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Add aggregation parameters
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        junifer_params["rois"] = rois

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        # Convert old PermutationEntropy parameters to new PermutationEntropyROIs format

        new_params = convert_permutation_entropy_params(junifer_params)

        junifer_marker = PermutationEntropyROIs(**new_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print(
            "\nRunning Junifer Permutation Entropy marker with aggregation..."
        )
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["permutationentropy"]["data"]
        print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (raw, non-aggregated)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE raw output shape: {nice_output.shape}")

        # Step 6: Compare using manual aggregation
        tolerance = self.reference_data["comparison_tolerance"]
        all_channels = epochs_data["info"]["ch_names"]
        match = check_aggregated_equivalence(
            nice_output,
            junifer_output,
            f"{self.reference_data['marker_name']} ({test_name})",
            channel_agg,
            trial_agg,
            roi_channels=rois,
            all_channels=all_channels,
            tolerance=tolerance,
        )

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


class TestSymbolicMutualInformationAggregated:
    """Test Symbolic Mutual Information marker with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for Symbolic Mutual Information marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = (
            reference_dir / "symbolic_mutual_information_reference.pkl"
        )

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print(
            "LOADED REFERENCE DATA FOR SYMBOLIC MUTUAL INFORMATION (AGGREGATED TESTS)"
        )
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        # NOTE: For connectivity markers, ROI filtering changes the computation
        # (filters BEFORE computing connectivity), so we test on all channels
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois=None,
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        # NOTE: For connectivity markers, ROI filtering changes the computation
        # (filters BEFORE computing connectivity), so we test on all channels
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois=None,
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        # NOTE: For connectivity markers, ROI filtering changes the computation
        # (filters BEFORE computing connectivity), so we test on all channels
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois=None,
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        # NOTE: For connectivity markers, ROI filtering changes the computation
        # (filters BEFORE computing connectivity), so we test on all channels
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois=None,
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run SymbolicMutualInformation aggregation test.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"={'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Remove deprecated parameters from old marker
        junifer_params.pop("average", None)
        junifer_params.pop("channel_method", None)
        junifer_params.pop("trial_method", None)
        junifer_params.pop("connectivity_aggregation_method", None)
        junifer_params.pop("rois", None)  # Not in new marker

        # Add aggregation parameters for new refactored marker
        # For connectivity markers, we need to first aggregate the connectivity dimension
        # to convert (channels x channels x trials) to (channels x trials)
        junifer_params["connectivity_method"] = "mean"
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        # Note: rois parameter removed - not used in these tests

        # Convert old tau parameter to new taus parameter for refactored marker
        if "tau" in junifer_params:
            junifer_params["taus"] = junifer_params.pop("tau")

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        junifer_marker = SymbolicMutualInformationROIs(**junifer_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print(
            "\nRunning Junifer Symbolic Mutual Information marker with aggregation..."
        )
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["symbolicmutualinformation"]["data"]
        # Handle scalar outputs properly
        if np.isscalar(junifer_output):
            print(
                f"Junifer aggregated output shape: scalar ({junifer_output})"
            )
        else:
            print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (connectivity matrix: channels x channels x trials)
        nice_output_connectivity = self.reference_data["nice_output"]
        print(
            f"NICE raw output shape (connectivity matrix): {nice_output_connectivity.shape}"
        )

        # Step 6: Compare using connectivity-specific aggregation function
        tolerance = self.reference_data["comparison_tolerance"]
        match = check_connectivity_aggregated_equivalence(
            nice_output_connectivity,
            junifer_output,
            f"{self.reference_data['marker_name']} ({test_name})",
            channel_agg,
            trial_agg,
            tolerance=tolerance,
        )

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


class TestPowerSpectralDensitySummaryAggregated:
    """Test PowerSpectralDensitySummary marker with aggregation."""

    @classmethod
    def setup_class(cls):
        """Load reference data for PowerSpectralDensitySummary marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = reference_dir / "psd_summary_msf_reference.pkl"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print("LOADED REFERENCE DATA FOR PSD SUMMARY MSF (AGGREGATED TESTS)")
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_aggregation_mean_mean(self):
        """Test with mean aggregation for both channels and trials."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg="mean",
            test_name="mean_mean",
            rois="first_half",
        )

    def test_aggregation_trim_mean80_trim_mean80(self):
        """Test with trim_mean80 aggregation (NICE standard)."""
        self._run_aggregation_test(
            channel_agg="trim_mean80",
            trial_agg="trim_mean80",
            test_name="trim_mean80_trim_mean80",
            rois="first_half",
        )

    def test_aggregation_mean_only(self):
        """Test with only trial aggregation (mean), no channel aggregation."""
        self._run_aggregation_test(
            channel_agg=None,
            trial_agg="mean",
            test_name="none_mean",
            rois="first_half",
        )

    def test_aggregation_channel_only(self):
        """Test with only channel aggregation (mean), no trial aggregation."""
        self._run_aggregation_test(
            channel_agg="mean",
            trial_agg=None,
            test_name="mean_none",
            rois="first_half",
        )

    def _run_aggregation_test(
        self,
        channel_agg: str | None,
        trial_agg: str | None,
        test_name: str,
        rois: list[str] | None = None,
    ):
        """Helper method to run PowerSpectralDensitySummary aggregation test.

        Parameters
        ----------
        channel_agg : str or None
            Channel aggregation method
        trial_agg : str or None
            Trial aggregation method
        test_name : str
            Name for this specific test case
        rois : list of str or None
            ROI specification to group channels for aggregation.
            If None, each channel is treated as a separate ROI.
        """
        print(f"\n{'=' * 80}")
        print(f"TESTING: {test_name}")
        print(f"={'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Handle ROI selection - use actual epochs channel names
        if rois == "first_half":
            # Use first 16 channels from actual epochs
            rois = epochs.ch_names[:16]  # First 16 out of 32 channels

        # Step 2: Get base Junifer parameters and add aggregation
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        # Add aggregation parameters
        junifer_params["channel_method"] = channel_agg
        junifer_params["trial_method"] = trial_agg
        junifer_params["rois"] = rois

        print("\nJunifer parameters (WITH aggregation):")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create Junifer marker with aggregation parameters
        junifer_marker = PowerSpectralDensitySummary(**junifer_params)

        # Step 4: Run Junifer marker
        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print(
            "\nRunning Junifer PowerSpectralDensitySummary marker with aggregation..."
        )
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["psdsummary"]["data"]
        # Handle scalar outputs properly
        if np.isscalar(junifer_output):
            print(
                f"Junifer aggregated output shape: scalar ({junifer_output})"
            )
        else:
            print(f"Junifer aggregated output shape: {junifer_output.shape}")

        # Step 5: Get NICE reference output (raw, non-aggregated)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE raw output shape: {nice_output.shape}")

        # Step 6: Compare using manual aggregation
        tolerance = self.reference_data["comparison_tolerance"]
        all_channels = epochs_data["info"]["ch_names"]
        match = check_aggregated_equivalence(
            nice_output,
            junifer_output,
            f"{self.reference_data['marker_name']} ({test_name})",
            channel_agg,
            trial_agg,
            roi_channels=rois,
            all_channels=all_channels,
            tolerance=tolerance,
        )

        # Step 7: Assert for pytest
        assert match, (
            f"Aggregated Junifer output does not match manually aggregated NICE "
            f"reference within {tolerance}% tolerance for {test_name}"
        )


class TestTimeLockedContrastAggregated:
    """Test TimeLockedContrast marker (already aggregated in reference)."""

    @classmethod
    def setup_class(cls):
        """Load reference data for TimeLockedContrast marker."""
        reference_dir = Path(__file__).parent / "reference_data"
        reference_file = (
            reference_dir / "time_locked_contrast_ld_ls_reference.pkl"
        )

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run generate_reference_data.py first."
            )

        with open(reference_file, "rb") as f:
            cls.reference_data = pickle.load(f)

        print("\n" + "=" * 80)
        print("LOADED REFERENCE DATA FOR TIME-LOCKED CONTRAST LD_LS")
        print("=" * 80)
        print(f"Marker: {cls.reference_data['marker_name']}")
        print(
            f"NICE raw output shape: {cls.reference_data['nice_output_shape']}"
        )

    def test_contrast_aggregated(self):
        """Test contrast marker (reference already contains aggregated scalar)."""
        print(f"\n{'=' * 80}")
        print("TESTING: TimeLockedContrast (LD vs LS)")
        print(f"={'=' * 80}")

        # Step 1: Recreate epochs from stored data
        epochs_data = self.reference_data["epochs_data"]
        epochs = mne.EpochsArray(
            epochs_data["data"],
            epochs_data["info"],
            events=epochs_data["events"],
            tmin=epochs_data["tmin"],
            verbose=False,
        )

        # Step 2: Create Junifer marker with reference parameters
        junifer_params = convert_legacy_params(
            self.reference_data["junifer_params"]
        )

        print("\nJunifer parameters:")
        for key, value in junifer_params.items():
            print(f"  {key}: {value}")

        # Step 3: Create and run Junifer marker
        junifer_marker = TimeLockedContrast(**junifer_params)

        input_dict = {
            "data": epochs,
            "meta": {
                "subject": "test",
                "session": "01",
                "task": "test",
                "run": "01",
            },
        }
        print("\nRunning Junifer TimeLockedContrast marker...")
        junifer_result = junifer_marker.compute(input_dict)
        junifer_output = junifer_result["timelockedcontrast"]["data"]
        print(f"Junifer output shape: {junifer_output.shape}")
        print(f"Junifer output value: {junifer_output}")

        # Step 4: Get NICE reference output (already aggregated scalar)
        nice_output = self.reference_data["nice_output"]
        print(f"NICE output value: {nice_output}")

        # Step 5: Compare values
        tolerance = self.reference_data["comparison_tolerance"]

        # Handle scalar comparison
        junifer_value = float(junifer_output.flat[0])
        nice_value = (
            float(nice_output)
            if np.ndim(nice_output) == 0
            else float(nice_output.flat[0])
        )

        abs_diff = abs(junifer_value - nice_value)
        rel_error = (
            (abs_diff / abs(nice_value)) * 100 if nice_value != 0 else 0
        )

        print("\nValue comparison:")
        print(f"  NICE value:     {nice_value:.6e}")
        print(f"  Junifer value:  {junifer_value:.6e}")
        print(f"  Absolute diff:  {abs_diff:.6e}")
        print(f"  Relative error: {rel_error:.4f}%")

        match = rel_error < tolerance

        if match:
            print(f"\n✅ PERFECT MATCH: {rel_error:.4f}% < {tolerance}%")
        else:
            print(f"\n❌ MISMATCH: {rel_error:.4f}% >= {tolerance}%")

        # Step 6: Assert for pytest
        assert match, (
            f"TimeLockedContrast output does not match NICE reference "
            f"within {tolerance}% tolerance (error: {rel_error:.4f}%)"
        )
