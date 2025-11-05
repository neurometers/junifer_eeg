"""Validation tests for ICM preprocessing pipeline against ground truth data.

This module tests the ICM preprocessing pipeline by comparing its output
against ground truth data from subject PA301 (cropped to 5% for GitHub). Tests verify:
1. Bad channels detection matches ground truth
2. Bad epochs detection matches ground truth
3. Final preprocessed epochs match ground truth exactly (0% error)

Ground truth files (cropped versions):
- Raw EEG data: sub-PA301_ses-01_task-lg_acq-01_run-01_eeg_cropped.fif
- Preprocessing info: sub-PA301_ses-01_task-lg_acq-01_preprocess_cropped.json
- Final epochs: sub-PA301_ses-01_task-lg_acq-01_epo_cropped.fif
"""

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from junifer_eeg.preprocessors.icm_lg_preprocessing import (
    ICMAdaptiveArtifactRejection,
    ICMEquipmentFilter,
    ICMLGEpoching,
)

# Ground truth data paths (using cropped data for GitHub size limits)
GROUND_TRUTH_DIR = Path(__file__).parent / "reference_data" / "preprocessing"
RAW_DATA_PATH = (
    GROUND_TRUTH_DIR / "sub-PA301_ses-01_task-lg_acq-01_run-01_eeg_raw.fif"
)
PREPROCESS_JSON_PATH = (
    GROUND_TRUTH_DIR
    / "sub-PA301_ses-01_task-lg_acq-01_preprocess_cropped.json"
)
GROUND_TRUTH_EPOCHS_PATH = (
    GROUND_TRUTH_DIR / "sub-PA301_ses-01_task-lg_acq-01_epo.fif"
)


@pytest.fixture(scope="module")
def ground_truth_preprocess_info():
    """Load ground truth preprocessing info from JSON."""
    if not PREPROCESS_JSON_PATH.exists():
        pytest.skip(f"Ground truth JSON not found: {PREPROCESS_JSON_PATH}")

    with open(PREPROCESS_JSON_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ground_truth_epochs():
    """Load ground truth final epochs."""
    if not GROUND_TRUTH_EPOCHS_PATH.exists():
        pytest.skip(
            f"Ground truth epochs not found: {GROUND_TRUTH_EPOCHS_PATH}"
        )

    return mne.read_epochs(
        GROUND_TRUTH_EPOCHS_PATH, preload=True, verbose=False
    )


@pytest.fixture(scope="module")
def preprocessed_epochs():
    """Run ICM preprocessing pipeline and return final epochs."""
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")

    # Load raw data (cropped FIF file already has triggers processed)
    raw = mne.io.read_raw_fif(RAW_DATA_PATH, preload=True, verbose=False)

    # Create preprocessing input dict
    input_dict = {
        "data": raw,
        "meta": {
            "subject": "PA301",
            "session": "01",
            "task": "lg",
            "run": "01",
        },
    }

    # Step 1: ICMEquipmentFilter
    equipment_filter = ICMEquipmentFilter(equipment_type="egi")
    input_dict, _ = equipment_filter.preprocess(input_dict)

    # Step 2: ICMLGEpoching
    epoching = ICMLGEpoching(
        tmin=-0.2,
        tmax=1.34,
        baseline=[-0.2, 0.0],
        event_id={
            "HSTD": 10,
            "HDVT": 20,
            "LSGS": 30,
            "LSGD": 40,
            "LDGD": 50,
            "LDGS": 60,
        },
    )
    input_dict, _ = epoching.preprocess(input_dict)

    # Step 3: ICMAdaptiveArtifactRejection
    artifact_rejection = ICMAdaptiveArtifactRejection(
        zscore_thresh=4.0,
        min_channels=0.5,
        min_events=0.1,
    )
    input_dict, _ = artifact_rejection.preprocess(input_dict)

    # Return final epochs and artifact rejection info
    return {
        "epochs": input_dict["data"],
        "artifact_rejection": artifact_rejection,
    }


class TestPreprocessingBadChannels:
    """Test bad channel detection against ground truth."""

    def test_bad_channels_match(
        self, preprocessed_epochs, ground_truth_preprocess_info
    ):
        """Verify detected bad channels match ground truth exactly."""
        # Get detected bad channels from preprocessing info
        epochs = preprocessed_epochs["epochs"]
        if (
            "temp" in epochs.info
            and "preprocessing_info" in epochs.info["temp"]
        ):
            detected_bad_channels = set(
                epochs.info["temp"]["preprocessing_info"][
                    "bad_channels_detected"
                ]
            )
        else:
            raise ValueError("Preprocessing info not found in epochs.info")

        # Get ground truth bad channels
        gt_bad_channels = set(ground_truth_preprocess_info["bad_channels"])

        # Assert exact match
        assert detected_bad_channels == gt_bad_channels, (
            f"Bad channels mismatch!\n"
            f"Expected: {sorted(gt_bad_channels)}\n"
            f"Detected: {sorted(detected_bad_channels)}"
        )


class TestPreprocessingBadEpochs:
    """Test bad epoch detection against ground truth."""

    def test_bad_epochs_match(
        self, preprocessed_epochs, ground_truth_preprocess_info
    ):
        """Verify detected bad epochs match ground truth exactly."""
        # Get detected bad epochs from preprocessing info stored in epochs
        epochs = preprocessed_epochs["epochs"]

        # Get the bad epochs from the preprocessing info stored in epochs.info
        if (
            "temp" in epochs.info
            and "preprocessing_info" in epochs.info["temp"]
        ):
            detected_bad_epochs = set(
                epochs.info["temp"]["preprocessing_info"][
                    "bad_epochs_detected"
                ]
            )
        else:
            raise ValueError("Preprocessing info not found in epochs.info")

        # Get ground truth bad epochs
        gt_bad_epochs = set(ground_truth_preprocess_info["bad_epochs"])

        # Assert exact match
        assert detected_bad_epochs == gt_bad_epochs, (
            f"Bad epochs mismatch!\n"
            f"Expected: {sorted(gt_bad_epochs)}\n"
            f"Detected: {sorted(detected_bad_epochs)}"
        )


class TestPreprocessingFinalEpochs:
    """Test final preprocessed epochs against ground truth."""

    def test_epochs_data_exact_match(
        self, preprocessed_epochs, ground_truth_epochs
    ):
        """Verify final epochs data matches ground truth exactly (0% error) for good channels only."""
        # Get preprocessed and ground truth epochs
        preprocessed_epochs_obj = preprocessed_epochs["epochs"]

        # Get bad channels that were detected and interpolated
        if (
            "temp" in preprocessed_epochs_obj.info
            and "preprocessing_info" in preprocessed_epochs_obj.info["temp"]
        ):
            bad_channels_detected = set(
                preprocessed_epochs_obj.info["temp"]["preprocessing_info"][
                    "bad_channels_detected"
                ]
            )
        else:
            bad_channels_detected = set()

        # Get data arrays
        preprocessed_data = preprocessed_epochs_obj.get_data()
        ground_truth_data = ground_truth_epochs.get_data()

        # Get channel info
        preprocessed_ch_names = preprocessed_epochs_obj.ch_names
        ground_truth_ch_names = ground_truth_epochs.ch_names

        # Find ONLY good channels (excluding the 22 bad/interpolated channels)
        good_channels = [
            ch
            for ch in preprocessed_ch_names
            if ch in ground_truth_ch_names and ch not in bad_channels_detected
        ]

        # Get indices for good channels only
        preprocessed_ch_indices = [
            preprocessed_ch_names.index(ch) for ch in good_channels
        ]
        ground_truth_ch_indices = [
            ground_truth_ch_names.index(ch) for ch in good_channels
        ]

        # Compare data ONLY for good channels (not interpolated)
        preprocessed_data_subset = preprocessed_data[
            :, preprocessed_ch_indices, :
        ]
        ground_truth_data_subset = ground_truth_data[
            :, ground_truth_ch_indices, :
        ]

        # Compute differences
        abs_diff = np.abs(preprocessed_data_subset - ground_truth_data_subset)
        mean_abs_diff = np.mean(abs_diff)

        # Compute relative error (avoiding division by zero)
        ground_truth_nonzero = np.abs(ground_truth_data_subset) > 1e-20
        rel_diff = np.zeros_like(abs_diff)
        rel_diff[ground_truth_nonzero] = abs_diff[
            ground_truth_nonzero
        ] / np.abs(ground_truth_data_subset[ground_truth_nonzero])
        mean_rel_diff = np.mean(rel_diff) * 100  # Convert to percentage

        # For good channels only (not interpolated), expect near-exact match
        mean_rel_tolerance = 0.01  # 0.01% mean relative error threshold

        # Assert that mean relative error is within strict tolerance
        assert mean_rel_diff < mean_rel_tolerance, (
            f"Data mismatch exceeds {mean_rel_tolerance}% tolerance for good channels!\n"
            f"Mean absolute error: {mean_abs_diff:.2e} V\n"
            f"Mean relative error: {mean_rel_diff:.6f}% (threshold: {mean_rel_tolerance}%)\n"
            f"This should be near-zero for good channels (not interpolated)."
        )


class TestPreprocessingEventCounts:
    """Test event counts and types match ground truth."""

    def test_event_counts_match(
        self, preprocessed_epochs, ground_truth_epochs
    ):
        """Verify event counts per condition match ground truth."""
        preprocessed = preprocessed_epochs["epochs"]
        ground_truth = ground_truth_epochs

        # Get event counts from preprocessed epochs
        preprocessed_event_id = preprocessed.event_id
        preprocessed_events = preprocessed.events

        # Get event counts from ground truth epochs
        ground_truth_event_id = ground_truth.event_id
        ground_truth_events = ground_truth.events

        # Assert event types match
        assert set(preprocessed_event_id.keys()) == set(
            ground_truth_event_id.keys()
        ), (
            f"Event types mismatch!\n"
            f"Preprocessed: {sorted(preprocessed_event_id.keys())}\n"
            f"Ground truth: {sorted(ground_truth_event_id.keys())}"
        )

        # Assert event counts match for each condition
        for event_name in preprocessed_event_id.keys():
            preprocessed_code = preprocessed_event_id[event_name]
            ground_truth_code = ground_truth_event_id[event_name]

            preprocessed_count = np.sum(
                preprocessed_events[:, 2] == preprocessed_code
            )
            ground_truth_count = np.sum(
                ground_truth_events[:, 2] == ground_truth_code
            )

            assert preprocessed_count == ground_truth_count, (
                f"Event count mismatch for {event_name}!\n"
                f"Preprocessed: {preprocessed_count}\n"
                f"Ground truth: {ground_truth_count}"
            )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
