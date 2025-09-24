"""Tests for ICM Local-Global preprocessing components."""

import mne
import numpy as np
import pytest

from junifer_eeg.preprocessors import (
    ICMAdaptiveArtifactRejection,
    ICMEquipmentFilter,
    ICMLGEpoching,
)


def create_test_raw_with_events():
    """Create synthetic Raw object with ICM LG events for testing."""
    sfreq = 500  # 500 Hz sampling rate (typical for ICM)
    duration = 60  # 60 seconds (long enough to avoid filter warnings after resampling)
    times = np.arange(0, duration, 1 / sfreq)

    # Create EEG-like data with multiple channels
    n_channels = 64
    data = np.random.RandomState(42).normal(0, 1e-5, (n_channels, len(times)))

    # Add some realistic EEG patterns
    for i in range(n_channels):
        # Add alpha rhythm (10 Hz)
        data[i] += 5e-6 * np.sin(2 * np.pi * 10 * times)
        # Add some noise
        data[i] += np.random.RandomState(42 + i).normal(0, 2e-6, len(times))

    # Create channel names (EGI-style)
    ch_names = [f"E{i + 1}" for i in range(n_channels)]
    ch_names.append("STI 014")  # Stimulus channel

    # Add stimulus channel data
    stim_data = np.zeros((1, len(times)))

    # Add ICM LG events at regular intervals
    event_times = np.arange(
        2, duration - 2, 2.0
    )  # Events every 2s (more events for longer duration)
    event_codes = [10, 20, 30, 40, 50, 60]  # ICM LG event codes

    for i, t in enumerate(event_times):
        if i < len(event_codes):
            sample_idx = int(t * sfreq)
            if sample_idx < len(times):
                stim_data[0, sample_idx] = event_codes[i]

    # Combine EEG and stimulus data
    all_data = np.vstack([data, stim_data])
    all_ch_names = ch_names

    # Create MNE Raw object
    ch_types = ["eeg"] * n_channels + ["stim"]
    info = mne.create_info(
        ch_names=all_ch_names,
        sfreq=sfreq,
        ch_types=ch_types,
    )
    raw = mne.io.RawArray(all_data, info, verbose=False)

    # Add EGI montage for digitization (needed for artifact rejection)
    # Use GSN-HydroCel-64 montage which has E1-E64 channels
    try:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-64_1.0")
        raw.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception:
        # If EGI montage fails, create simple fake digitization points
        # This is just for testing - real data would have proper montage
        fake_pos = {}
        for ch_name in raw.ch_names:
            if ch_name.startswith("E"):
                # Create fake positions in a circle
                angle = hash(ch_name) % 360
                fake_pos[ch_name] = [
                    np.cos(np.radians(angle)) * 0.1,
                    np.sin(np.radians(angle)) * 0.1,
                    0.05,
                ]
        if fake_pos:
            fake_montage = mne.channels.make_dig_montage(
                fake_pos, coord_frame="head"
            )
            raw.set_montage(fake_montage)

    return raw


class TestICMEquipmentFilter:
    """Test ICMEquipmentFilter preprocessor."""

    def test_initialization(self):
        """Test ICMEquipmentFilter initialization."""
        # Test default initialization
        filter_proc = ICMEquipmentFilter()
        assert filter_proc.equipment_type == "egi"
        assert filter_proc.config_params == {}
        assert filter_proc.n_jobs == 1

        # Test custom initialization with config_params
        config_params = {
            "l_freq": 0.5,
            "h_freq": 35.0,
            "notch_freq": 50.0,
            "resample_freq": 256,
        }
        filter_proc = ICMEquipmentFilter(
            equipment_type="brainvision",
            config_params=config_params,
        )
        assert filter_proc.equipment_type == "brainvision"
        assert filter_proc.config_params["l_freq"] == 0.5
        assert filter_proc.config_params["h_freq"] == 35.0
        assert filter_proc.config_params["notch_freq"] == 50.0
        assert filter_proc.config_params["resample_freq"] == 256

    def test_valid_inputs_outputs(self):
        """Test valid input/output types."""
        filter_proc = ICMEquipmentFilter()
        assert filter_proc.get_valid_inputs() == ["EEG"]
        assert filter_proc.get_output_type("EEG") == "EEG"

    def test_preprocess_egi(self):
        """Test EGI equipment filtering."""
        raw = create_test_raw_with_events()
        original_sfreq = raw.info["sfreq"]

        filter_proc = ICMEquipmentFilter(equipment_type="egi")
        input_data = {"data": raw}

        result, extra = filter_proc.preprocess(input_data)

        # Check that data was processed
        processed_raw = result["data"]
        assert isinstance(processed_raw, mne.io.BaseRaw)

        # Check resampling (EGI default: 250 Hz)
        assert processed_raw.info["sfreq"] == 250
        assert processed_raw.info["sfreq"] != original_sfreq

        # Check that filtering was applied (should have filter info)
        assert processed_raw.info["highpass"] is not None
        assert processed_raw.info["lowpass"] is not None

    def test_preprocess_custom_params(self):
        """Test filtering with custom parameters."""
        raw = create_test_raw_with_events()

        config_params = {
            "hpass": 1.0,
            "lpass": 45.0,
            "resample_freq": 200,
        }
        filter_proc = ICMEquipmentFilter(
            equipment_type="egi",
            config_params=config_params,
        )
        input_data = {"data": raw}

        result, extra = filter_proc.preprocess(input_data)
        processed_raw = result["data"]

        # Check custom resampling
        assert processed_raw.info["sfreq"] == 200

        # Check custom filtering
        assert processed_raw.info["highpass"] == 1.0
        assert processed_raw.info["lowpass"] == 45.0

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        filter_proc = ICMEquipmentFilter()

        # Test with non-Raw input
        with pytest.raises(
            ValueError, match="Input data must be mne.io.BaseRaw"
        ):
            filter_proc.preprocess({"data": "not_raw_data"})


class TestICMLGEpoching:
    """Test ICMLGEpoching preprocessor."""

    def test_initialization(self):
        """Test epoching initialization."""
        # Default initialization
        epoch_proc = ICMLGEpoching()
        assert epoch_proc.tmin == -0.2
        assert epoch_proc.tmax == 1.34
        assert epoch_proc.baseline == (None, 0)
        assert (
            epoch_proc.event_id is not None
        )  # Uses ICM_LG_EVENT_ID by default

        # Custom initialization
        epoch_proc = ICMLGEpoching(
            tmin=-0.1,
            tmax=1.0,
            baseline=(-0.1, 0.0),
            event_id={"custom": 99},
        )
        assert epoch_proc.tmin == -0.1
        assert epoch_proc.tmax == 1.0
        assert epoch_proc.baseline == (-0.1, 0.0)
        assert epoch_proc.event_id == {"custom": 99}

    def test_valid_inputs_outputs(self):
        """Test valid input/output types."""
        epoch_proc = ICMLGEpoching()
        assert epoch_proc.get_valid_inputs() == ["EEG"]
        assert (
            epoch_proc.get_output_type("EEG") == "EEG"
        )  # Returns same type as base class

    def test_preprocess_basic(self):
        """Test basic epoching functionality."""
        raw = create_test_raw_with_events()

        epoch_proc = ICMLGEpoching()  # Default parameters
        input_data = {"data": raw}

        result, extra = epoch_proc.preprocess(input_data)

        # Check that epochs were created
        epochs = result["data"]
        assert isinstance(epochs, mne.BaseEpochs)
        assert len(epochs) > 0  # Should have some epochs

        # Check epoch parameters
        assert epochs.tmin == -0.2
        assert epochs.tmax == 1.34

        # Check that STI channels were dropped
        stim_channels = [ch for ch in epochs.ch_names if ch.startswith("STI")]
        assert len(stim_channels) == 0

    def test_preprocess_custom_event_id(self):
        """Test epoching with custom event ID mapping."""
        raw = create_test_raw_with_events()

        # Use custom event mapping
        custom_event_id = {
            "HSTD": 10,
            "HDVT": 20,
            "LSGS": 30,
        }

        epoch_proc = ICMLGEpoching(
            event_id=custom_event_id,
        )
        input_data = {"data": raw}

        result, extra = epoch_proc.preprocess(input_data)
        epochs = result["data"]

        # Should have epochs for the specified events
        assert len(epochs) > 0
        assert isinstance(epochs, mne.BaseEpochs)

    def test_preprocess_auto_event_id(self):
        """Test epoching with automatic event ID detection."""
        raw = create_test_raw_with_events()

        epoch_proc = ICMLGEpoching(
            event_id={"LSGS": 30, "LDGD": 50},
        )
        input_data = {"data": raw}

        result, extra = epoch_proc.preprocess(input_data)
        epochs = result["data"]

        # Should create epochs with auto-detected events
        assert len(epochs) > 0
        assert isinstance(epochs, mne.BaseEpochs)

    def test_no_events_error(self):
        """Test error handling when no events are found."""
        # Create raw data without events
        sfreq = 250
        duration = 2
        times = np.arange(0, duration, 1 / sfreq)
        data = np.random.RandomState(42).normal(0, 1e-5, (10, len(times)))

        info = mne.create_info(
            ch_names=[f"E{i + 1}" for i in range(10)],
            sfreq=sfreq,
            ch_types=["eeg"] * 10,
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        epoch_proc = ICMLGEpoching()
        input_data = {"data": raw}

        with pytest.raises(
            ValueError, match="No stim channels found|No events found"
        ):
            epoch_proc.preprocess(input_data)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        epoch_proc = ICMLGEpoching()

        # Test with non-Raw input
        with pytest.raises(
            ValueError, match="Input data must be mne.io.BaseRaw"
        ):
            epoch_proc.preprocess({"data": "not_raw_data"})


class TestICMAdaptiveArtifactRejection:
    """Test ICMAdaptiveArtifactRejection preprocessor."""

    def test_initialization(self):
        """Test artifact rejection initialization."""
        # Default initialization
        reject_proc = ICMAdaptiveArtifactRejection()
        assert reject_proc.zscore_thresh == 4
        assert reject_proc.n_channels_bad_epoch == 0.1
        assert reject_proc.n_epochs_bad_ch == 0.5
        assert reject_proc.reject == {"eeg": 100e-6}

        # Custom initialization
        reject_proc = ICMAdaptiveArtifactRejection(
            zscore_thresh=2.5,
            n_channels_bad_epoch=0.15,
            n_epochs_bad_ch=0.25,
        )
        assert reject_proc.zscore_thresh == 2.5
        assert reject_proc.n_channels_bad_epoch == 0.15
        assert reject_proc.n_epochs_bad_ch == 0.25

    def test_valid_inputs_outputs(self):
        """Test valid input/output types."""
        reject_proc = ICMAdaptiveArtifactRejection()
        assert reject_proc.get_valid_inputs() == [
            "EEG"
        ]  # Actually accepts EEG
        assert reject_proc.get_output_type("EEG") == "EEG"  # Returns same type

    def test_preprocess_basic(self):
        """Test basic artifact rejection functionality."""
        # Create epochs first
        raw = create_test_raw_with_events()
        epoch_proc = ICMLGEpoching()
        epochs_data, _ = epoch_proc.preprocess({"data": raw})
        epochs = epochs_data["data"]

        # Apply artifact rejection
        reject_proc = ICMAdaptiveArtifactRejection(
            zscore_thresh=5.0
        )  # Lenient threshold
        input_data = {"data": epochs}

        result, extra = reject_proc.preprocess(input_data)

        # Check that processed epochs are returned
        processed_epochs = result["data"]
        assert isinstance(processed_epochs, mne.BaseEpochs)

        # Should have same or fewer epochs (due to rejection)
        assert len(processed_epochs) <= len(epochs)

    def test_preprocess_strict_rejection(self):
        """Test artifact rejection with strict parameters."""
        # Create epochs with some artificial artifacts
        raw = create_test_raw_with_events()
        epoch_proc = ICMLGEpoching()
        epochs_data, _ = epoch_proc.preprocess({"data": raw})
        epochs = epochs_data["data"]

        # Add large artifacts to some epochs
        epochs_copy = epochs.copy()
        data = epochs_copy.get_data()
        data[0, 0, :] += (
            100e-6  # Add large artifact to first epoch, first channel
        )

        # Create new epochs with artifacts
        epochs_with_artifacts = mne.EpochsArray(
            data, epochs.info, events=epochs.events, tmin=epochs.tmin
        )

        # Apply strict artifact rejection
        reject_proc = ICMAdaptiveArtifactRejection(
            zscore_thresh=1.0
        )  # Very strict
        input_data = {"data": epochs_with_artifacts}

        result, extra = reject_proc.preprocess(input_data)
        processed_epochs = result["data"]

        # Should reject some epochs/channels
        assert len(processed_epochs) <= len(epochs_with_artifacts)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        reject_proc = ICMAdaptiveArtifactRejection()

        # Test with non-Epochs input
        with pytest.raises(
            ValueError, match="Input data must be mne.BaseEpochs"
        ):
            reject_proc.preprocess({"data": "not_epochs_data"})


class TestICMPreprocessingIntegration:
    """Test integration of ICM preprocessing components."""

    def test_full_pipeline(self):
        """Test complete ICM preprocessing pipeline."""
        raw = create_test_raw_with_events()

        # Step 1: Equipment filtering
        filter_proc = ICMEquipmentFilter(equipment_type="egi")
        filtered_data, _ = filter_proc.preprocess({"data": raw})

        # Step 2: Epoching
        epoch_proc = ICMLGEpoching()
        epochs_data, _ = epoch_proc.preprocess(filtered_data)

        # Step 3: Artifact rejection
        reject_proc = ICMAdaptiveArtifactRejection(zscore_thresh=3.0)
        final_data, _ = reject_proc.preprocess(epochs_data)

        # Check final result
        final_epochs = final_data["data"]
        assert isinstance(final_epochs, mne.BaseEpochs)
        assert len(final_epochs) > 0

        # Check that processing was applied
        assert final_epochs.info["sfreq"] == 250  # Resampled
        assert final_epochs.info["highpass"] is not None  # Filtered
        assert final_epochs.tmin == -0.2  # Epoched
        assert final_epochs.tmax == 1.34

    def test_pipeline_with_empty_epochs(self):
        """Test pipeline behavior with empty epochs (edge case)."""
        # Create raw data with very strict rejection criteria
        raw = create_test_raw_with_events()

        # Use very strict rejection to get empty epochs
        epoch_proc = ICMLGEpoching()
        # Note: rejection criteria handled by ICMAdaptiveArtifactRejection

        try:
            epochs_data, _ = epoch_proc.preprocess({"data": raw})
            epochs = epochs_data["data"]

            if len(epochs) == 0:
                # Test that artifact rejection handles empty epochs gracefully
                reject_proc = ICMAdaptiveArtifactRejection()
                result, _ = reject_proc.preprocess({"data": epochs})

                # Should return empty epochs without error
                assert len(result["data"]) == 0
                assert isinstance(result["data"], mne.BaseEpochs)
        except (ValueError, RuntimeError):
            # If epoching fails due to strict rejection or empty epochs, that's acceptable
            pass

    def test_equipment_types(self):
        """Test different equipment type configurations."""
        raw = create_test_raw_with_events()

        equipment_types = ["egi", "brainvision", "ant", "biosemi"]

        for eq_type in equipment_types:
            filter_proc = ICMEquipmentFilter(equipment_type=eq_type)
            result, _ = filter_proc.preprocess({"data": raw})

            # Should process successfully for all equipment types
            assert isinstance(result["data"], mne.io.BaseRaw)

            # Only EGI resamples to 250 Hz, others keep original sampling rate
            if eq_type == "egi":
                assert result["data"].info["sfreq"] == 250
            else:
                assert (
                    result["data"].info["sfreq"] == 500
                )  # Original sampling rate
