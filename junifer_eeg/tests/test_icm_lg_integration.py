"""Integration tests for ICM LG components."""

import mne
import numpy as np
import pytest

from junifer_eeg.datareader import ICMLGDataReader
from junifer_eeg.markers import (
    ICMLGContrast,
    ICMLGGlobalLocalIndex,
    ICMLGMismatchNegativity,
)
from junifer_eeg.preprocessors import (
    ICMAdaptiveArtifactRejection,
    ICMEquipmentFilter,
    ICMLGEpoching,
)


class TestICMLGIntegration:
    """Test ICM LG component integration."""

    def create_mock_icm_lg_raw(
        self,
        equipment="egi",
        n_channels=64,
        sfreq=250,
        duration=60,
    ):
        """Create mock ICM LG raw data with proper events."""
        # Create channel names
        if equipment == "egi":
            ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
        else:
            ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]

        # Add stimulus channel
        ch_names.append("STI 014")
        ch_types = ["eeg"] * n_channels + ["stim"]

        # Create info
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types,
        )

        # Create data
        n_samples = int(duration * sfreq)
        data = np.random.randn(n_channels + 1, n_samples) * 1e-6

        # Create ICM LG events in stimulus channel
        icm_events = {
            "HSTD": 10,
            "HDVT": 20,
            "LSGS": 30,
            "LSGD": 40,
            "LDGS": 60,
            "LDGD": 50,
        }

        # Add some events to stimulus channel
        stim_data = np.zeros(n_samples)
        event_times = np.arange(1000, n_samples - 2000, 2000)
        event_codes = list(icm_events.values())

        for i, time_idx in enumerate(event_times):
            if i < len(event_codes):
                stim_data[time_idx] = event_codes[i % len(event_codes)]

        data[-1, :] = stim_data

        # Create Raw object
        raw = mne.io.RawArray(data, info, verbose=False)

        return raw

    def test_datareader_basic_functionality(self):
        """Test basic functionality of ICM LG data reader."""
        reader = ICMLGDataReader()

        # Test that it can be instantiated
        assert reader is not None
        assert hasattr(reader, "_fit_transform")

        # Test equipment type setting
        reader_with_equipment = ICMLGDataReader(equipment_type="egi")
        assert reader_with_equipment.equipment_type == "egi"

    def test_equipment_filter_integration(self):
        """Test equipment-specific filtering."""
        # Create mock data
        raw = self.create_mock_icm_lg_raw(equipment="egi")

        # Create filter with correct parameter name
        filter_proc = ICMEquipmentFilter(equipment_type="egi")

        # Test filtering
        input_data = {"data": raw}
        result, _ = filter_proc.preprocess(input_data)

        # Check that result is a dictionary with 'data' key
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], mne.io.BaseRaw)

    def test_epoching_integration(self):
        """Test ICM LG epoching."""
        # Create mock data
        raw = self.create_mock_icm_lg_raw(equipment="egi")

        # Create epoching processor
        epoching = ICMLGEpoching(tmin=-0.2, tmax=1.34)

        # Test epoching
        input_data = {"data": raw}
        result, _ = epoching.preprocess(input_data)

        # Check that result is a dictionary with 'data' key
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], mne.Epochs)

    def test_adaptive_rejection_integration(self):
        """Test adaptive artifact rejection."""
        # Create mock epochs
        raw = self.create_mock_icm_lg_raw(equipment="egi")

        # Create simple epochs for testing
        events = np.array([[1000, 0, 10], [3000, 0, 20], [5000, 0, 30]])
        epochs = mne.Epochs(
            raw,
            events,
            {"HSTD": 10, "HDVT": 20, "LSGS": 30},
            tmin=-0.2,
            tmax=1.34,
            preload=True,
            baseline=None,
            verbose=False,
        )

        # Create adaptive rejection processor
        adaptive_rej = ICMAdaptiveArtifactRejection()

        # Test adaptive rejection
        input_data = {"data": epochs}
        result, extra_result = adaptive_rej.preprocess(input_data)

        # Check that result is a dictionary with 'data' key
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], mne.Epochs)

    def test_icm_lg_markers_integration(self):
        """Test ICM LG-specific markers."""
        # Create mock epochs with ICM LG events
        raw = self.create_mock_icm_lg_raw(equipment="egi")

        # Create events
        events = np.array(
            [
                [1000, 0, 10],  # HSTD
                [3000, 0, 20],  # HDVT
                [5000, 0, 30],  # LSGS
                [7000, 0, 40],  # LSGD
                [9000, 0, 60],  # LDGS
                [11000, 0, 50],  # LDGD
            ],
        )

        event_id = {
            "HSTD": 10,
            "HDVT": 20,
            "LSGS": 30,
            "LSGD": 40,
            "LDGS": 60,
            "LDGD": 50,
        }

        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=-0.2,
            tmax=1.34,
            preload=True,
            baseline=None,
            verbose=False,
        )

        # Test ICM LG Contrast marker with correct parameter name
        contrast_marker = ICMLGContrast()
        result = contrast_marker.compute({"data": epochs})
        assert isinstance(result, dict)
        assert len(result) > 0

        # Test ICM LG MMN marker
        mmn_marker = ICMLGMismatchNegativity()
        result = mmn_marker.compute({"data": epochs})
        assert isinstance(result, dict)
        assert len(result) > 0

        # Test ICM LG Global Local Index marker
        indices_marker = ICMLGGlobalLocalIndex()
        result = indices_marker.compute({"data": epochs})
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_full_pipeline_integration(self):
        """Test full ICM LG pipeline integration."""
        # Create mock data
        raw = self.create_mock_icm_lg_raw(equipment="egi")

        # Step 1: Equipment filtering with correct parameter name
        filter_proc = ICMEquipmentFilter(equipment_type="egi")
        filtered_result, _ = filter_proc.preprocess({"data": raw})
        filtered_raw = filtered_result["data"]

        # Step 2: Epoching
        epoching = ICMLGEpoching(tmin=-0.2, tmax=1.34)
        epoched_result, _ = epoching.preprocess({"data": filtered_raw})
        epochs = epoched_result["data"]

        # Step 3: Adaptive artifact rejection
        adaptive_rej = ICMAdaptiveArtifactRejection()
        cleaned_result, _ = adaptive_rej.preprocess({"data": epochs})
        cleaned_epochs = cleaned_result["data"]

        # Step 4: Marker computation
        contrast_marker = ICMLGContrast()
        contrast_result = contrast_marker.compute({"data": cleaned_epochs})

        # Verify pipeline results
        assert isinstance(cleaned_epochs, mne.Epochs)
        assert isinstance(contrast_result, dict)
        assert len(contrast_result) > 0

    def test_marker_parameter_validation(self):
        """Test parameter validation for ICM LG markers."""
        # Test ICM LG Contrast marker parameters with correct parameter name
        contrast_marker = ICMLGContrast()
        # Test that marker initializes correctly
        assert hasattr(contrast_marker, "time_windows")

        # Test ICM LG MMN marker parameters
        mmn_marker = ICMLGMismatchNegativity(
            mmn_window=[0.15, 0.25],
            n1_window=[0.08, 0.14],
        )
        assert mmn_marker.mmn_window == [0.15, 0.25]
        assert mmn_marker.n1_window == [0.08, 0.14]

        # Test ICM LG Global Local Index marker parameters
        indices_marker = ICMLGGlobalLocalIndex(
            early_window=[0.08, 0.20],
            late_window=[0.20, 0.40],
        )
        assert indices_marker.early_window == [0.08, 0.20]
        assert indices_marker.late_window == [0.20, 0.40]

    def test_trigger_processing(self):
        """Test trigger processing functionality."""
        from junifer_eeg.datareader.icm_lg_datareader import (
            ICMTriggerProcessor,
        )

        # Create mock raw data
        raw = self.create_mock_icm_lg_raw(equipment="egi")

        # Test trigger processing with correct instantiation
        processor = ICMTriggerProcessor("egi")
        processed_raw = processor.process_triggers(raw)

        # Should return a Raw object
        assert isinstance(processed_raw, mne.io.BaseRaw)

    def test_component_attributes(self):
        """Test that all components have required attributes."""
        # Test data reader
        reader = ICMLGDataReader()
        assert hasattr(reader, "_fit_transform")

        # Test preprocessors
        filter_proc = ICMEquipmentFilter()
        assert hasattr(filter_proc, "preprocess")
        assert hasattr(filter_proc, "get_valid_inputs")

        epoching = ICMLGEpoching()
        assert hasattr(epoching, "preprocess")
        assert hasattr(epoching, "get_valid_inputs")

        adaptive_rej = ICMAdaptiveArtifactRejection()
        assert hasattr(adaptive_rej, "preprocess")
        assert hasattr(adaptive_rej, "get_valid_inputs")

        # Test markers
        contrast = ICMLGContrast()
        assert hasattr(contrast, "compute")

        mmn = ICMLGMismatchNegativity()
        assert hasattr(mmn, "compute")

        indices = ICMLGGlobalLocalIndex()
        assert hasattr(indices, "compute")


if __name__ == "__main__":
    pytest.main([__file__])
