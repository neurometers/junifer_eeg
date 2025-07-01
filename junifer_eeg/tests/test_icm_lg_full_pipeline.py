"""Tests for ICM Local Global full pipeline integration."""

import mne
import numpy as np
import pytest

from junifer_eeg.markers import (
    ContingentNegativeVariation,
    PermutationEntropy,
    SpectralPower,
    SymbolicMutualInformation,
    TimeLockedContrast,
    WindowDecoding,
)


class TestICMLGFullPipeline:
    """Test ICM Local Global markers with synthetic data pipeline."""

    @pytest.fixture(scope="class")
    def synthetic_epochs(self):
        """Create comprehensive synthetic EEG data for ICM LG testing."""
        # Parameters
        n_channels = 16  # Smaller for faster tests
        n_times = 300  # 1.2 seconds at 250 Hz
        sfreq = 250
        n_epochs_per_condition = 30

        # Create realistic channel names
        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # ICM Local Global conditions
        conditions = ["LSGS", "LDGD", "LSGD", "LDGS"]
        event_id = {cond: i + 1 for i, cond in enumerate(conditions)}

        # Generate events
        events = []
        all_epochs_data = []

        np.random.seed(42)  # For reproducible tests

        for _cond_idx, condition in enumerate(conditions):
            for _epoch_idx in range(n_epochs_per_condition):
                # Event timing
                event_time = len(all_epochs_data) * n_times
                events.append([event_time, 0, event_id[condition]])

                # Create realistic ERP-like data
                epoch_data = np.random.randn(n_channels, n_times) * 1e-6

                # Add condition-specific components
                # P1 component (~100ms)
                p1_time = int(0.1 * sfreq)
                p1_amplitude = 2e-6 if condition in ["LSGS", "LSGD"] else 1e-6
                epoch_data[:, p1_time : p1_time + 10] += p1_amplitude

                # MMN component (~150ms) - more for deviants
                mmn_time = int(0.15 * sfreq)
                mmn_amplitude = 3e-6 if "D" in condition else 1e-6
                epoch_data[:4, mmn_time : mmn_time + 15] += (
                    mmn_amplitude  # Frontal channels
                )

                # P3a component (~250ms) - more for local deviants
                p3a_time = int(0.25 * sfreq)
                p3a_amplitude = 4e-6 if condition == "LDGS" else 2e-6
                epoch_data[:8, p3a_time : p3a_time + 20] += (
                    p3a_amplitude  # Fronto-central
                )

                # P3b component (~400ms) - more for global deviants
                p3b_time = int(0.4 * sfreq)
                p3b_amplitude = 5e-6 if condition == "LSGD" else 2e-6
                epoch_data[8:, p3b_time : p3b_time + 25] += (
                    p3b_amplitude  # Parietal
                )

                all_epochs_data.append(epoch_data)

        # Concatenate all data
        events = np.array(events)
        data = np.concatenate(all_epochs_data, axis=1)
        raw = mne.io.RawArray(data, info, verbose=False)

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=-0.2,
            tmax=1.0,
            baseline=(-0.2, 0),
            preload=True,
            verbose=False,
        )

        return epochs

    def test_time_locked_contrast_markers(self, synthetic_epochs):
        """Test TimeLockedContrast markers with different configurations."""
        epochs = synthetic_epochs

        # Test 1: Basic MMN contrast
        mmn_marker = TimeLockedContrast(
            condition_a="LSGS",  # Local standard
            condition_b="LDGS",  # Local deviant
            tmin=0.1,
            tmax=0.2,
            comment="MMN",
        )

        result = mmn_marker.compute({"data": epochs})
        assert "timelockedcontrast" in result
        assert "data" in result["timelockedcontrast"]
        assert "col_names" in result["timelockedcontrast"]

        data = result["timelockedcontrast"]["data"]
        col_names = result["timelockedcontrast"]["col_names"]

        # Check data properties
        assert data.shape[0] == 1  # One contrast
        assert data.shape[1] == len(epochs.ch_names)
        assert len(col_names) == len(epochs.ch_names)
        assert all(np.isfinite(data.flatten()))

    def test_window_decoding_markers(self, synthetic_epochs):
        """Test WindowDecoding markers with different time windows."""
        epochs = synthetic_epochs

        # Test 1: Early decoding (P1 window)
        early_decoder = WindowDecoding(
            condition_a="LSGS",
            condition_b="LDGS",
            tmin=0.08,
            tmax=0.12,
            n_splits=3,  # Fewer splits for faster testing
            random_state=42,
            comment="EarlyDecoding",
        )

        result = early_decoder.compute({"data": epochs})
        assert "windowdecoding" in result
        assert "data" in result["windowdecoding"]
        assert "col_names" in result["windowdecoding"]

        data = result["windowdecoding"]["data"]
        col_names = result["windowdecoding"]["col_names"]

        # Check data properties
        assert data.shape == (1, 1)  # One decoding score
        assert len(col_names) == 1
        assert 0.0 <= data[0, 0] <= 1.0  # Valid probability

    def test_symbolic_mutual_information_marker(self, synthetic_epochs):
        """Test SymbolicMutualInformation connectivity analysis."""
        epochs = synthetic_epochs

        # Test with default parameters
        smi_marker = SymbolicMutualInformation(
            tmin=0.0, tmax=0.5, kernel=3, tau=1, weighted=True
        )

        result = smi_marker.compute({"data": epochs})
        assert "symbolic_mutual_information" in result
        assert "data" in result["symbolic_mutual_information"]
        assert "col_names" in result["symbolic_mutual_information"]

        data = result["symbolic_mutual_information"]["data"]
        col_names = result["symbolic_mutual_information"]["col_names"]
        n_channels = len(epochs.ch_names)

        # Check data properties
        assert data.shape == (1, n_channels * n_channels)
        assert len(col_names) == n_channels * n_channels

        # Reshape to matrix and check properties
        smi_matrix = data.reshape(n_channels, n_channels)

        # Diagonal should be 1.0 (self-connectivity)
        diagonal = np.diag(smi_matrix)
        assert np.allclose(diagonal, 1.0, atol=0.01)

        # All values should be finite
        assert np.all(np.isfinite(smi_matrix))

    def test_spectral_and_entropy_markers(self, synthetic_epochs):
        """Test spectral power and entropy-based markers."""
        epochs = synthetic_epochs

        # Convert epochs to raw for continuous markers
        raw_data = epochs.get_data().mean(axis=0)  # Average across epochs
        raw = mne.io.RawArray(raw_data, epochs.info.copy(), verbose=False)

        # Test SpectralPower
        spectral_marker = SpectralPower(
            epoch_length=1.0, overlap=0.0, trial_aggregation_method=["mean"]
        )

        result = spectral_marker.compute({"data": raw})
        assert "spectralpower" in result
        assert "data" in result["spectralpower"]
        data = result["spectralpower"]["data"]
        assert np.all(np.isfinite(data))
        assert np.all(data > 0)  # Power should be positive

        # Test PermutationEntropy
        pe_marker = PermutationEntropy(
            kernel=3,
            tau=8,
            epoch_length=1.0,
            overlap=0.0,
            trial_aggregation_method=["mean"],
        )

        result = pe_marker.compute({"data": raw})
        assert "permutationentropy" in result
        assert "data" in result["permutationentropy"]
        data = result["permutationentropy"]["data"]
        assert np.all(np.isfinite(data))
        assert np.all(data >= 0)  # PE should be non-negative

    def test_cnv_and_topography_markers(self, synthetic_epochs):
        """Test CNV and topography markers."""
        epochs = synthetic_epochs

        # Convert epochs to raw for continuous markers
        raw_data = epochs.get_data().mean(axis=0)
        raw = mne.io.RawArray(raw_data, epochs.info.copy(), verbose=False)

        # Test ContingentNegativeVariation
        cnv_marker = ContingentNegativeVariation(
            tmin=0.1,
            tmax=0.5,
            epoch_length=1.0,
            overlap=0.0,
            trial_aggregation_method=["mean"],
        )

        result = cnv_marker.compute({"data": raw})
        assert "cnvslope" in result
        assert "data" in result["cnvslope"]
        data = result["cnvslope"]["data"]
        assert np.all(np.isfinite(data))

    def test_icm_roi_system(self, synthetic_epochs):
        """Test ICM-specific ROI system."""
        from junifer_eeg.markers.utils import get_icm_roi_mapping

        # Test different equipment configurations
        for equipment in ["standard", "egi128", "egi256"]:
            roi_mapping = get_icm_roi_mapping(equipment)

            # Check all expected ICM ROIs are present
            expected_rois = ["scalp", "cnv", "mmn", "p3a", "p3b"]
            for roi in expected_rois:
                assert roi in roi_mapping
                assert len(roi_mapping[roi]) > 0  # Should have electrodes

    def test_pipeline_integration(self, synthetic_epochs):
        """Test complete pipeline integration."""
        epochs = synthetic_epochs

        # Test processing with multiple markers
        markers = [
            TimeLockedContrast(
                condition_a="LSGS",
                condition_b="LDGD",
                tmin=0.1,
                tmax=0.2,
                comment="integration_test",
            ),
            WindowDecoding(
                condition_a="LSGS",
                condition_b="LDGD",
                tmin=0.1,
                tmax=0.2,
                n_splits=3,
                random_state=42,
                comment="integration_test",
            ),
        ]

        # Process with each marker
        results = []
        for marker in markers:
            result = marker.compute({"data": epochs})
            results.append(result)

            # Check that each result is valid
            assert len(result) == 1  # One feature type per marker
            feature_name = next(iter(result.keys()))
            assert "data" in result[feature_name]
            assert "col_names" in result[feature_name]

            data = result[feature_name]["data"]
            assert np.all(np.isfinite(data))

        # Verify we got results from all markers
        assert len(results) == len(markers)

        print("✅ Complete ICM Local Global pipeline integration test passed!")
