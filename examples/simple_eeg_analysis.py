#!/usr/bin/env python3
"""Simple EEG analysis example using junifer_eeg extension.

This example demonstrates the new architecture following Fede's feedback:
- Uses junifer's PatternDataGrabber
- Uses EEGDataReader for file loading
- Uses EEGFilter for preprocessing
- Uses SpectralPower for feature extraction
"""

from pathlib import Path

import numpy as np

# Import junifer components
from junifer.datagrabber import PatternDataGrabber

# Import junifer_eeg components
from junifer_eeg.datareader import EEGDataReader
from junifer_eeg.markers import SpectralPower
from junifer_eeg.preprocessors import EEGFilter


def create_test_data():
    """Create synthetic EEG test data."""
    import mne

    # Create synthetic EEG data
    sfreq = 100  # 100 Hz sampling rate
    duration = 10  # 10 seconds
    times = np.arange(0, duration, 1 / sfreq)

    # Create different frequency components
    alpha_signal = np.sin(2 * np.pi * 10 * times)  # 10 Hz alpha
    beta_signal = np.sin(2 * np.pi * 20 * times)  # 20 Hz beta
    noise = np.random.normal(0, 0.1, len(times))

    # Combine signals
    data = alpha_signal + 0.5 * beta_signal + noise

    # Create MNE Raw object
    info = mne.create_info(
        ch_names=["Cz", "Fz", "Pz"], sfreq=sfreq, ch_types=["eeg"] * 3
    )
    raw_data = np.tile(data, (3, 1))  # 3 channels
    raw = mne.io.RawArray(raw_data, info)

    return raw


def main():
    """Run simple EEG analysis example."""
    print("=== Simple EEG Analysis Example ===")

    # Create test data directory
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)

    # Create synthetic EEG data
    print("Creating synthetic EEG data...")
    raw = create_test_data()

    # Save as EDF file using MNE's write_raw_edf
    test_file = data_dir / "subject1_eeg.edf"
    import mne

    mne.export.export_raw(test_file, raw, fmt="edf", overwrite=True)
    print(f"Saved test data to: {test_file}")

    # Configure data grabber using junifer's PatternDataGrabber
    # Now we can use EEG as the data type!
    print("\nConfiguring data grabber...")
    dg = PatternDataGrabber(
        datadir=str(data_dir),
        patterns={"EEG": {"pattern": "{subject}_eeg.edf", "space": "native"}},
        replacements=["subject"],
        types=["EEG"],
    )

    # Use EEGDataReader
    print("Using EEGDataReader...")
    reader = EEGDataReader()

    # Process data
    print("\nProcessing data...")
    with dg:
        element = "subject1"
        print(f"Processing element: {element}")

        # Get data using data grabber
        data = dg[element]
        print(f"Data keys: {list(data.keys())}")

        # Read data using EEGDataReader
        data = reader.fit_transform(data)
        print(f"After reading - Data keys: {list(data.keys())}")

        # Apply filtering
        print("Applying EEG filter...")
        filter_obj = EEGFilter(low_freq=1.0, high_freq=40.0, on="EEG")
        data = filter_obj.fit_transform(data)
        print(f"After filtering - Data keys: {list(data.keys())}")

        # Compute spectral power
        print("Computing spectral power...")
        marker = SpectralPower(on="EEG")
        result = marker.fit_transform(data)

        # Debug: Check what the marker returned
        print(f"Marker result keys: {list(result.keys())}")

        # Display results
        print("\n=== Results ===")
        # The result is nested under the data type key ('EEG')
        if "EEG" in result and "spectral_power" in result["EEG"]:
            spectral_data = result["EEG"]["spectral_power"]
            print(f"Number of features: {spectral_data['data'].shape[1]}")
            print(f"Feature names: {spectral_data['col_names']}")

            # Show some sample values
            print("\nSample feature values:")
            for i, name in enumerate(
                spectral_data["col_names"][:6]
            ):  # First 6 features
                value = spectral_data["data"][0, i]
                print(f"  {name}: {value:.4f}")

            # Show that alpha power is higher (since we have 10Hz signal)
            print("\nVerification:")
            alpha_indices = [
                i
                for i, name in enumerate(spectral_data["col_names"])
                if "alpha" in name
            ]
            for idx in alpha_indices:
                name = spectral_data["col_names"][idx]
                value = spectral_data["data"][0, idx]
                print(
                    f"  {name}: {value:.4f} (should be higher for 10Hz signal)"
                )
        else:
            print("No spectral_power key found in result")
            print(f"Available keys: {list(result.keys())}")

    print("\n=== Analysis Complete ===")
    print("This demonstrates the new junifer_eeg architecture:")
    print("- Uses junifer's PatternDataGrabber with EEG data type")
    print("- Uses EEGDataReader for file loading")
    print("- Uses EEGFilter for preprocessing")
    print("- Uses SpectralPower for feature extraction")


if __name__ == "__main__":
    main()
