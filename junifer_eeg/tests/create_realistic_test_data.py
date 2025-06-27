"""Create realistic 64-channel EDF test data for junifer_eeg testing."""

import mne
import numpy as np


def create_realistic_eeg_data():
    """Create realistic 64-channel EEG data with typical frequency content."""

    # 64-channel standard 10-20 extended system
    ch_names = [
        # Frontal
        "Fp1",
        "Fpz",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "AF4",
        "AF8",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        # Central
        "T7",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        # Parietal
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        # Occipital
        "O1",
        "Oz",
        "O2",
    ]

    # Parameters
    sfreq = 250  # Sampling frequency
    duration = 120  # 2 minutes of data
    n_channels = len(ch_names)
    n_samples = int(sfreq * duration)

    # Create time vector
    times = np.arange(n_samples) / sfreq

    # Initialize data array
    data = np.zeros((n_channels, n_samples))

    # Generate realistic EEG signals for each channel
    np.random.seed(42)  # For reproducibility

    for ch_idx, ch_name in enumerate(ch_names):
        # Base signal components
        signal = np.zeros(n_samples)

        # 1. Alpha rhythm (8-13 Hz) - stronger in posterior channels
        alpha_strength = 2.0 if any(x in ch_name for x in ["P", "O"]) else 1.0
        for freq in np.linspace(8, 13, 3):
            alpha_amplitude = alpha_strength * np.random.uniform(0.5, 1.5)
            signal += alpha_amplitude * np.sin(
                2 * np.pi * freq * times + np.random.uniform(0, 2 * np.pi)
            )

        # 2. Beta rhythm (13-30 Hz) - stronger in frontal/central channels
        beta_strength = 1.5 if any(x in ch_name for x in ["F", "C"]) else 0.8
        for freq in np.linspace(15, 25, 2):
            beta_amplitude = beta_strength * np.random.uniform(0.3, 0.8)
            signal += beta_amplitude * np.sin(
                2 * np.pi * freq * times + np.random.uniform(0, 2 * np.pi)
            )

        # 3. Theta rhythm (4-8 Hz) - moderate across all channels
        for freq in np.linspace(4, 8, 2):
            theta_amplitude = np.random.uniform(0.5, 1.0)
            signal += theta_amplitude * np.sin(
                2 * np.pi * freq * times + np.random.uniform(0, 2 * np.pi)
            )

        # 4. Delta rhythm (1-4 Hz) - low frequency background
        for freq in np.linspace(1, 4, 2):
            delta_amplitude = np.random.uniform(1.0, 2.0)
            signal += delta_amplitude * np.sin(
                2 * np.pi * freq * times + np.random.uniform(0, 2 * np.pi)
            )

        # 5. Add some realistic artifacts and noise
        # Slow drift
        drift = 0.5 * np.sin(
            2 * np.pi * 0.02 * times + np.random.uniform(0, 2 * np.pi)
        )
        signal += drift

        # Eye blinks (stronger in frontal channels)
        if any(x in ch_name for x in ["Fp", "AF", "F"]):
            blink_times = np.random.poisson(
                0.3, n_samples
            )  # ~0.3 blinks per second
            blink_kernel = np.exp(-(np.linspace(-3, 3, int(0.3 * sfreq)) ** 2))
            blinks = np.convolve(blink_times, blink_kernel, mode="same")[
                :n_samples
            ]
            signal += 5.0 * blinks

        # Muscle artifacts (stronger in temporal channels)
        if any(x in ch_name for x in ["T", "TP", "FT"]):
            muscle_noise = np.random.normal(0, 0.3, n_samples)
            # High-pass filter to simulate EMG
            from scipy.signal import butter, filtfilt

            b, a = butter(4, 30 / (sfreq / 2), "high")
            muscle_filtered = filtfilt(b, a, muscle_noise)
            signal += muscle_filtered

        # 6. Add white noise
        noise = np.random.normal(0, 0.5, n_samples)
        signal += noise

        # 7. Add some event-related potentials occasionally
        # Simulate P300-like responses
        if np.random.random() < 0.1:  # 10% chance per channel
            event_times = np.random.choice(
                n_samples, size=int(duration / 10), replace=False
            )
            for event_time in event_times:
                if event_time + int(0.5 * sfreq) < n_samples:
                    # Create P300-like waveform
                    t_erp = np.linspace(-0.1, 0.4, int(0.5 * sfreq))
                    erp = (
                        3 * np.exp(-((t_erp - 0.3) ** 2) / 0.05) * (t_erp > 0)
                    )
                    signal[event_time : event_time + len(erp)] += erp

        # Scale to microvolts (typical EEG amplitude range)
        signal *= 0.1  # Scale to ~1-5 µV range (more realistic)

        data[ch_idx] = signal

    # Create MNE info object
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)

    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    return raw


def save_test_data():
    """Save realistic test data as EDF file."""
    raw = create_realistic_eeg_data()

    # Save as EDF
    filename = "test_data/subject1_eeg.edf"
    raw.export(filename, fmt="edf", overwrite=True, verbose=False)
    print(f"Created realistic EDF file: {filename}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    return filename


if __name__ == "__main__":
    # Create directory if it doesn't exist
    import os

    os.makedirs("test_data", exist_ok=True)

    save_test_data()
