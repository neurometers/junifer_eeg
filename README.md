# JUnifer EEG

JUnifer EEG is an extension for the JUelich NeuroImaging FEature extractoR (junifer) that provides EEG-specific components for electrophysiological data analysis.

## Overview

This extension adds EEG support to junifer by providing:

- **EEGDataReader**: Extends junifer's DefaultDataReader to handle EEG file formats (EDF, BDF)
- **EEGFilter**: Simple EEG filtering preprocessor using MNE
- **SpectralPower**: Marker for computing spectral power in standard EEG frequency bands

## Installation

```bash
pip install junifer-eeg
```

## Quick Start

### Using YAML Configuration (Recommended)

Create a YAML configuration file following junifer's standard pattern:

```yaml
# eeg_analysis.yaml
workdir: .

# Import the junifer_eeg extension
with:
  - junifer_eeg

# Configure data grabber using junifer's PatternDataGrabber
datagrabber:
  kind: PatternDataGrabber
  datadir: data
  patterns:
    EEG:
      pattern: "{subject}_eeg.edf"
      space: "native"
  replacements:
    - subject
  types:
    - EEG

# Define subjects to process
elements:
  - subject1
  - subject2

# Configure preprocessing
preprocess:
  - kind: EEGFilter
    low_freq: 1.0
    high_freq: 40.0
    on: EEG

# Configure markers
markers:
  - name: spectral_power_analysis
    kind: SpectralPower
    on: EEG

# Configure storage using HDF5
storage:
  kind: HDF5FeatureStorage
  uri: eeg_results.hdf5
```

Run the analysis using junifer:

```bash
junifer run eeg_analysis.yaml
```

### Using Python API

```python
from junifer import DataGrabber, DefaultDataReader
from junifer_eeg import EEGDataReader, EEGFilter, SpectralPower

# Use junifer's PatternDataGrabber
from junifer.datagrabber import PatternDataGrabber

# Configure data grabber
dg = PatternDataGrabber(
    datadir="data",
    patterns={"EEG": {"pattern": "{subject}_eeg.edf", "space": "native"}},
    replacements=["subject"],
    types=["EEG"]
)

# Use EEGDataReader
reader = EEGDataReader()

# Process data
with dg:
    for element in dg:
        data = dg[element]
        data = reader.fit_transform(data)
        
        # Apply filtering
        filter_obj = EEGFilter(low_freq=1.0, high_freq=40.0, on="EEG")
        data = filter_obj.fit_transform(data)
        
        # Compute spectral power
        marker = SpectralPower(on="EEG")
        result = marker.fit_transform(data)
```

## Components

### EEGDataReader

Extends junifer's `DefaultDataReader` to handle EEG file formats:

- **Supported formats**: EDF, BDF
- **Dependencies**: MNE-Python
- **Usage**: Automatically registered with junifer when the extension is imported
- **Data type**: Extends junifer to support EEG data type

### EEGFilter

Simple EEG filtering preprocessor:

- **Input**: EEG data (MNE Raw objects)
- **Output**: Filtered EEG data
- **Parameters**:
  - `low_freq`: Low frequency cutoff (Hz)
  - `high_freq`: High frequency cutoff (Hz)
  - `on`: Data type to process (default: "EEG")

### SpectralPower

Computes spectral power in standard EEG frequency bands:

- **Input**: EEG data (MNE Raw objects)
- **Output**: Spectral power features
- **Frequency bands**: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz)
- **Features**: One feature per channel per frequency band
- **Data type**: Uses EEG data type

## Data Organization

Organize your EEG data following this structure:

```
data/
├── subject1_eeg.edf
├── subject2_eeg.edf
└── ...
```

## Technical Notes

- **Data Type**: Properly supports EEG data type in junifer's PatternDataGrabber
- **File Formats**: Currently supports EDF and BDF files via MNE-Python
- **Storage**: Uses HDF5FeatureStorage (SQLite is deprecated in junifer)
- **Extension**: Extends junifer's validation schema to recognize EEG data type

## Dependencies

- junifer >= 0.0.4
- mne >= 1.6.0
- numpy >= 1.26.0
- pandas >= 2.0.0

## Development

```bash
# Clone the repository
git clone https://github.com/neurometers/junifer_eeg.git
cd junifer_eeg

# Install in development mode
pip install -e .

# Run tests
pytest

# Install development dependencies
pip install -e ".[dev]"
```

