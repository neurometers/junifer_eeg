# JUnifer EEG Extension Examples

This directory contains examples demonstrating how to use the `junifer_eeg` extension for EEG analysis within the junifer framework.

## Overview

The `junifer_eeg` extension follows the junifer extension pattern documented in the [Junifer Extension Documentation](https://github.com/juaml/junifer/blob/main/docs/extending.rst). It provides:

- **EEG Data Grabbing**: Support for EEG file formats via `EEGDataGrabber`
- **EEG Data Loading**: Load EEG files into MNE Raw objects via `EEGLoader` preprocessor
- **Spectral Power Analysis**: Extract power in frequency bands via `SpectralPower` marker
- **Seamless Integration**: Works with junifer's codeless YAML configuration

## Architecture

Our extension follows the proper junifer architecture pattern:

```
junifer_eeg/
├── __init__.py              # Main module with imports
├── markers/                 # EEG-specific markers
│   ├── __init__.py
│   └── spectral_power.py   # SpectralPower marker (@register_marker)
├── datagrabber/            # EEG data grabbers  
│   ├── __init__.py
│   └── eeg_datagrabber.py  # EEGDataGrabber (@register_datagrabber)
├── preprocessors/          # EEG preprocessors
│   ├── __init__.py
│   └── eeg_loader.py       # EEGLoader (@register_preprocessor)
└── cli.py                  # Command-line interface
```

## Data Flow

The extension follows the standard junifer data flow:

1. **DataGrabber** (`EEGDataGrabber`): Provides file paths to EEG files
2. **Preprocessor** (`EEGLoader`): Loads EEG files into MNE Raw objects
3. **Marker** (`SpectralPower`): Processes loaded EEG data to extract features

## Usage Patterns

### 1. Programmatic Usage

```python
# Import the extension to register components
import junifer_eeg

# Use components directly
from junifer_eeg.markers import SpectralPower
from junifer_eeg.datagrabber import EEGDataGrabber
from junifer_eeg.preprocessors import EEGLoader

# Create and use components
grabber = EEGDataGrabber(datadir="/path/to/data")
loader = EEGLoader()
marker = SpectralPower()
```

### 2. Codeless YAML Configuration (Recommended)

The recommended approach is to use YAML configuration:

```yaml
# Import the extension
with:
  - junifer_eeg

# Configure working directory
workdir: .

# Configure data grabbing
datagrabber:
  kind: EEGDataGrabber
  datadir: /path/to/eeg/data

# Define subjects to process
elements:
  - subject1
  - subject2

# Configure preprocessing to load EEG data
preprocess:
  - kind: EEGLoader
    on: BOLD

# Configure analysis
markers:
  - name: spectral_power_analysis
    kind: SpectralPower
    on: BOLD

# Configure storage
storage:
  kind: SQLiteFeatureStorage
  uri: eeg_features.sqlite
```

### 3. Command Line Usage

```bash
# Show available components
junifer-eeg info

# Run analysis with config file
junifer run eeg_pipeline.yaml

# Or using our CLI
junifer-eeg run eeg_pipeline.yaml --verbose
```

## File Descriptions

- **`eeg_pipeline.yaml`**: Complete YAML configuration for EEG analysis

## Key Components

### EEGDataGrabber (DataGrabber)

- **Input**: EDF files following pattern `{subject}_eeg.edf`
- **Output**: File paths to EEG data (using BOLD data type)
- **Dependencies**: PatternDataGrabber from junifer
- **Registration**: Uses `@register_datagrabber` decorator

### EEGLoader (Preprocessor)

- **Input**: File paths from DataGrabber (BOLD data type)
- **Output**: EEG data with `raw_object` containing MNE Raw instance
- **Dependencies**: MNE-Python
- **Registration**: Uses `@register_preprocessor` decorator

### SpectralPower (Marker)

- **Input**: BOLD data type containing `raw_object` (MNE Raw instance)
- **Output**: Power values in standard frequency bands (delta, theta, alpha, beta)
- **Dependencies**: MNE-Python, pandas
- **Registration**: Uses `@register_marker` decorator

## Technical Notes

- **Data Type**: Uses BOLD data type internally (closest to EEG time-series in junifer)
- **MNE Integration**: Leverages MNE-Python for all EEG processing
- **File Format**: Currently supports EDF files, easily extensible for other formats

## Extension Development Notes

This extension demonstrates best practices for junifer extensions:

1. **Proper Architecture**: Uses DataGrabber → Preprocessor → Marker pattern
2. **Proper Registration**: Uses decorators (`@register_datagrabber`, `@register_preprocessor`, `@register_marker`)
3. **Class Attributes**: Includes `_DEPENDENCIES` and `_MARKER_INOUT_MAPPINGS`
4. **Simple Implementation**: Leverages existing libraries (MNE) instead of reinventing
5. **Codeless Support**: Can be imported via `with:` statement in YAML
6. **Standard Structure**: Follows junifer patterns for maintainability

## Running Examples

```bash
# Install the extension
pip install -e .

# Prepare your EEG data with pattern: {subject}_eeg.edf
mkdir data
# Place your EDF files: subject1_eeg.edf, subject2_eeg.edf, etc.

# Create a configuration file (see YAML example above)
nano my_eeg_config.yaml

# Run analysis
junifer run my_eeg_config.yaml
```

For more information, see the main [junifer documentation](https://juaml.github.io/junifer/). 