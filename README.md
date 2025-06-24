# JUnifer EEG

EEG extension for the JUelich NeuroImaging FEature extractoR ([junifer](https://github.com/juaml/junifer)).

## Overview

`junifer_eeg` provides specialized tools for extracting features from EEG data within the junifer framework. This extension follows the [junifer extension documentation](https://github.com/juaml/junifer/blob/main/docs/extending.rst) pattern perfectly, enabling seamless "no code" EEG analysis through YAML configuration.

## Features

- **🎯 No Code Analysis**: Complete EEG analysis using only YAML configuration
- **📁 EEG Data Grabbing**: Automatic file discovery using patterns  
- **🔄 Preprocessing**: Load EEG files into MNE Raw objects
- **📊 Spectral Power Analysis**: Extract power in frequency bands (delta, theta, alpha, beta)
- **🧩 Full Junifer Integration**: Seamless integration with junifer's pipeline system
- **✅ Production Ready**: Comprehensive tests, CLI tools, and proper error handling

## Architecture

Following the junifer extension documentation pattern:

```
junifer_eeg/
├── datagrabber/         # EEGDataGrabber - finds EEG files
├── preprocessors/       # EEGLoader - loads EDF files into MNE objects  
├── markers/            # SpectralPower - computes frequency band power
└── cli.py             # Command-line interface
```

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/neurometers/junifer_eeg.git
cd junifer_eeg

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

- `junifer >= 0.0.4`
- `mne >= 1.6.0`
- `numpy >= 1.26.0`
- `pandas >= 2.0.0`

## Quick Start - No Code Approach

### 1. Prepare your EEG data

Organize your EDF files using the pattern `{subject}_eeg.edf`:

```
data/
├── subject1_eeg.edf
├── subject2_eeg.edf
└── subject3_eeg.edf
```

### 2. Create a YAML configuration

Create `eeg_analysis.yaml`:

```yaml
# EEG Analysis Pipeline Configuration

# Import the junifer_eeg extension to register all components
with:
  - junifer_eeg

# Configure working directory  
workdir: .

# Configure the data grabber for EEG files
datagrabber:
  kind: EEGDataGrabber
  datadir: data

# Define the subjects to process
elements:
  - subject1
  - subject2
  - subject3

# Configure preprocessing to load EEG data from files
preprocess:
  - kind: EEGLoader
    on: BOLD

# Configure the markers to compute
markers:
  - name: spectral_power_analysis
    kind: SpectralPower
    on: BOLD

# Configure storage
storage:
  kind: SQLiteFeatureStorage
  uri: eeg_results.sqlite
```

### 3. Run the analysis

```bash
# Using junifer directly (no code needed!)
junifer run eeg_analysis.yaml

# Or using our CLI
junifer-eeg run eeg_analysis.yaml

# Check available components
junifer-eeg info
```

### 4. View results

Results are automatically saved to the SQLite database specified in your configuration.

## Components

### EEGDataGrabber
- **Purpose**: Discovers EEG files using patterns
- **Pattern**: `{subject}_eeg.edf`  
- **Registration**: `@register_datagrabber`
- **Base Class**: `PatternDataGrabber`

### EEGLoader (Preprocessor)
- **Purpose**: Loads EDF files into MNE Raw objects
- **Input**: File paths from DataGrabber
- **Output**: MNE Raw objects for analysis
- **Registration**: `@register_preprocessor`

### SpectralPower (Marker)
- **Purpose**: Computes power in frequency bands
- **Bands**: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz)
- **Output**: Vector features for each channel and band
- **Registration**: `@register_marker`

## Programmatic Usage

While the no-code approach is recommended, you can also use the components programmatically:

```python
import junifer_eeg  # This registers all components

from junifer_eeg.datagrabber import EEGDataGrabber
from junifer_eeg.preprocessors import EEGLoader  
from junifer_eeg.markers import SpectralPower

# Components are now available in junifer's registry
# and can be used in pipelines or directly
```

## Development

### Running Tests

```bash
# Run all tests
pytest junifer_eeg/tests/ -v

# Run specific test
pytest junifer_eeg/tests/test_spectral_power.py -v
```

### Code Quality

This project uses:
- `ruff` for linting and formatting
- `pytest` for testing
- Pre-commit hooks for automatic formatting

```bash
# Check code quality
ruff check junifer_eeg/

# Format code
ruff format junifer_eeg/

# Install pre-commit hooks
pre-commit install
```

## Supported File Formats

Currently supported:
- **EDF/EDF+**: European Data Format files
- More formats can be easily added by extending the EEGLoader preprocessor

## Technical Notes

- Uses BOLD data type internally (closest to EEG time-series in junifer)
- Follows junifer's data flow: DataGrabber → Preprocessor → Marker → Storage
- Fully compatible with junifer's validation and metadata systems
- Supports all junifer storage backends (SQLite, HDF5, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.

## Authors

- Giovanni Marraffini (g.marraffini@neurometers.ai)
- Fede Raimondo (f.raimondo@fz-juelich.de)

## Acknowledgments

- Built following the [junifer extension documentation](https://github.com/juaml/junifer/blob/main/docs/extending.rst)
- Uses [MNE-Python](https://mne.tools/) for EEG data handling
- Integrates seamlessly with the [junifer](https://github.com/juaml/junifer) ecosystem 