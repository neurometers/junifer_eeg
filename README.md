# JUnifer EEG

JUnifer EEG is an extension for the JUelich NeuroImaging FEature extractoR (junifer) that provides EEG-specific components for electrophysiological data analysis.

## Overview

This extension follows junifer's architectural patterns to add EEG support by:

- **Using junifer's default `PatternDataGrabber`** - No custom data grabber needed
- **Extending `DefaultDataReader`** - Adds EEG file format support (.edf, .bdf) via global extension mappings
- **Simple preprocessing** - `EEGFilter` for basic frequency filtering
- **Feature extraction** - `SpectralPower` marker for frequency band analysis
- **HDF5 storage** - Uses junifer's recommended storage format
- **EEG data type** - Extends junifer's validation to support EEG alongside BOLD/T1w

## Architecture

This extension follows the recommended junifer pattern:

```
PatternDataGrabber → EEGDataReader → EEGFilter → SpectralPower → HDF5FeatureStorage
```

All components integrate seamlessly with junifer's existing infrastructure and CLI.

## Installation

```bash
pip install junifer-eeg
```

For development:
```bash
git clone https://github.com/neurometers/junifer_eeg.git
cd junifer_eeg
pip install -e .
```

## Quick Start

### YAML Configuration (Recommended)

Create a YAML configuration file using junifer's standard `PatternDataGrabber`:

```yaml
# eeg_analysis.yaml
workdir: .

# Import the junifer_eeg extension to register EEG components
with:
  - junifer_eeg

# Use junifer's default PatternDataGrabber with EEG data type
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

# Configure storage using HDF5 (SQLite is deprecated)
storage:
  kind: HDF5FeatureStorage
  uri: eeg_results.hdf5

# Optional: HTCondor queue configuration
queue:
  jobname: eeg_spectral_power
  kind: HTCondor
  env:
    kind: conda
    name: junifer_dev
  mem: 8G
  cpus: 1
  disk: 5G
  verbose: info
  collect: yes
```

Run using junifer CLI:

```bash
junifer run eeg_analysis.yaml
```

### Python API

```python
from junifer.datagrabber import PatternDataGrabber
from junifer_eeg.datareader import EEGDataReader
from junifer_eeg.preprocessors import EEGFilter
from junifer_eeg.markers import SpectralPower

# Configure data grabber (uses junifer's default PatternDataGrabber)
dg = PatternDataGrabber(
    datadir="data",
    patterns={"EEG": {"pattern": "{subject}_eeg.edf", "space": "native"}},
    replacements=["subject"],
    types=["EEG"]
)

# Use EEGDataReader (extends DefaultDataReader)
reader = EEGDataReader()

# Process data
with dg:
    element = "subject1"
    data = dg[element]
    
    # Read EEG data
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

Extends junifer's `DefaultDataReader` by modifying global extension mappings:

- **Architecture**: Extends `_extensions`, `_readers`, and `PATTERNS_SCHEMA` at import time
- **Supported formats**: EDF, BDF files via MNE-Python
- **Data type**: Adds `EEG` as a valid data type in junifer's validation schema
- **Integration**: Works seamlessly with `PatternDataGrabber`

```python
# The reader appears "empty" because it extends junifer's global mappings:
default_module._extensions.update({".edf": "EDF", ".bdf": "EDF"})
default_module._readers["EDF"] = {"func": _read_edf, "params": None}
validation_module.PATTERNS_SCHEMA["EEG"] = {...}
```

### EEGFilter

Simple preprocessing component for frequency filtering:

- **Input**: EEG data (MNE Raw objects)
- **Output**: Filtered EEG data
- **Parameters**:
  - `low_freq`: Low frequency cutoff (Hz)
  - `high_freq`: High frequency cutoff (Hz)
  - `on`: Data type to process ("EEG")
- **Method**: Uses MNE's `filter()` method

### SpectralPower

Marker for computing spectral power features:

- **Input**: EEG data (MNE Raw objects)
- **Output**: Power spectral density features per channel and frequency band
- **Frequency bands**: 
  - Delta: 1-4 Hz
  - Theta: 4-8 Hz  
  - Alpha: 8-13 Hz
  - Beta: 13-30 Hz
- **Features**: `{channel}_{band}` format (e.g., "Cz_alpha", "Fz_beta")
- **Method**: Uses MNE's `compute_psd()` with mean power per band

## Data Organization

Organize your EEG data to match your YAML pattern:

```
data/
├── subject1_eeg.edf
├── subject2_eeg.edf
├── subject3_eeg.edf
└── ...
```

Or for more complex patterns:
```
data/
├── sub-001/
│   └── ses-baseline/
│       └── eeg/
│           └── sub-001_ses-baseline_task-localglobal.edf
└── sub-002/
    └── ses-baseline/
        └── eeg/
            └── sub-002_ses-baseline_task-localglobal.edf
```

With corresponding pattern:
```yaml
patterns:
  EEG:
    pattern: "{subject}/{session}/eeg/{subject}_{session}_task-localglobal.edf"
    space: "native"
replacements:
  - subject
  - session
```

## Key Features

- ✅ **No custom data grabber** - Uses junifer's `PatternDataGrabber`
- ✅ **Extends DefaultDataReader** - Clean extension via global mappings
- ✅ **EEG data type support** - Proper semantic data type (not BOLD)
- ✅ **HDF5 storage** - Uses recommended storage format
- ✅ **junifer CLI integration** - No custom CLI needed
- ✅ **HTCondor ready** - Queue configuration for cluster computing
- ✅ **MNE integration** - Leverages MNE-Python for EEG processing

## Dependencies

- junifer >= 0.0.4
- mne >= 1.6.0
- numpy >= 1.26.0
- pandas >= 2.0.0

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=junifer_eeg

# Run specific test
pytest junifer_eeg/tests/test_spectral_power.py -v
```

## Examples

See the `examples/` directory for:

- `eeg_pipeline.yaml` - Complete YAML configuration
- `simple_eeg_analysis.py` - Python API usage
- `README.md` - Detailed examples and explanations

## Contributing

This extension follows junifer's architectural patterns. When contributing:

1. Use junifer's existing components when possible
2. Extend global mappings for new data types/formats
3. Follow junifer's registration decorators
4. Use proper data type semantics
5. Include comprehensive tests

## License

MIT License - see LICENSE file for details.

