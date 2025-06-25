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

### For Non-Python Users (Command Line Only)

1. **Install**: `pip install junifer junifer_eeg`
2. **Prepare Data**: Convert EEG to EDF format, name as `{subject}_eeg.edf`
3. **Copy Example**: `cp examples/simple_eeg_pipeline.yaml my_analysis.yaml`
4. **Edit Config**: Update `datadir` path to your EEG files
5. **Run**: `junifer run my_analysis.yaml`
6. **Results**: Features saved in `eeg_basic_features.hdf5`

### Simple Configuration (`examples/simple_eeg_pipeline.yaml`)

```yaml
with:
  - "junifer_eeg"
workdir: "."

datagrabber:
  kind: PatternDataGrabber
  patterns:
    EEG:
      pattern: "{subject}_eeg.edf"
      space: "native"
  datadir: "./your_data_directory"  # Update this
  types: [EEG]
  replacements: [subject]

preprocess:
  - kind: EEGFilter
    low_freq: 1.0
    high_freq: 40.0
    on: EEG

markers:
  - kind: SpectralPower
    on: EEG
  - kind: KolmogorovComplexity
    nbins: 16
    on: EEG

storage:
  kind: HDF5FeatureStorage
  uri: "eeg_basic_features.hdf5"

elements:
  - subject: "subject1"
```

### Advanced Configuration (`examples/eeg_pipeline_config.yaml`)

Full pipeline with all available markers - see file for complete example.

### Working with Results

```python
import pandas as pd
import h5py

# Load HDF5 results
df_spectral = pd.read_hdf('eeg_basic_features.hdf5', key='SpectralPower')
df_complexity = pd.read_hdf('eeg_basic_features.hdf5', key='KolmogorovComplexity')

# Analyze features
print(df_spectral.mean())  # Average power per frequency band
```

## Available Markers

| Marker | Description | Parameters |
|--------|-------------|------------|
| `SpectralPower` | Power in EEG frequency bands (delta, theta, alpha, beta) | None |
| `KolmogorovComplexity` | Algorithmic complexity measure | `nbins` |
| `PermutationEntropy` | Temporal complexity measure | `kernel`, `tau` |
| `PowerSpectralDensityEstimator` | Detailed PSD analysis | `fmin`, `fmax`, `n_fft` |
| `ContingentNegativeVariation` | Event-related potential analysis | None |
| `TimeDecoding` | Temporal information decoding | `condition_method`, `n_splits` |

## Data Requirements

- **Format**: EDF files (European Data Format)
- **Naming**: `{subject}_eeg.edf` (e.g., `subject1_eeg.edf`)
- **Directory**: All files in single folder
- **Channels**: Standard EEG electrode names recommended

## Testing

```bash
# Run all tests
pytest

# Run specific tests  
pytest junifer_eeg/tests/test_spectral_power.py -v

# Run validation tests (requires NICE package)
pytest junifer_eeg/tests/validation/ -v
```



## Contributing

This extension follows junifer's architectural patterns. When contributing:

1. Use junifer's existing components when possible
2. Follow junifer's registration decorators  
3. Include comprehensive tests
4. Use HDF5 for data storage

## License

MIT License - see LICENSE file for details.

