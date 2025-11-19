# JUnifer EEG

JUnifer EEG is an extension for the JUelich NeuroImaging FEature extractoR (junifer) that provides EEG-specific components for electrophysiological data analysis.

## Overview

This extension follows junifer's architectural patterns to add EEG support by:

- **Extending `PatternDataGrabber`** - Just a way to use the extended datareader.
- **Extending `DefaultDataReader`** - Adds EEG file format support (.edf, .bdf) via global extension mappings
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

# Create a virtual environment (works with UV tools also) 

```bash
python3 -m venv venv
```
# Install requiremetns

```bash
pip install -r requirements.txt
```
# Finally install junifer-eeg

```bash
pip install -e .
```

## Quick Start (Command Line Only)

1. **Install**: Follow previous installation instructions.
2. **Prepare Data**: Use EEG in .mff / .edf /.fif
3. **Copy Example**: `cp examples/icm_complete_individual_markers.yaml my_analysis.yaml`
4. **Edit Config**: Update `my_analysis.yaml` with your desired path, preprocessing and markers.
5. **Run**: `junifer run my_analysis.yaml`
6. **Results**: Features saved in the path specified in your yaml.


## Testing

```bash
# Run all tests  
pytest junifer_eeg/tests/

# Run validation tests
pytest junifer_eeg/tests/validation/ 
```



## Contributing

This extension follows junifer's architectural patterns. When contributing:

1. Use junifer's existing components when possible
2. Include comprehensive tests
3. Use HDF5 for data storage



