# junifer-eeg

EEG extension for the JUelich NeuroImaging FEature extractoR (junifer) providing comprehensive electrophysiological data analysis capabilities.

## Overview

junifer-eeg extends junifer with EEG-specific components:

- **Data Support**: .vhdr, .edf, .fif, .mff formats via global reader registry
- **Equipment Handling**: Automatic detection and montage setting for Biosemi, BrainVision, EGI systems
- **Preprocessing**: Filtering, epoching, artifact rejection, interpolation, CSD computation
- **Feature Extraction**: 15+ marker types including spectral, connectivity, complexity, ERP, and decoding
- **Storage**: HDF5-based feature storage with junifer integration

## Markers

### Spectral Analysis
- **SpectralPowerBands** - Power in delta, theta, alpha, beta, gamma bands
- **PowerSpectralDensityEstimator** - Full PSD with customizable parameters
- **PowerSpectralDensitySummary** - Band power summaries and statistics

### Connectivity
- **SymbolicMutualInformation** - SMI and weighted SMI (WSMI) connectivity
- **SymbolicMutualInformationROIs** - Region-based connectivity analysis

### Complexity
- **PermutationEntropy** - Frequency-band specific entropy analysis
- **KolmogorovComplexity** - Compression-based complexity measures

### Event-Related
- **TimeLockedTopography** - ERP topographies and time courses
- **TimeLockedContrast** - Condition contrasts and differential analysis
- **ContingentNegativeVariation** - CNV component detection

### Decoding
- **TimeDecoding** - Temporal decoding with sliding estimator
- **WindowDecoding** - Fixed window decoding analysis

### Oscillation Detection
- **SpindlesDetection** - Sleep spindle detection and characterization
- **SlowWavesDetection** - Slow wave identification and analysis

## Quick Start

### Docker (Recommended)

```bash
# Build image
docker build -t junifer-eeg:latest .

# Run with smart path detection
./docker-junifer.sh run examples/icm_complete_individual_markers.yaml

# With HDF5 to pickle conversion
./docker-junifer.sh run examples/icm_complete_individual_markers.yaml --dump
```

### Local Installation

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install junifer-eeg
pip install -e .
```

## Example Configurations

The `examples/` directory contains ready-to-use YAML configurations:

- **`icm_complete_individual_markers.yaml`** - Full ICM pipeline with all markers
- **`icm_aggregated_markers.yaml`** - Optimized pipeline with ROI aggregation
- **`icm_optimized_markers.yaml`** - Fast pipeline with essential markers
- **`wsmi_only.yaml`** - Connectivity-only analysis
- **`sevo_pipeline.yaml`** - Sevoflurane analysis pipeline

Run with:
```bash
junifer run examples/your_config.yaml
```

## Architecture

```
EEGDataGrabber → Preprocessors → Markers → HDF5FeatureStorage
```

Key preprocessors:
- **ICMEquipmentFilter** - Equipment-specific filtering and resampling
- **ICMLGEpoching** - ICM paradigm epoching with event mapping
- **ICMAdaptiveArtifactRejection** - Adaptive bad channel/epoch detection

## Testing

```bash
# Run all tests
pytest junifer_eeg/tests/

# Run validation tests
pytest junifer_eeg/tests/validation/

# Run with coverage
pytest --cov=junifer_eeg junifer_eeg/tests/
```

## Docker Guide

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for comprehensive Docker usage instructions including:
- Smart script with automatic path mounting
- Manual Docker run examples
- YAML configuration strategies
- Troubleshooting

## Contributing

1. Follow junifer's architectural patterns
2. Add comprehensive tests for new markers
3. Use HDF5 for data storage
4. Ensure proper equipment handling
5. Maintain backward compatibility

