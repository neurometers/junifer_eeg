# JUnifer EEG

JUnifer EEG is an extension for the JUelich NeuroImaging FEature extractoR (junifer) that provides EEG-specific components for electrophysiological data analysis.

## Overview

This extension adds EEG support to junifer by:

- **Custom datagrabber** - `EEGDataGrabber` with EEG file format support (.vhdr, .edf, .fif) via global reader registry
- **Equipment detection** - Automatic detection and montage setting for Biosemi, BrainVision, EGI systems
- **Preprocessing** - Filtering, epoching, artifact rejection, and interpolation
- **Feature extraction** - Spectral power, connectivity, complexity, and ERP markers
- **HDF5 storage** - Uses junifer's recommended storage format

## Architecture

This extension follows junifer's pattern:

```
EEGDataGrabber → Preprocessors → Markers → HDF5FeatureStorage
```

The global reader registry automatically handles montage setting for all EEG formats, enabling CSD computation and proper channel localization.

## Installation

### Option 1: Docker (Recommended)

The easiest way to use junifer-eeg is with Docker. The smart `docker-junifer.sh` script automatically detects and mounts all required paths from your YAML file!

```bash
# Build the Docker image (one-time)
docker build -t junifer-eeg:latest .

# Run with the smart script - works with ANY yaml!
./docker-junifer.sh run examples/icm_preprocessing_only.yaml
```

**Two Ways to Run Docker:**

#### Smart Script (Easiest) - Works with Your Existing YAMLs
Use your existing YAML files as-is, even with absolute paths:

```bash
# Your local YAML with absolute paths - works directly!
./docker-junifer.sh run examples/icm_preprocessing_only.yaml
```

The script automatically:
- Parses your YAML to find all paths (`datadir`, `uri`, `dump_location`, etc.)
- Creates the necessary Docker volume mounts
- Handles both relative and absolute paths intelligently
- Creates output directories if needed

#### Manual Docker Run (For Custom Setups)
For manual control, create a Docker-specific YAML with relative paths:

```yaml
# examples/icm_preprocessing_only_docker.yaml
datadir: "./data"
dump_location: ./output
uri: ./output/results.h5
```

Then run with manual mounts:
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd)/examples/icm_preprocessing_only_docker.yaml:/app/config.yaml \
  -v $(pwd)/site/bckp/ground_truth:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/.docker_cache:/cache \
  -e HOME=/cache \
  -e TEMPLATEFLOW_HOME=/cache/templateflow \
  junifer-eeg:latest run /app/config.yaml
```

**See [DOCKER_USAGE.md](DOCKER_USAGE.md) for detailed Docker instructions and examples.**

### Option 2: Local Installation

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

## Quick Start

### Docker Quick Start

```bash
# 1. Build the image (first time only)
docker build -t junifer-eeg:latest .

# 2. Run with your existing YAML
./docker-junifer.sh run examples/icm_preprocessing_only.yaml

# 3. Check results
ls -lh examples/  # Output files will be here
```

### Local Quick Start

1. **Install**: Follow local installation instructions above
2. **Prepare Data**: Use EEG in .mff / .edf / .fif
3. **Copy Example**: `cp examples/icm_complete_individual_markers.yaml my_analysis.yaml`
4. **Edit Config**: Update `my_analysis.yaml` with your desired path, preprocessing and markers
5. **Run**: `junifer run my_analysis.yaml`
6. **Results**: Features saved in the path specified in your yaml

## Example YAML Files

The `examples/` directory contains two versions of configuration files to demonstrate different usage patterns:

### For Local/Smart Docker Script Use
- **`icm_preprocessing_only.yaml`** - Uses absolute paths (your normal workflow)
- **`icm_complete_individual_markers.yaml`** - Full pipeline with local paths
- **`icm_gamma_21.yaml`** - Gamma band analysis example

**These work directly with:**
```bash
# Local
junifer run examples/icm_preprocessing_only.yaml

# Docker smart script
./docker-junifer.sh run examples/icm_preprocessing_only.yaml
```

### For Manual Docker Run
- **`icm_preprocessing_only_docker.yaml`** - Uses relative paths like `./data`, `./output`

**For manual docker run with custom mounts:**
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd)/examples/icm_preprocessing_only_docker.yaml:/app/config.yaml \
  -v $(pwd)/site/bckp/ground_truth:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/.docker_cache:/cache \
  -e HOME=/cache \
  -e TEMPLATEFLOW_HOME=/cache/templateflow \
  junifer-eeg:latest run /app/config.yaml
```

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



