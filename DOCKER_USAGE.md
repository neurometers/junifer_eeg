# Docker Usage Guide for junifer-eeg

## Overview

This guide explains how to use the junifer-eeg Docker container to run your EEG analysis pipelines.

## Building the Image

```bash
docker build -t junifer-eeg:latest .
```

## Running the Container

### Recommended: Using the Smart Script

The `docker-junifer.sh` script automatically parses your YAML file and creates all necessary volume mounts:

```bash
# Automatic path detection and mounting
./docker-junifer.sh run examples/icm_preprocessing_only.yaml
```

**What the script does:**
1. Reads your YAML file
2. Extracts all paths (`datadir`, `uri`, `dump_location`, etc.)
3. Automatically creates volume mounts for each path
4. Handles both relative and absolute paths intelligently
5. Creates output directories if they don't exist
6. Sets up cache directories for templateflow

### Manual Method: Using docker run Directly

If you prefer to manually specify mounts, the container's entrypoint is set to `junifer`:

```bash
# Basic help
docker run --rm junifer-eeg:latest --help
```

#### Running with YAML Configuration Files (Manual Mounts)

To run a pipeline manually, you need to mount:
1. Your YAML configuration file
2. Your data directory
3. Your output directory
4. Cache directory

**Complete manual example:**

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

**For manual mounting, your YAML should reference the mounted paths:**
```yaml
workdir: "."
datagrabber:
  datadir: "/app/data"  # Matches the mount point
preprocess:
  - dump_location: /app/output
storage:
  uri: /app/output/results.h5
```

### Interactive Mode

To explore the container interactively:

```bash
docker run --rm -it \
  -v $(pwd):/app \
  -v $(pwd)/.docker_cache:/cache \
  -e HOME=/cache \
  -w /app \
  --entrypoint /bin/bash \
  junifer-eeg:latest
```

Inside the container, you can then run:
```bash
# Your repo is mounted at /app
junifer run examples/your_config.yaml
```

## Path Configuration Strategies

### Strategy 1: Relative Paths (Recommended)

Use relative paths from the repository root:

```yaml
workdir: "."

datagrabber:
  datadir: "./site/bckp/ground_truth"  # Relative to repo root

preprocess:
  - kind: ICMEquipmentFilter
    dump_location: ./output  # Will be created automatically

storage:
  uri: ./output/results.h5  # Relative output path
```

Then run:
```bash
./docker-junifer.sh run examples/your_config.yaml
```

The script automatically:
- Detects all paths in your YAML
- Mounts `site/bckp/ground_truth` → `/app/site/bckp/ground_truth`
- Creates and mounts `output` → `/app/output`
- Handles everything transparently

### Strategy 2: Container Absolute Paths (For manual docker run)

Use absolute paths inside the container:

```yaml
workdir: "."

datagrabber:
  datadir: "/app/data"

preprocess:
  - dump_location: /app/output

storage:
  uri: /app/output/results.h5
```

Then manually mount:
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd)/examples/config.yaml:/app/config.yaml \
  -v $(pwd)/site/bckp/ground_truth:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/.docker_cache:/cache \
  -e HOME=/cache \
  -e TEMPLATEFLOW_HOME=/cache/templateflow \
  junifer-eeg:latest run /app/config.yaml
```

## Common Commands

### Using docker-junifer.sh (Simple)

```bash
# Check version
./docker-junifer.sh --version

# Get help
./docker-junifer.sh --help

# Run a pipeline
./docker-junifer.sh run examples/your_config.yaml

# List elements (for other junifer commands)
./docker-junifer.sh list-elements --help
```

### Using docker run directly

```bash
# Check version
docker run --rm junifer-eeg:latest --version

# Get help
docker run --rm junifer-eeg:latest --help

# Run selftest
docker run --rm junifer-eeg:latest selftest --help
```

## Docker Compose (Optional)

For repeated runs, you can create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  junifer-eeg:
    image: junifer-eeg:latest
    build: .
    user: "${UID:-1000}:${GID:-1000}"
    volumes:
      - .:/app
      - ./.docker_cache:/cache
    working_dir: /app
    environment:
      - HOME=/cache
      - TEMPLATEFLOW_HOME=/cache/templateflow
      - GIT_AUTHOR_NAME=junifer-docker
      - GIT_AUTHOR_EMAIL=junifer@docker.local
    command: run examples/your_config.yaml
```

Then run:
```bash
docker-compose run --rm junifer-eeg
```

