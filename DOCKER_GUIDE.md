# Docker Guide for junifer-eeg

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t junifer-eeg:latest .
```

### 2. Run with the Smart Script (Recommended)

The `docker-junifer.sh` script automatically parses your YAML file and mounts all required directories.

```bash
# Run a pipeline - automatic path mounting
./docker-junifer.sh run examples/icm_preprocessing_only.yaml

# Run with HDF5 to pickle conversion
./docker-junifer.sh run examples/icm_preprocessing_only.yaml --dump

# Check version
./docker-junifer.sh --version

# Get help
./docker-junifer.sh --help

# List elements
./docker-junifer.sh list-elements --help
```

**What the script does:**
- Parses your YAML file to extract all paths (`datadir`, `uri`, `dump_location`, `workdir`)
- Automatically creates volume mounts for each path
- Handles both relative and absolute paths:
  - Relative paths (`./data`) → Mounted from repo root
  - Absolute paths (`/home/user/data`) → Mounted at same location
- Creates output directories if they don't exist
- Sets up cache directories for templateflow
- Runs with your user ID to avoid permission issues

### 3. Manual Docker Run

If you prefer manual control:

```bash
# Basic help
docker run --rm junifer-eeg:latest --help

# Run with manual mounts
docker run --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/examples/config.yaml:/app/config.yaml \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/.docker_cache:/cache \
    -e HOME=/cache \
    -e TEMPLATEFLOW_HOME=/cache/templateflow \
    junifer-eeg:latest run /app/config.yaml
```

## YAML Configuration

### Using Relative Paths (Recommended)

```yaml
workdir: "."

datagrabber:
  datadir: "./site/bckp/ground_truth"

preprocess:
  - kind: ICMEquipmentFilter
    dump_location: ./output

storage:
  uri: ./output/results.h5
```

### Using Container Paths (for manual docker run)

```yaml
workdir: "."

datagrabber:
  datadir: "/app/data"

preprocess:
  - dump_location: /app/output

storage:
  uri: /app/output/results.h5
```

## Script Parameters

The `docker-junifer.sh` script supports:

- **Commands**: `run`, `--help`, `--version`, `selftest`, `list-elements`
- **--dump flag**: After pipeline completion, converts HDF5 output to pickle format and deletes the HDF5 file
- **Automatic path detection**: Extracts and mounts all paths from your YAML
- **User mapping**: Runs with your user ID to prevent permission issues

## Interactive Mode

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

Inside container:
```bash
junifer run examples/your_config.yaml
```

## Docker Compose (Optional)

Create `docker-compose.yml`:

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

Run:
```bash
docker-compose run --rm junifer-eeg
```

## Troubleshooting

- **"Permission denied"**: The script handles this automatically with user mapping
- **"Image not found"**: Run `docker build -t junifer-eeg:latest .`
- **Update dependencies**: Rebuild with `docker build --no-cache -t junifer-eeg:latest .`
- **Git warnings**: Informational only, won't affect analysis

## Key Features

1. **No local Python needed** - Everything runs in the container
2. **Automatic path mounting** - No manual volume specification required
3. **Reproducible environment** - Same versions every time
4. **Clean isolation** - No system package conflicts
5. **Easy sharing** - Others can use the same Docker image
