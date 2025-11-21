# Docker Quick Start Guide

## Quick Start

### 1. Build the Docker Image

```bash
cd /home/gio/Documents/Repos/junifer_eeg
docker build -t junifer-eeg:latest .
```

### 2. Run Commands

#### Using the convenience script (recommended - automatic path mounting):

The script automatically parses your YAML file and mounts all required directories!

```bash
# Run a pipeline - the script handles all mounting automatically
./docker-junifer.sh run examples/icm_preprocessing_only.yaml

# Show help
./docker-junifer.sh --help

# Show version
./docker-junifer.sh --version

# List elements
./docker-junifer.sh list-elements --help
```

**How it works:** The script reads your YAML file and automatically creates volume mounts for:
- `datadir` (where your EEG data is located)
- `uri` (storage output location)
- `dump_location` (preprocessor output directories)
- Any other paths specified in your YAML

**Works with both relative and absolute paths!**

#### Using docker run directly (manual mounting):

If you prefer to manually specify mounts:

```bash
# Show help
docker run --rm junifer-eeg:latest --help

# Run with specific mounts
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

## How It Works

### The docker-junifer.sh script (Automatic Mode):
When you run `./docker-junifer.sh run your_config.yaml`, the script:

1. **Parses your YAML file** to find all paths (`datadir`, `uri`, `dump_location`, etc.)
2. **Automatically creates volume mounts** for each path found
3. **Handles both relative and absolute paths**:
   - Relative paths (like `./data` or `output/`) → Mounted from repo root
   - Absolute paths (like `/home/user/data`) → Mounted at same location in container
4. **Sets up cache directories** for templateflow and other dependencies
5. **Configures Git** to avoid warnings
6. **Runs with your user ID** to avoid permission issues

### Key Features:
1. **No local Python installation needed** - Everything runs in the container
2. **Automatic path detection** - No need to manually specify volume mounts
3. **Reproducible environment** - Same versions every time
4. **Clean isolation** - No conflicts with your system packages
5. **Easy sharing** - Others can use the same Docker image


## Troubleshooting

### "Permission denied" errors
The `docker-junifer.sh` script handles this automatically by running with your user ID.

### "Git configuration" warnings
These are informational and won't affect your analysis. The script sets basic Git config to minimize warnings.

### "Image not found"
Run: `docker build -t junifer-eeg:latest .`

### Need to update dependencies?
Rebuild the image: `docker build --no-cache -t junifer-eeg:latest .`

For more detailed information, see [DOCKER_USAGE.md](DOCKER_USAGE.md).

