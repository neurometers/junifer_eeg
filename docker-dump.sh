#!/bin/bash
# Dump junifer HDF5 file to pickle format using Docker
# Usage: ./docker-dump.sh <input.h5> <output.pkl>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="junifer-eeg:latest"

# Show help
if [ $# -lt 2 ]; then
    echo "Junifer HDF5 to Pickle Dumper"
    echo ""
    echo "Usage: $0 <input.h5> <output.pkl>"
    echo ""
    echo "Arguments:"
    echo "  input.h5    Path to the HDF5 file to read"
    echo "  output.pkl  Path where the pickle file will be saved"
    echo ""
    echo "Example:"
    echo "  $0 output/results.h5 output/results.pkl"
    echo ""
    echo "The output pickle file contains all markers with:"
    echo "  - data: numpy arrays"
    echo "  - dimensions: description of what each axis represents"
    echo "  - labels: channel names, band names, etc."
    echo "  - summary: human-readable description"
    exit 0
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Error: Docker image '$IMAGE_NAME' not found."
    echo "Build it with: docker build -t $IMAGE_NAME ."
    exit 1
fi

H5_FILE="$1"
PKL_FILE="$2"

# Check input file exists
if [ ! -f "$H5_FILE" ]; then
    echo "Error: Input file not found: $H5_FILE"
    exit 1
fi

# Get absolute paths
H5_FILE_ABS="$(cd "$(dirname "$H5_FILE")" && pwd)/$(basename "$H5_FILE")"
H5_DIR="$(dirname "$H5_FILE_ABS")"
H5_NAME="$(basename "$H5_FILE_ABS")"

# Handle output path (may not exist yet)
PKL_DIR="$(cd "$(dirname "$PKL_FILE")" 2>/dev/null && pwd || echo "$(pwd)/$(dirname "$PKL_FILE")")"
PKL_NAME="$(basename "$PKL_FILE")"

# Create output directory if needed
mkdir -p "$PKL_DIR"

echo "Reading: $H5_FILE"
echo "Writing: $PKL_DIR/$PKL_NAME"

# Run the dump inside Docker
# Mount local junifer_eeg to use latest code
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$H5_DIR:/input:ro" \
    -v "$PKL_DIR:/output" \
    -v "$SCRIPT_DIR/junifer_eeg:/app/junifer_eeg:ro" \
    -v "$SCRIPT_DIR/.docker_cache:/cache" \
    -e HOME=/cache \
    --entrypoint python \
    "$IMAGE_NAME" \
    -c "
from junifer_eeg.reader import read_h5
reader = read_h5('/input/$H5_NAME')
result = reader.dump_to_pkl('/output/$PKL_NAME')
print(result['summary'])
"

echo ""
echo "Done! Pickle file saved to: $PKL_DIR/$PKL_NAME"
