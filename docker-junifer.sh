#!/bin/bash

# docker-junifer.sh - Convenience script to run junifer-eeg in Docker
# 
# Usage:
#   ./docker-junifer.sh run path/to/config.yaml [--dump]
#   ./docker-junifer.sh --help
#   ./docker-junifer.sh selftest
#
# Options:
#   --dump    After running the pipeline, convert HDF5 output to pickle and delete HDF5

set -e

# Configuration
IMAGE_NAME="junifer-eeg:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP_MODE=false

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if image exists, if not, build it
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Image $IMAGE_NAME not found. Building..."
    docker build -t "$IMAGE_NAME" "$REPO_ROOT"
fi

# Parse arguments
COMMAND="$1"
shift || true

# Check for --dump flag in remaining arguments
REMAINING_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dump)
            DUMP_MODE=true
            shift
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${REMAINING_ARGS[@]}"

# Function to extract paths from YAML
extract_paths_from_yaml() {
    local yaml_file="$1"
    local paths=()
    
    if [ ! -f "$yaml_file" ]; then
        return
    fi
    
    # Extract datadir (look for 'datadir:' line)
    local datadir=$(grep -E '^\s*datadir:' "$yaml_file" | sed -E 's/.*datadir:\s*["'"'"']?([^"'"'"']+)["'"'"']?.*/\1/' | tr -d ' ')
    [ -n "$datadir" ] && paths+=("$datadir")
    
    # Extract storage uri (look for 'uri:' line)
    local uri=$(grep -E '^\s*uri:' "$yaml_file" | sed -E 's/.*uri:\s*["'"'"']?([^"'"'"']+)["'"'"']?.*/\1/' | head -1 | tr -d ' ')
    [ -n "$uri" ] && paths+=("$(dirname "$uri")")
    
    # Extract dump_location (look for 'dump_location:' lines)
    while IFS= read -r dump_loc; do
        dump_loc=$(echo "$dump_loc" | sed -E 's/.*dump_location:\s*["'"'"']?([^"'"'"']+)["'"'"']?.*/\1/' | tr -d ' ')
        [ -n "$dump_loc" ] && paths+=("$dump_loc")
    done < <(grep -E '^\s*dump_location:' "$yaml_file")
    
    # Extract workdir
    local workdir=$(grep -E '^\s*workdir:' "$yaml_file" | sed -E 's/.*workdir:\s*["'"'"']?([^"'"'"']+)["'"'"']?.*/\1/' | tr -d ' ')
    [ -n "$workdir" ] && [ "$workdir" != "." ] && paths+=("$workdir")
    
    # Return unique paths
    printf '%s\n' "${paths[@]}" | sort -u
}

# Function to mount path appropriately
mount_path() {
    local path="$1"
    
    # Remove leading/trailing spaces and quotes
    path=$(echo "$path" | sed -E 's/^[[:space:]"'"'"']+|[[:space:]"'"'"']+$//g')
    
    if [ -z "$path" ] || [ "$path" = "." ]; then
        return
    fi
    
    # Handle relative paths
    if [[ "$path" != /* ]]; then
        # Relative path - convert to absolute from repo root
        path="${path#./}"  # Remove leading ./
        local host_path="$REPO_ROOT/$path"
        local container_path="/app/$path"
        
        # Create directory if it doesn't exist
        mkdir -p "$host_path"
        
        echo "-v $host_path:$container_path"
    else
        # Absolute path
        local host_path="$path"
        
        # Check if this absolute path is within the repo
        if [[ "$path" == "$REPO_ROOT"* ]]; then
            # Path is within repo - mount at same absolute path in container
            local container_path="$path"
            
            # Create directory if it doesn't exist
            if [ ! -e "$host_path" ]; then
                mkdir -p "$host_path" 2>/dev/null || true
            fi
            
            echo "-v $host_path:$container_path"
        else
            # Path is outside repo - mount at same location
            local container_path="$path"
            
            # Only mount if path exists on host
            if [ -e "$host_path" ]; then
                echo "-v $host_path:$container_path"
            fi
        fi
    fi
}

# Build docker run command
DOCKER_CMD="docker run --rm"

# Add user mapping to avoid permission issues
DOCKER_CMD="$DOCKER_CMD --user $(id -u):$(id -g)"

# Create cache directory if it doesn't exist
CACHE_DIR="$REPO_ROOT/.docker_cache"
mkdir -p "$CACHE_DIR"

# Mount cache directory
DOCKER_CMD="$DOCKER_CMD -v $CACHE_DIR:/cache"

# Set HOME to writable location for cache files
DOCKER_CMD="$DOCKER_CMD -e HOME=/cache"

# Set templateflow cache location
DOCKER_CMD="$DOCKER_CMD -e TEMPLATEFLOW_HOME=/cache/templateflow"

# Add Git configuration to avoid warnings
DOCKER_CMD="$DOCKER_CMD -e GIT_AUTHOR_NAME='junifer-eeg-docker'"
DOCKER_CMD="$DOCKER_CMD -e GIT_AUTHOR_EMAIL='junifer@docker.local'"
DOCKER_CMD="$DOCKER_CMD -e GIT_COMMITTER_NAME='junifer-eeg-docker'"
DOCKER_CMD="$DOCKER_CMD -e GIT_COMMITTER_EMAIL='junifer@docker.local'"

# If running 'run' command, parse YAML and mount required paths
if [ "$COMMAND" = "run" ] && [ -n "$1" ]; then
    YAML_FILE="$1"
    
    # Convert to absolute path if relative
    if [[ "$YAML_FILE" != /* ]]; then
        YAML_FILE="$REPO_ROOT/$YAML_FILE"
    fi
    
    if [ -f "$YAML_FILE" ]; then
        echo "Parsing YAML file: $YAML_FILE"
        
        # Mount the YAML file
        DOCKER_CMD="$DOCKER_CMD -v $YAML_FILE:/app/config.yaml"
        
        # Extract and mount all required paths
        while IFS= read -r path; do
            mount_cmd=$(mount_path "$path")
            [ -n "$mount_cmd" ] && DOCKER_CMD="$DOCKER_CMD $mount_cmd"
        done < <(extract_paths_from_yaml "$YAML_FILE")
        
        # Set working directory to /app
        DOCKER_CMD="$DOCKER_CMD -w /app"
        
        # Add the image name
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        # Add the command with mounted config path
        DOCKER_CMD="$DOCKER_CMD run /app/config.yaml"
        
        # Shift to remove the yaml file from arguments (already added)
        shift
        
        # Add any remaining arguments
        [ $# -gt 0 ] && DOCKER_CMD="$DOCKER_CMD $@"
    else
        echo "Error: YAML file not found: $YAML_FILE"
        exit 1
    fi
else
    # For non-run commands, use simple mounting
    DOCKER_CMD="$DOCKER_CMD -v $REPO_ROOT:/workspace"
    DOCKER_CMD="$DOCKER_CMD -w /workspace"
    DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
    
    # Add the command and remaining arguments
    if [ -z "$COMMAND" ]; then
        DOCKER_CMD="$DOCKER_CMD --help"
    else
        DOCKER_CMD="$DOCKER_CMD $COMMAND $@"
    fi
fi

# Print the command (for debugging)
echo "Running: $DOCKER_CMD"
echo ""

# Execute the command
eval $DOCKER_CMD

# If --dump flag was set and we ran the 'run' command, dump HDF5 to pickle
if [ "$DUMP_MODE" = true ] && [ "$COMMAND" = "run" ] && [ -n "$YAML_FILE" ]; then
    echo ""
    echo "=== Dumping HDF5 to Pickle ==="
    
    # Extract HDF5 output path from YAML
    H5_FILE=$(grep -E '^\s*uri:' "$YAML_FILE" | sed -E 's/.*uri:\s*["'"'"']?([^"'"'"']+)["'"'"']?.*/\1/' | head -1 | tr -d ' ')
    
    if [ -z "$H5_FILE" ]; then
        echo "Warning: Could not find 'uri:' in YAML file. Skipping dump."
    else
        # Convert relative path to absolute
        if [[ "$H5_FILE" != /* ]]; then
            H5_FILE="$REPO_ROOT/$H5_FILE"
        fi
        
        if [ ! -f "$H5_FILE" ]; then
            echo "Warning: HDF5 file not found: $H5_FILE. Skipping dump."
        else
            # Create pickle filename (same path, .pkl extension)
            PKL_FILE="${H5_FILE%.h5}.pkl"
            
            H5_FILE_ABS="$H5_FILE"
            H5_DIR="$(dirname "$H5_FILE_ABS")"
            H5_NAME="$(basename "$H5_FILE_ABS")"
            PKL_DIR="$(dirname "$PKL_FILE")"
            PKL_NAME="$(basename "$PKL_FILE")"
            
            echo "Reading: $H5_FILE"
            echo "Writing: $PKL_FILE"
            
            # Run the dump inside Docker
            docker run --rm \
                --user "$(id -u):$(id -g)" \
                -v "$H5_DIR:/input:ro" \
                -v "$PKL_DIR:/output" \
                -v "$REPO_ROOT/junifer_eeg:/app/junifer_eeg:ro" \
                -v "$CACHE_DIR:/cache" \
                -e HOME=/cache \
                --entrypoint python \
                "$IMAGE_NAME" \
                -c "
from junifer_eeg.reader import read_h5
reader = read_h5('/input/$H5_NAME')
markers = reader.list_markers()
print(f'Found {len(markers)} markers: {', '.join(markers)}')
reader.dump_all_to_pkl('/output/$PKL_NAME')
print('Successfully dumped all markers to pickle file')
"
            
            # Delete HDF5 file
            if [ -f "$PKL_FILE" ]; then
                echo ""
                echo "Pickle file created successfully. Deleting HDF5 file..."
                rm "$H5_FILE"
                echo "Deleted: $H5_FILE"
                echo ""
                echo "Done! Pickle file saved to: $PKL_FILE"
            else
                echo "Error: Pickle file was not created. Keeping HDF5 file."
                exit 1
            fi
        fi
    fi
fi

