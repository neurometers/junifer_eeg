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

# Function to mount path appropriately (handles spaces and SSH mounts)
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
        mkdir -p "$host_path" 2>/dev/null || true
        
        # Use proper quoting for paths with spaces
        echo "-v \"$host_path:$container_path\""
    else
        # Absolute path (could be SSH mount)
        local host_path="$path"
        
        # Check if this absolute path is within the repo
        if [[ "$path" == "$REPO_ROOT"* ]]; then
            # Path is within repo - mount at same absolute path in container
            local container_path="$path"
            
            # Create directory if it doesn't exist
            if [ ! -e "$host_path" ]; then
                mkdir -p "$host_path" 2>/dev/null || true
            fi
            
            # Use proper quoting for paths with spaces
            echo "-v \"$host_path:$container_path\""
        else
            # Path is outside repo (SSH mount, network drive, etc.)
            local container_path="$path"
            
            # Check if path exists OR is potentially an SSH mount
            # SSH mounts might not show up with -e check immediately
            if [ -e "$host_path" ] || [ -d "$(dirname "$host_path")" ]; then
                echo "-v \"$host_path:$container_path\""
            else
                echo "Warning: Path not accessible: $host_path" >&2
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
        
        # Mount the YAML file (with proper quoting for spaces)
        DOCKER_CMD="$DOCKER_CMD -v \"$YAML_FILE:/app/config.yaml\""
        
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
        echo "Note: If using SSH-mounted paths, ensure the mount is active"
        exit 1
    fi
else
    # For non-run commands, use simple mounting (with proper quoting)
    DOCKER_CMD="$DOCKER_CMD -v \"$REPO_ROOT:/workspace\""
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
    echo "=== Dumping HDF5 to Pickle (BIDS format) ==="
    
    # Extract output directory from YAML
    OUTPUT_URI=$(grep -E '^\s*uri:' "$YAML_FILE" | sed -E 's/.*uri:\s*["'"'"']?([^"'"'"']+)["'"'"']?.*/\1/' | head -1 | tr -d ' ')
    
    if [ -z "$OUTPUT_URI" ]; then
        echo "Warning: Could not find 'uri:' in YAML file. Skipping dump."
    else
        # Convert relative path to absolute
        if [[ "$OUTPUT_URI" != /* ]]; then
            OUTPUT_URI="$REPO_ROOT/$OUTPUT_URI"
        fi
        
        # Get the output directory (may be a file path, so get parent)
        if [ -f "$OUTPUT_URI" ]; then
            OUTPUT_DIR="$(dirname "$OUTPUT_URI")"
        elif [ -d "$OUTPUT_URI" ]; then
            OUTPUT_DIR="$OUTPUT_URI"
        else
            OUTPUT_DIR="$(dirname "$OUTPUT_URI")"
        fi
        
        # Check if the directory exists
        if [ ! -d "$OUTPUT_DIR" ]; then
            echo "Warning: Output directory not found: $OUTPUT_DIR. Skipping dump."
        else
            # Find all HDF5 files in the output directory
            H5_FILES=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.h5" -type f)
            
            if [ -z "$H5_FILES" ]; then
                echo "Warning: No HDF5 files found in $OUTPUT_DIR. Skipping dump."
            else
                echo "Found HDF5 files in: $OUTPUT_DIR"
                
                # Process each HDF5 file
                while IFS= read -r H5_FILE; do
                    # Create pickle filename (same path, .pkl extension)
                    PKL_FILE="${H5_FILE%.h5}.pkl"
                    
                    H5_DIR="$(dirname "$H5_FILE")"
                    H5_NAME="$(basename "$H5_FILE")"
                    PKL_NAME="$(basename "$PKL_FILE")"
                    
                    echo ""
                    echo "Converting: $H5_NAME"
                    
                    # Run the dump inside Docker
                    docker run --rm \
                        --user "$(id -u):$(id -g)" \
                        -v "$H5_DIR:/data" \
                        -v "$REPO_ROOT/junifer_eeg:/app/junifer_eeg:ro" \
                        -v "$CACHE_DIR:/cache" \
                        -e HOME=/cache \
                        --entrypoint python \
                        "$IMAGE_NAME" \
                        -c "
from junifer_eeg.reader import read_h5
reader = read_h5('/data/$H5_NAME')
markers = reader.list_markers()
print(f'Found {len(markers)} markers: {\", \".join(markers)}')
reader.dump_all_to_pkl('/data/$PKL_NAME')
print('Successfully dumped all markers to pickle file')
"
                    
                    # Check if pickle was created successfully
                    if [ -f "$PKL_FILE" ]; then
                        echo "✓ Created: $PKL_NAME"
                        
                        # Delete HDF5 file
                        rm "$H5_FILE"
                        echo "✓ Deleted: $H5_NAME"
                    else
                        echo "✗ Error: Pickle file was not created for $H5_NAME. Keeping HDF5 file."
                    fi
                done <<< "$H5_FILES"
                
                echo ""
                echo "=== Dump Complete ==="
                echo "All pickle files saved to: $OUTPUT_DIR"
            fi
        fi
    fi
fi

