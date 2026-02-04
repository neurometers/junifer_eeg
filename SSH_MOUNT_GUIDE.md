# Using junifer-eeg with SSH-Mounted Data

## Overview

You can process data directly from SSH-mounted directories without copying to local disk. Both input data and output can be on SSH mounts.

## Setup SSH Mount (SSHFS)

### Install SSHFS

```bash
# Ubuntu/Debian
sudo apt-get install sshfs

# macOS
brew install macfuse sshfs
```

### Mount Remote Directory

```bash
# Create local mount point
mkdir -p ~/mounts/remote_data

# Mount SSH directory
sshfs user@remote.server.com:/path/to/data ~/mounts/remote_data

# Or with specific options for better performance
sshfs user@remote.server.com:/path/to/data ~/mounts/remote_data \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    -o cache=yes,kernel_cache,compression=yes
```

### Verify Mount

```bash
# Check if mounted
mount | grep sshfs

# Test access
ls ~/mounts/remote_data
```

## Configure YAML for SSH Paths

### Example 1: SSH Input, Local Output

```yaml
datagrabber:
  kind: EEGDataGrabber
  patterns:
    EEG:
      pattern: "{subject}/epochs/{subject}_ses-{session}_task-{task}_acq-{acq}_epo.fif"
      space: "native"
  replacements: ["subject", "session", "task", "acq"]
  types:
    - EEG
  datadir: "/home/user/mounts/remote_data"  # SSH-mounted input

storage:
  kind: BIDSFeatureStorage
  uri: output/analysis.h5  # Local output (relative to repo)
  suffix: markers
  single_output: false
  output_pickle: true
```

### Example 2: SSH Input and SSH Output

```yaml
datagrabber:
  kind: EEGDataGrabber
  patterns:
    EEG:
      pattern: "{subject}/epochs/{subject}_ses-{session}_task-{task}_acq-{acq}_epo.fif"
      space: "native"
  replacements: ["subject", "session", "task", "acq"]
  types:
    - EEG
  datadir: "/home/user/mounts/remote_data"  # SSH-mounted input

storage:
  kind: BIDSFeatureStorage
  uri: /home/user/mounts/remote_output/analysis.h5  # SSH-mounted output
  suffix: markers
  single_output: false
  output_pickle: true
```

### Example 3: Paths with Spaces

The script now handles spaces in paths:

```yaml
datagrabber:
  kind: EEGDataGrabber
  datadir: "/home/user/My Data/EEG Studies/Study 01"  # Spaces work fine

storage:
  kind: BIDSFeatureStorage
  uri: "/home/user/My Results/BIDS Output/analysis.h5"  # Spaces handled
```

## Run Pipeline with SSH Data

```bash
# Standard run
./docker-junifer.sh run examples/icm_part1_wsmi_bids.yaml

# The script will automatically:
# 1. Detect SSH-mounted paths
# 2. Mount them in Docker container
# 3. Process data directly from SSH
# 4. Write results to SSH (if configured)
```

## Performance Tips

### 1. Use Compression

```bash
sshfs user@server:/data ~/mounts/data -o compression=yes
```

### 2. Enable Caching

```bash
sshfs user@server:/data ~/mounts/data -o cache=yes,kernel_cache
```

### 3. Increase Buffer Size

```bash
sshfs user@server:/data ~/mounts/data -o cache_timeout=115200
```

### 4. Use Parallel Processing

For large datasets, process subjects in parallel:

```bash
# Create separate configs for different subjects
# Run them in parallel in different terminals
./docker-junifer.sh run config_subjects_01-10.yaml &
./docker-junifer.sh run config_subjects_11-20.yaml &
```

## Troubleshooting

### Mount Not Accessible in Docker

**Problem**: Docker can't access SSHFS mount

**Solution**: Mount with `allow_other` option:

```bash
# Add user to fuse group first
sudo usermod -a -G fuse $USER

# Mount with allow_other
sshfs user@server:/data ~/mounts/data -o allow_other
```

### Slow Performance

**Problem**: Processing is very slow

**Solutions**:
1. Enable compression and caching (see above)
2. Consider copying data locally for very large datasets
3. Use faster network connection
4. Process smaller batches

### Connection Drops

**Problem**: SSH connection drops during processing

**Solution**: Use reconnect and keepalive options:

```bash
sshfs user@server:/data ~/mounts/data \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

### Permission Denied

**Problem**: Docker can't read/write to SSH mount

**Solution**: Check permissions and use `allow_other`:

```bash
# Unmount if mounted
fusermount -u ~/mounts/data

# Remount with proper permissions
sshfs user@server:/data ~/mounts/data \
    -o allow_other,default_permissions,uid=$(id -u),gid=$(id -g)
```

## Unmount When Done

```bash
# Unmount SSH directory
fusermount -u ~/mounts/remote_data

# Or on macOS
umount ~/mounts/remote_data
```

## Best Practices

1. **Test connection first**: Verify SSH mount works before running pipeline
2. **Use absolute paths**: Always use full paths for SSH mounts in YAML
3. **Monitor space**: Check available space on remote storage
4. **Backup important data**: Keep copies of original data
5. **Use screen/tmux**: For long-running jobs, use screen or tmux:
   ```bash
   screen -S junifer_job
   ./docker-junifer.sh run config.yaml
   # Detach with Ctrl+A, D
   ```

## Complete Example Workflow

```bash
# 1. Mount remote data
mkdir -p ~/ssh_data ~/ssh_results
sshfs user@cluster.university.edu:/lab/eeg_data ~/ssh_data -o reconnect,compression=yes
sshfs user@cluster.university.edu:/lab/results ~/ssh_results -o reconnect,compression=yes

# 2. Verify mounts
ls ~/ssh_data
ls ~/ssh_results

# 3. Create config with SSH paths
cat > my_ssh_config.yaml << 'EOF'
with:
  - "junifer_eeg"

datagrabber:
  kind: EEGDataGrabber
  patterns:
    EEG:
      pattern: "{subject}/epochs/{subject}_ses-{session}_task-{task}_acq-{acq}_epo.fif"
      space: "native"
  replacements: ["subject", "session", "task", "acq"]
  types:
    - EEG
  datadir: "/home/user/ssh_data"

storage:
  kind: BIDSFeatureStorage
  uri: /home/user/ssh_results/bids_output.h5
  suffix: markers
  single_output: false
  output_pickle: true
  delete_h5_after_pickle: false

markers:
  - kind: KolmogorovComplexity
    name: kolmogorov
    channel_method: "mean"
    trial_method: "mean"
    on: EEG
EOF

# 4. Run pipeline (data never touches local disk!)
./docker-junifer.sh run my_ssh_config.yaml

# 5. Check results (on remote server)
ls ~/ssh_results/sub-*/ses-*/eeg/

# 6. Unmount when done
fusermount -u ~/ssh_data
fusermount -u ~/ssh_results
```

## Security Notes

- Use SSH keys instead of passwords for automation
- Consider using SSH config file for connection settings
- Ensure proper permissions on both local and remote systems
- Be aware of data transfer over network (use VPN if needed)

## Network Storage Alternatives

Besides SSHFS, you can also use:

- **NFS**: Network File System
- **SMB/CIFS**: Windows network shares
- **S3FS**: Amazon S3 buckets
- **rclone**: Various cloud storage backends

All work the same way once mounted - just provide the mount point path in your YAML config.
