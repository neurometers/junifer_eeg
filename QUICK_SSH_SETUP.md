# Quick SSH Setup Guide

## TL;DR - Process Data on SSH Without Local Copying

```bash
# 1. Mount SSH directories
sshfs user@server.com:/lab/eeg_data ~/mounts/ssh_data
sshfs user@server.com:/lab/results ~/mounts/ssh_results

# 2. Update your YAML config paths
# datadir: "/home/youruser/mounts/ssh_data"
# uri: /home/youruser/mounts/ssh_results/output.h5

# 3. Run normally - data never touches your local disk!
./docker-junifer.sh run examples/icm_part1_wsmi_bids.yaml

# 4. Results are written directly to SSH mount
ls ~/mounts/ssh_results/sub-*/ses-*/eeg/

# 5. Unmount when done
fusermount -u ~/mounts/ssh_data ~/mounts/ssh_results
```

## ✅ What Works Now

1. **SSH-mounted input data** - Read EEG files from remote server
2. **SSH-mounted output** - Write results directly to remote server
3. **Paths with spaces** - Script properly handles: `/My Data/EEG Files/`
4. **Mixed setups** - Input from SSH, output local (or vice versa)
5. **No local copying** - Data streams directly through Docker

## Setup SSHFS (One-Time)

### Ubuntu/Debian
```bash
sudo apt-get install sshfs
```

### macOS
```bash
brew install macfuse sshfs
```

## Mount Remote Directory

```bash
# Create mount points
mkdir -p ~/mounts/ssh_data ~/mounts/ssh_results

# Mount with good defaults (reconnect on disconnect)
sshfs user@remote.server.com:/path/to/data ~/mounts/ssh_data \
    -o reconnect,ServerAliveInterval=15,compression=yes

sshfs user@remote.server.com:/path/to/results ~/mounts/ssh_results \
    -o reconnect,ServerAliveInterval=15,compression=yes
```

## Update Your YAML Config

Just change the paths to point to your mounts:

```yaml
datagrabber:
  datadir: "/home/youruser/mounts/ssh_data"  # ← SSH mount

storage:
  uri: /home/youruser/mounts/ssh_results/analysis.h5  # ← SSH mount
```

**That's it!** The Docker script handles everything else.

## Example Configs

See:
- `examples/example_ssh_config.yaml` - Template for SSH usage
- `examples/icm_part1_wsmi_bids.yaml` - Just update the paths

## Troubleshooting

### Docker can't access SSH mount

**Fix**: Mount with `allow_other`

```bash
# Add yourself to fuse group (one-time)
sudo usermod -a -G fuse $USER
# Log out and back in

# Mount with allow_other
sshfs user@server:/data ~/mounts/data -o allow_other
```

### Connection drops during long jobs

**Fix**: Use screen/tmux

```bash
screen -S my_job
./docker-junifer.sh run config.yaml
# Detach: Ctrl+A then D
# Reattach: screen -r my_job
```

### Slow performance

**Fixes**:
```bash
# Use compression
sshfs ... -o compression=yes

# Enable caching
sshfs ... -o cache=yes,kernel_cache

# Combine both
sshfs user@server:/data ~/mounts/data \
    -o reconnect,compression=yes,cache=yes,kernel_cache
```

## Unmount

```bash
fusermount -u ~/mounts/ssh_data
fusermount -u ~/mounts/ssh_results
```

## Complete Example

```bash
# Mount
sshfs me@cluster.edu:/lab/data ~/data -o reconnect,compression=yes
sshfs me@cluster.edu:/lab/out ~/results -o reconnect,compression=yes

# Update YAML (just change paths to ~/data and ~/results)

# Run
./docker-junifer.sh run examples/icm_part3_fast_bids.yaml

# Check results on remote
ls ~/results/sub-*/ses-*/eeg/*.pkl

# Unmount
fusermount -u ~/data ~/results
```

**No data copied to your local drive!** Everything processes directly on SSH mounts.
