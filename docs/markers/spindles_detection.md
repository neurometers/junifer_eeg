# SpindlesDetection

`junifer_eeg.markers.SpindlesDetection`

Detects **sleep spindles** in EEG epochs using [YASA](https://raphaelvallat.com/yasa/build/html/index.html) and returns one requested feature per epoch and channel.

Detection runs once per unique parameter set and epochs object. Uses a **singleton with internal caching** (`SpindlesDetectionBase`): multiple `SpindlesDetection` markers requesting different features on the same epochs with the same detection parameters will reuse the cached detection results.

---

## Parameters

| Parameter            | Type             | Default          | Description |
|----------------------|------------------|------------------|-------------|
| `feature`            | `str`            | **required**     | Feature to extract. Must be exactly one of: `"Duration"`, `"Amplitude"`, `"Frequency"`, `"Density"`. Raises `ValueError` at initialisation if missing or invalid. |
| `freq_sp`            | `tuple of float` | `(12, 15)`       | Spindle frequency range in Hz `(fmin, fmax)` passed to YASA. |
| `freq_broad`         | `tuple of float` | `(1, 30)`        | Broad band frequency range in Hz `(fmin, fmax)` used by YASA for normalisation. |
| `duration`           | `tuple of float` | `(0.5, 2.5)`     | Minimum and maximum spindle duration in seconds `(min, max)`. |
| `min_distance`       | `int`            | `500`            | Minimum distance between two spindles in milliseconds. |
| `thresh_rms`         | `float`          | `1.5`            | RMS threshold expressed as standard deviations above the mean. |
| `thresh_corr`        | `float`          | `0.65`           | Correlation threshold for spindle detection. |
| `reference_channels` | `tuple of str`   | `("TP7", "TP8")` | Channel names used for re-referencing before detection. The mean of these channels is subtracted from all channels. Falls back to unreferenced data if channels are missing or contain all zeros. |
| `channel_method`     | `str` or `None`  | `None`           | Aggregation method applied **across channels**. If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`       | `str` or `None`  | `None`           | Aggregation method applied **across epochs/trials**. If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `equipment`          | `str`            | `"egi256"`       | Equipment configuration. |
| `on`                 | `str` or `None`  | `"EEG"`          | Data type to apply this marker to. |
| `name`               | `str` or `None`  | `None`           | Name for the marker. If `None`, the class name is used. |

---

## Input

- **Required type:** `mne.BaseEpochs` (data must have been epoched in preprocessing)
- Epochs must be non-empty

---

## Output

Feature name: **`"spindlesdetection"`**

The output is always a 2D tensor `(n_epochs, n_channels)`. Cells with no detected spindles for a given epoch/channel are set to `NaN`.

Output shape depends on which aggregations are applied:

| `channel_method` | `trial_method` | Output shape |
|------------------|----------------|--------------|
| `None`           | `None`         | `(n_epochs, n_channels)` |
| set              | `None`         | `(n_epochs, 1)` |
| `None`           | set            | `(1, n_channels)` |
| set              | set            | `(1, 1)` |

Collapsed dimensions are preserved as size-1. `col_names` (channel names) are stored only when `channel_method=None`.

Aggregation is applied in **channel-then-trial** order.

### Feature descriptions

| Feature         | Unit   | Description |
|-----------------|--------|-------------|
| `"Duration"`    | s      | Mean spindle duration per epoch/channel |
| `"Amplitude"`   | µV     | Mean spindle amplitude per epoch/channel |
| `"Frequency"`   | Hz     | Mean spindle frequency per epoch/channel |
| `"Density"`     | count  | Number of detected spindles per epoch/channel |

---

## Aggregation Methods

| Value           | Description |
|-----------------|-------------|
| `"mean"`        | Arithmetic mean |
| `"median"`      | Median |
| `"std"`         | Standard deviation |
| `"min"`         | Minimum |
| `"max"`         | Maximum |
| `"sum"`         | Sum |
| `"trim_mean80"` | Trimmed mean, central 80% |
| `"trim_mean90"` | Trimmed mean, central 90% |

---

## Computation Details

1. **EEG channel selection:** Non-EEG channels are dropped.
2. **Re-referencing:** The mean signal of `reference_channels` is subtracted from all channels. If any reference channel is missing from the data or contains all-zero values, a warning is logged and the original unreferenced data is used instead. Re-referencing failures are caught and handled gracefully.
3. **Per-epoch detection:** Each epoch is processed individually:
   - Data is converted from volts to microvolts (YASA expects µV).
   - YASA's `spindles_detect` is called with `thresh={"rms": thresh_rms, "corr": thresh_corr}`, `multi_only=False`, `remove_outliers=False`.
4. **Feature extraction:** For each epoch × channel group of detected spindles:
   - `Duration`, `Amplitude`, `Frequency`: mean across detected spindles in that group.
   - `Density`: count of detected spindles in that group.
   - Epoch × channel cells with no detected spindles remain `NaN`.
5. **Stacking:** all four feature arrays are stored in the cache; only the requested `feature` is returned.
6. **Aggregation:** applied to the 2D tensor in trial-then-channel order.

### Caching

`SpindlesDetectionBase` is a **singleton** that caches detection results keyed by `(id(epochs), freq_sp, freq_broad, duration, min_distance, thresh_rms, thresh_corr, reference_channels)`. The cache is valid as long as the epochs object remains in memory. All four features are computed and cached together in a single YASA run; subsequent markers requesting a different feature on the same epochs object simply look up the cache.

---

## Notes

- `feature` is case-sensitive: use `"Duration"`, `"Amplitude"`, `"Frequency"`, `"Density"` exactly.
- There is no `tmin`/`tmax` or `rois` parameter. Detection operates on the full epoch with all EEG channels.
- When no spindles are detected in a given epoch/channel, the output value is `NaN`, not zero.

---

## YAML Examples

### Spindle density, no aggregation

```yaml
markers:
  - kind: SpindlesDetection
    feature: Density
    name: spindles_density
```

### Mean amplitude with trial aggregation

```yaml
markers:
  - kind: SpindlesDetection
    feature: Amplitude
    trial_method: mean
    name: spindles_amplitude_mean
```

### Custom detection parameters, both aggregations

```yaml
markers:
  - kind: SpindlesDetection
    feature: Duration
    freq_sp: [11, 16]
    duration: [0.5, 3.0]
    min_distance: 300
    thresh_rms: 1.5
    thresh_corr: 0.65
    channel_method: mean
    trial_method: mean
    name: spindles_duration_collapsed
```

### Multiple features from one detection (shared cache)

```yaml
markers:
  - kind: SpindlesDetection
    feature: Duration
    name: spindles_duration

  - kind: SpindlesDetection
    feature: Density
    name: spindles_density
```

When both markers share the same detection parameters, YASA runs only once.

---

## See Also

- [SlowWavesDetection](slow_waves_detection.md) — sleep slow wave detection using YASA.
