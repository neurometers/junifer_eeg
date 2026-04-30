# SlowWavesDetection

`junifer_eeg.markers.SlowWavesDetection`

Detects **sleep slow waves** in EEG epochs and returns one requested feature per epoch and channel. Two backends are available:

- `detection_method: yasa` uses [YASA](https://raphaelvallat.com/yasa/build/html/index.html)
- `detection_method: custom` uses a zero-crossing detector inspired by Andrillon et al. 2021

Detection runs once per unique parameter set and epochs object. Uses a **singleton with internal caching** (`SlowWavesDetectionBase`): multiple `SlowWavesDetection` markers requesting different features on the same epochs with the same detection parameters will reuse the cached detection results.

---

## Parameters

| Parameter             | Type                     | Default          | Description |
|-----------------------|--------------------------|------------------|-------------|
| `feature`             | `str`                    | **required**     | Feature to extract. Must be exactly one of: `"Duration"`, `"PTP"`, `"Frequency"`, `"Slope"`, `"Density"`. Raises `ValueError` at initialisation if missing or invalid. |
| `detection_method`    | `{"yasa", "custom"}`    | `"yasa"`         | Detection backend. |
| `freq_sw`             | `tuple of float`         | `(0.3, 1.5)`     | Slow wave frequency range in Hz `(fmin, fmax)`. Passed to YASA for `detection_method="yasa"`; used as the custom band-pass for `detection_method="custom"`. |
| `amp_ptp_initial`     | `float`                  | `15.0`           | Minimum peak-to-peak amplitude threshold in µV. YASA receives `amp_ptp=(amp_ptp_initial, inf)`. |
| `freq_threshold`      | `float`                  | `7.0`            | Maximum frequency (Hz) used in post-detection filtering: detected waves with `Frequency > freq_threshold` are removed. |
| `artifact_threshold`  | `float`                  | `75.0`           | Accepted as a parameter and included in the detection cache key. |
| `ptp_threshold_mode`  | `{"adaptive", "fixed"}` | `"adaptive"`     | Whether to apply per-channel PTP thresholding after detection. |
| `ptp_percentile`      | `float`                  | `90.0`           | Percentile used for adaptive per-channel PTP thresholds when no CSV is provided. |
| `max_ptp_amplitude`   | `float`                  | `150.0`          | Maximum PTP amplitude (µV) allowed during post-detection filtering. |
| `ptp_thresholds_path` | `str` or `None`          | `None`           | Optional CSV containing fixed `(subject, channel, ptp_threshold)` values. |
| `reference_channels`  | `tuple of str`           | `("TP7", "TP8")` | Channel names used for re-referencing before detection. The mean of these channels is subtracted from all channels. Falls back to unreferenced data if channels are missing or contain all zeros. |
| `channel_method`      | `str` or `None`          | `None`           | Aggregation method applied **across channels**. If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`        | `str` or `None`          | `None`           | Aggregation method applied **across epochs/trials**. If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `equipment`           | `str`                    | `"egi256"`       | Equipment configuration. |
| `on`                  | `str` or `None`          | `"EEG"`          | Data type to apply this marker to. |
| `name`                | `str` or `None`          | `None`           | Name for the marker. If `None`, the class name is used. |

---

## Input

- **Required type:** `mne.BaseEpochs` (data must have been epoched in preprocessing)
- Epochs must be non-empty

---

## Output

Feature name: **`"slowwavesdetection"`**

The output is always a 2D tensor `(n_epochs, n_channels)`. Cells with no detected slow waves for a given epoch/channel are set to `NaN`.

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

| Feature       | Unit   | Description |
|---------------|--------|-------------|
| `"Duration"`  | s      | Mean slow wave duration per epoch/channel |
| `"PTP"`       | µV     | Mean peak-to-peak amplitude per epoch/channel |
| `"Frequency"` | Hz     | Mean slow wave frequency per epoch/channel |
| `"Slope"`     | µV/s   | Mean slow wave slope per epoch/channel |
| `"Density"`   | count  | Number of detected slow waves per epoch/channel |

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
   - Data is converted from volts to microvolts.
   - If `detection_method="yasa"`, YASA's `sw_detect` is called with `amp_ptp=(amp_ptp_initial, inf)`, `coupling=False`, `remove_outliers=False`.
   - If `detection_method="custom"`, a Chebyshev type-II band-pass is applied and waves are built from successive zero-crossings around a negative trough followed by a positive rebound.
4. **Post-detection filtering** (Andrillon & Pinggal criteria):
   - Waves with `Frequency > freq_threshold` are removed.
   - Waves with `PTP >= max_ptp_amplitude` are removed.
   - If a positive peak column is available, waves with positive peak `>= artifact_threshold` are removed.
   - If a positive-half-period proxy is available, legacy custom-method bounds are applied.
   - For each channel, the 90th percentile of PTP across all remaining detected waves is computed. Only waves with `PTP ≥` that channel-specific threshold are retained.
   - If `ptp_thresholds_path` is provided, the CSV thresholds are used instead of computing the percentile from the current element.
5. **Feature extraction:** For each epoch × channel group of surviving waves:
   - `Duration`, `PTP`, `Frequency`, `Slope`: mean across detected waves in that group.
   - `Density`: count of detected waves in that group.
   - Epoch × channel cells with no surviving waves remain `NaN`.
6. **Stacking:** all five feature arrays are stored in the cache; only the requested `feature` is returned.
7. **Aggregation:** applied to the 2D tensor in trial-then-channel order.

### Caching

`SlowWavesDetectionBase` is a **singleton** that caches detection results keyed by `(id(epochs), detection_method, freq_sw, amp_ptp_initial, freq_threshold, artifact_threshold, reference_channels)`. The cache is valid as long as the epochs object remains in memory. All five features are computed and cached together in a single detection run; subsequent markers requesting a different feature on the same epochs object simply look up the cache.

---

## Notes

- `feature` is case-sensitive: use `"Duration"`, `"PTP"`, `"Frequency"`, `"Slope"`, `"Density"` exactly.
- There is no `tmin`/`tmax` or `rois` parameter. Detection operates on the full epoch with all EEG channels.
- When no slow waves are detected in a given epoch/channel, the output value is `NaN`, not zero.

---

## YAML Examples

### Slow wave density, no aggregation

```yaml
markers:
  - kind: SlowWavesDetection
    feature: Density
    name: sw_density
```

### Mean PTP amplitude with trial aggregation

```yaml
markers:
  - kind: SlowWavesDetection
    feature: PTP
    trial_method: mean
    name: sw_ptp_mean
```

### Custom detection parameters, both aggregations

```yaml
markers:
  - kind: SlowWavesDetection
    feature: Duration
    detection_method: custom
    freq_sw: [1.0, 10.0]
    amp_ptp_initial: 20.0
    freq_threshold: 7.0
    channel_method: mean
    trial_method: mean
    name: sw_duration_collapsed
```

### Use precomputed thresholds from a reference condition

```yaml
markers:
  - kind: SlowWavesDetection
    feature: Density
    detection_method: custom
    freq_sw: [1.0, 10.0]
    ptp_thresholds_path: /path/to/sw_thresholds.csv
    name: sw_density_fixed_threshold
```

### Multiple features from one detection (shared cache)

```yaml
markers:
  - kind: SlowWavesDetection
    feature: Duration
    name: sw_duration

  - kind: SlowWavesDetection
    feature: Density
    name: sw_density
```

When both markers share the same detection parameters, the backend runs only once.

---

## See Also

- [SpindlesDetection](spindles_detection.md) — sleep spindle detection using YASA.
