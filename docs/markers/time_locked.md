# Time-Locked Markers

This module contains two markers that operate on event-related data in specified time windows, sharing a common base (`TimeLockedBase`):

- **`TimeLockedTopography`** — computes the time-averaged ERP topography for all epochs in a given time window.
- **`TimeLockedContrast`** — computes the difference between averaged ERPs from two experimental conditions in a given time window.

---

## TimeLockedTopography

`junifer_eeg.markers.TimeLockedTopography`

Extracts **time-locked topographies (ERPs)** by averaging the EEG signal across a specified time window for every epoch. Supports optional baseline correction, re-referencing, ROI filtering, and epoch/channel aggregation.

### Parameters

| Parameter        | Type                               | Default      | Description |
|------------------|------------------------------------|--------------|-------------|
| `tmin`           | `float`                            | **required** | Start time (seconds) of the analysis window. Clamped to epoch boundaries if out of range. |
| `tmax`           | `float`                            | **required** | End time (seconds) of the analysis window. Clamped to epoch boundaries if out of range. |
| `baseline`       | `tuple of float` or `None`         | `None`       | Baseline period `(tmin, tmax)` for baseline correction (e.g. `(-0.2, 0)`). Applied **after** re-referencing and **before** cropping. |
| `reference`      | `str`, `list of str`, or `None`    | `None`       | EEG reference to apply before analysis. Options: `"average"` (average reference), a single channel name string (e.g. `"Cz"`), or a list of channel names (e.g. `["TP9", "TP10"]`). Invalid channel names raise `ValueError`. If `None`, no re-referencing is applied. |
| `rois`           | `list of str or int` or `None`     | `None`       | Channels to include. Applied **after** re-referencing and baseline correction, **before** time averaging. Each item can be a channel index (`int`), a channel name (`str`), or a semantic ROI name (`str`, e.g. `"scalp"`). If `None`, uses all channels. |
| `channel_method` | `str` or `None`                    | `None`       | Aggregation method applied **across channels**. If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`   | `str` or `None`                    | `None`       | Aggregation method applied **across epochs/trials**. If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `equipment`      | `str`                              | `"standard"` | Equipment configuration for resolving semantic ROI names. Supported values: `"egi256"`, `"egi128"`, `"egi64"`, `"standard"`. |
| `on`             | `str` or `None`                    | `"EEG"`      | Data type to apply this marker to. |
| `name`           | `str` or `None`                    | `None`       | Name for the marker. If `None`, the class name is used. |

### Input

- **Required type:** `mne.BaseEpochs` (raises `ValueError` for non-Epochs input)
- Epochs must be non-empty

### Output

Feature name: **`"timelockedtopo"`**

The output is always a 2D tensor `(n_epochs, n_channels)` representing the mean amplitude per epoch and channel in the specified time window.

Output shape depends on which aggregations are applied:

| `channel_method` | `trial_method` | Output shape |
|------------------|----------------|--------------|
| `None`           | `None`         | `(n_epochs, n_channels)` |
| set              | `None`         | `(n_epochs, 1)` |
| `None`           | set            | `(1, n_channels)` |
| set              | set            | `(1, 1)` |

Collapsed dimensions are preserved as size-1. `col_names` (channel names) are stored only when `channel_method=None`.

Aggregation is applied in **channel-then-trial** order.

### Computation Details

1. **EEG channel selection:** Non-EEG channels are dropped.
2. **Re-referencing** (if `reference` is set): applied to the full epochs before any windowing.
3. **Baseline correction** (if `baseline` is set): applied after re-referencing.
4. **Time cropping:** Epochs are cropped to `[tmin, tmax]`. Boundaries are clamped to the epoch's actual time range to avoid errors.
5. **ROI filtering** (if `rois` is set): channels are filtered to the specified subset.
6. **Time averaging:** Mean across the time axis → `(n_epochs, n_channels)`.
7. **Aggregation:** applied to the 2D tensor in trial-then-channel order.

---

## TimeLockedContrast

`junifer_eeg.markers.TimeLockedContrast`

Computes the **condition contrast** (A − B) between two groups of epochs, each processed through the same ERP pipeline. Conditions can have **different epoch counts** because each condition is averaged across its own epochs independently before subtraction.

### Parameters

| Parameter        | Type                               | Default      | Description |
|------------------|------------------------------------|--------------|-------------|
| `condition_a`    | `str`, `int`, or `list`            | **required** | First condition label(s). Can be an event name, event code (`int`), or a list of these. Matched against `epochs.event_id` with fallback string/integer conversion. |
| `condition_b`    | `str`, `int`, or `list`            | **required** | Second condition label(s). Same format as `condition_a`. |
| `tmin`           | `float`                            | **required** | Start time (seconds) of the analysis window. |
| `tmax`           | `float`                            | **required** | End time (seconds) of the analysis window. |
| `baseline`       | `tuple of float` or `None`         | `None`       | Baseline period `(tmin, tmax)` for baseline correction, applied before condition splitting. |
| `reference`      | `str`, `list of str`, or `None`    | `None`       | EEG reference (`"average"`, a channel name, or list of channel names). Applied before condition splitting. |
| `comment`        | `str` or `None`                    | `None`       | Free-text annotation. Stored as an attribute but **not used in computation**. |
| `rois`           | `list of str or int` or `None`     | `None`       | Channels to include. Applied independently to each condition after cropping. |
| `channel_method` | `str` or `None`                    | `None`       | Aggregation method across channels. Only `"mean"` and `"median"` are supported; other values raise `ValueError`. If `None`, no channel aggregation. |
| `trial_method`   | `str` or `None`                    | `None`       | Parameter accepted but **not used in computation**. Each condition is averaged across its own epochs internally before subtraction. |
| `equipment`      | `str`                              | `"egi256"`   | Equipment configuration for resolving semantic ROI names. |
| `on`             | `str` or `None`                    | `"EEG"`      | Data type to apply this marker to. |
| `name`           | `str` or `None`                    | `None`       | Name for the marker. |

### Input

- **Required type:** `mne.BaseEpochs` (raises `ValueError` for non-Epochs input)
- Epochs must be non-empty and must contain events matching both `condition_a` and `condition_b`

### Output

Feature name: **`"timelockedcontrast"`**

The output is a 1D array `(n_channels,)` representing the per-channel contrast (mean of condition A minus mean of condition B), after averaging within each condition across time and epochs.

| `channel_method` | Output shape |
|------------------|--------------|
| `None`           | `(n_channels,)` |
| set              | `(1,)` |

`col_names` (channel names) are stored when `channel_method=None`. When channel aggregation is applied, `col_names` is not stored.

> **Note:** `trial_method` is accepted as a parameter for API consistency but is silently ignored. Each condition is always averaged across its own epochs before subtraction, which is what makes different-sized conditions possible.

### Computation Details

1. **EEG channel selection:** Non-EEG channels are dropped.
2. **Re-referencing** (if `reference` is set): applied before condition splitting.
3. **Baseline correction** (if `baseline` is set): applied before condition splitting.
4. **Condition splitting:** epochs are split into `epochs_a` and `epochs_b` using MNE's native event indexing. Integer and string condition labels are tried with fallback conversion.
5. **Per-condition pipeline** (applied independently to each condition):
   - Crop to `[tmin, tmax]` (clamped to epoch boundaries).
   - ROI filtering (if `rois` is set).
   - Average across time axis → `(n_epochs_cond, n_channels)`.
   - Average across epochs → `(n_channels,)`.
6. **Contrast:** `result = mean_A − mean_B` → `(n_channels,)`.
7. **Channel aggregation** (if `channel_method` is set): only `"mean"` and `"median"` are supported.

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

> **Note for `TimeLockedContrast`:** only `"mean"` and `"median"` are supported for `channel_method`. Other values raise `ValueError`.

---

## YAML Examples

### TimeLockedTopography — ERP in a window, average reference

```yaml
markers:
  - kind: TimeLockedTopography
    tmin: 0.1
    tmax: 0.5
    reference: average
    name: erp_topo
```

### TimeLockedTopography — with baseline correction, ROI, and trial mean

```yaml
markers:
  - kind: TimeLockedTopography
    tmin: 0.0
    tmax: 0.6
    baseline: [-0.2, 0.0]
    rois: [scalp]
    trial_method: mean
    equipment: egi256
    name: erp_scalp_mean
```

### TimeLockedContrast — two event codes

```yaml
markers:
  - kind: TimeLockedContrast
    condition_a: "target"
    condition_b: "standard"
    tmin: 0.2
    tmax: 0.5
    baseline: [-0.2, 0.0]
    name: p300_contrast
```

### TimeLockedContrast — list conditions, channel mean

```yaml
markers:
  - kind: TimeLockedContrast
    condition_a: ["go_left", "go_right"]
    condition_b: "nogo"
    tmin: 0.1
    tmax: 0.4
    rois: [scalp]
    channel_method: mean
    equipment: egi256
    name: erp_go_nogo_mean
```

---

## See Also

- [PowerSpectralDensity](power_spectral_density.md) — frequency-domain analysis of EEG epochs.
- [SpectralPowerBands](spectral_power_bands.md) — band-averaged spectral power.
