# Symbolic Mutual Information

`junifer_eeg.markers` provides two classes for this marker:

| Class | Purpose |
|-------|---------|
| [`SymbolicMutualInformation`](#symbolicmutualinformation) | Full 4D output — no aggregation |
| [`SymbolicMutualInformationROIs`](#symbolicmutualinformationrois) | Same computation + aggregation across connectivity, channel, and/or trial dimensions |

Both compute **Symbolic Mutual Information (SMI)** or **Weighted Symbolic Mutual Information (wSMI)** between all pairs of EEG channels, following the NICE pipeline algorithm. `SymbolicMutualInformationROIs` internally delegates the full computation to `SymbolicMutualInformation` and applies aggregation on top.

---

## Algorithm Overview

1. Optional Current Source Density (CSD) spatial filter.
2. Adaptive low-pass filtering per tau value: cutoff = `sfreq / kernel / tau` Hz.
3. Ordinal pattern (symbolic) transformation.
4. Pairwise mutual information across all channel pairs, with optional weighting.
5. Normalisation by `log(kernel!)`.

Computation uses a **singleton with internal caching**: for the same filtered data array, `kernel`, `tau`, and `weighted` combination, the connectivity matrix is computed only once and reused within a session.

---

## SymbolicMutualInformation

`junifer_eeg.markers.SymbolicMutualInformation`

Computes SMI/wSMI and returns the full `(n_taus, n_epochs, n_channels, n_channels)` tensor. No aggregation is applied.

### Parameters

| Parameter   | Type                           | Default    | Description |
|-------------|--------------------------------|------------|-------------|
| `kernel`    | `int`                          | `3`        | Length of ordinal patterns. Determines the number of possible symbols: `kernel!`. Must be > 1. |
| `taus`      | `int` or `list of int`         | `8`        | Time delay(s) in samples for ordinal pattern extraction. A single `int` is automatically wrapped in a list. Must be > 0. One connectivity matrix is computed per tau value. |
| `weighted`  | `bool`                         | `True`     | If `True`, computes weighted SMI (wSMI) by applying a weight matrix that zeroes out same-pattern and exact-reversal-pattern pairs. If `False`, computes standard unweighted SMI. |
| `csd`       | `bool`                         | `True`     | If `True`, applies Current Source Density preprocessing (`lambda2=1e-5`) before computation. Any remaining bad channels are interpolated before CSD is applied. |
| `rois`      | `list of str or int` or `None` | `None`     | Channels to include, applied **before** connectivity computation. Each item can be a channel index (`int`), a channel name (`str`, e.g. `"E1"`), or a semantic ROI name (`str`, e.g. `"scalp"`). If `None`, all EEG channels are used. **Note:** when ROI filtering is applied, output channel names become generic labels (`"ch_0"`, `"ch_1"`, etc.). |
| `tmin`      | `float` or `None`              | `None`     | Start time (seconds) of the analysis window. Applied as a **time mask** on the filtered data — not a crop of the original epochs. |
| `tmax`      | `float` or `None`              | `None`     | End time (seconds) of the analysis window. Applied as a **time mask** on the filtered data — not a crop of the original epochs. |
| `equipment` | `str`                          | `"egi256"` | Equipment configuration used for resolving semantic ROI names. Supported values: `"egi256"`, `"egi128"`, `"egi64"`, `"standard"`. |
| `on`        | `str` or `None`                | `"EEG"`    | Data type to apply this marker to. |
| `name`      | `str` or `None`                | `None`     | Name for the marker. If `None`, the class name is used. |

### Output

Feature name: **`"symbolicmutualinformation"`**

**Shape:** always `(n_taus, n_epochs, n_channels, n_channels)`

The connectivity matrix is **symmetric** — the upper triangular is computed then mirrored. The **diagonal is always zero** (no self-MI). `col_names` (channel names from picked channels, or generic `"ch_0"`... after ROI filtering) are stored alongside the data.

---

## SymbolicMutualInformationROIs

`junifer_eeg.markers.SymbolicMutualInformationROIs`

Identical computation to `SymbolicMutualInformation`, with three additional aggregation parameters. The 4D structure is always preserved — collapsed dimensions become size-1.

> **Note:** The tau dimension (axis 0) is never aggregated, regardless of which methods are set.

### Parameters

All parameters from `SymbolicMutualInformation` apply, plus:

| Parameter              | Type            | Default | Description |
|------------------------|-----------------|---------|-------------|
| `connectivity_method`  | `str` or `None` | `None`  | Aggregation across the **channel_y** dimension (axis 3). |
| `channel_method`       | `str` or `None` | `None`  | Aggregation across the **channel_x** dimension (axis 2). |
| `trial_method`         | `str` or `None` | `None`  | Aggregation across the **epoch/trial** dimension (axis 1). |

### Output

Feature name: **`"symbolicmutualinformation"`**

Aggregation order is fixed: `connectivity_method` (axis 3) → `channel_method` (axis 2) → `trial_method` (axis 1).

| `connectivity_method` | `channel_method` | `trial_method` | Output shape |
|-----------------------|------------------|----------------|--------------|
| `None`                | `None`           | `None`         | `(n_taus, n_epochs, n_channels, n_channels)` |
| set                   | `None`           | `None`         | `(n_taus, n_epochs, n_channels, 1)` |
| `None`                | set              | `None`         | `(n_taus, n_epochs, 1, n_channels)` |
| `None`                | `None`           | set            | `(n_taus, 1, n_channels, n_channels)` |
| set                   | set              | set            | `(n_taus, 1, 1, 1)` |

`col_names` are **never stored** for this marker (`channel_aggregated=True` is always set in the storage formatter).

---

## Input (both classes)

- **Required type:** `mne.BaseEpochs`
- Epochs must be non-empty
- At least 2 channels required after picking

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
| `"trim_mean80"` | Trimmed mean removing the top and bottom 10% (uses central 80%) |
| `"trim_mean90"` | Trimmed mean removing the top and bottom 5% (uses central 90%) |

---

## Computation Details

1. **EEG channel selection:** Non-EEG channels are dropped (EGI `E*` naming first; MNE channel types as fallback).
2. **Parameter validation:** `kernel > 1` and all `tau > 0`; `ValueError` raised otherwise.
3. **CSD preprocessing** (if `csd=True`):
   - Any bad channels are interpolated on a copy.
   - `mne.preprocessing.compute_current_source_density` is applied with `lambda2=1e-5`.
   - If CSD channels are produced, they replace the EEG channels for all subsequent steps.
4. **Channel picking:** Types `eeg`, `csd`, `meg`, `seeg`, `ecog` are picked (excluding bads). At least 2 required.
5. **ROI filtering** (if `rois` is set): channels are filtered before connectivity computation. Channel names are relabelled as `"ch_0"`, `"ch_1"`, etc.
6. **For each tau value:**
   - Adaptive low-pass filter: cutoff = `sfreq / kernel / tau` Hz, 6th-order Butterworth. Applied by concatenating all epochs, filtering, then splitting back.
   - Time mask applied using `tmin` and `tmax`.
   - Minimum samples required after masking: `tau × (kernel − 1) + 1`. A `ValueError` is raised if not met.
   - Ordinal patterns of length `kernel` with delay `tau` are extracted for each channel pair × epoch. The joint symbol distribution is computed and mutual information calculated.
   - If `weighted=True`, pairs where both symbols are identical or exact reversals of each other receive weight 0.
   - Result normalised by `log(kernel!)`.
   - Upper triangular connectivity matrix `(n_channels, n_channels, n_epochs)` computed and symmetrized.
7. **Stacking:** All tau connectivity matrices stacked → `(n_taus, n_epochs, n_channels, n_channels)`.

### Minimum Samples Requirement

```
min_samples = tau × (kernel − 1) + 1
```

| kernel | tau | min_samples |
|--------|-----|-------------|
| 3      | 8   | 17          |
| 3      | 16  | 33          |
| 5      | 8   | 33          |
| 5      | 4   | 17          |

---

## YAML Examples

### `SymbolicMutualInformation` — default wSMI, single tau

```yaml
markers:
  - kind: SymbolicMutualInformation
    kernel: 3
    taus: 8
    weighted: true
    csd: true
    name: wsmi
```

### `SymbolicMutualInformation` — multiple tau values

```yaml
markers:
  - kind: SymbolicMutualInformation
    kernel: 3
    taus: [4, 8, 16]
    weighted: true
    csd: true
    name: wsmi_multiscale
```

### `SymbolicMutualInformationROIs` — aggregate connectivity and trials

```yaml
markers:
  - kind: SymbolicMutualInformationROIs
    kernel: 3
    taus: 8
    weighted: true
    csd: true
    connectivity_method: median
    trial_method: mean
    name: wsmi_rois
```

### `SymbolicMutualInformationROIs` — multiple taus, all dimensions aggregated

```yaml
markers:
  - kind: SymbolicMutualInformationROIs
    kernel: 3
    taus: [4, 8, 16]
    weighted: true
    csd: true
    connectivity_method: median
    channel_method: mean
    trial_method: mean
    name: wsmi_multiscale_collapsed
```

---

## See Also

- [PermutationEntropy](permutation_entropy.md) — ordinal-pattern entropy per channel (no connectivity).
- [KolmogorovComplexity](kolmogorov_complexity.md) — compression-based complexity per channel.
