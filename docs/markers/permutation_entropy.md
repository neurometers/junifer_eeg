# PermutationEntropy

`junifer_eeg.markers.PermutationEntropy`

Computes **Permutation Entropy (PE)** at one or more temporal scales (tau values). For each epoch and channel, the signal is symbolised into ordinal patterns of length `kernel` sampled with delay `tau`, and Shannon entropy is computed over the pattern distribution, normalised to `[0, 1]` by dividing by `log(kernel!)`.

Supports multiple tau values in a single call — each tau represents a different temporal scale. Uses a **singleton with internal caching**: multiple `PermutationEntropy` instances with different aggregation settings but the same `kernel`, `tau`, `tmin`, `tmax` on the same epochs object reuse cached PE values.

If numba is installed, computation is accelerated via a JIT-compiled kernel; otherwise a vectorised NumPy implementation is used automatically.

---

## Parameters

| Parameter        | Type                           | Default    | Description |
|------------------|--------------------------------|------------|-------------|
| `taus`           | `int` or `list of int`         | `8`        | Time delay(s) in samples for ordinal pattern extraction. A single `int` is automatically wrapped in a list. Each value produces one layer in the output tensor. |
| `kernel`         | `int`                          | `3`        | Length of ordinal patterns. Determines the number of possible symbols: `kernel!`. |
| `rois`           | `list of str or int` or `None` | `None`     | Channels to include, applied **before** computation. Each item can be a channel index (`int`), a channel name (`str`, e.g. `"E1"`), or a semantic ROI name (`str`, e.g. `"scalp"`). If `None`, all EEG channels are used. |
| `channel_method` | `str` or `None`                | `None`     | Aggregation method applied **across channels** (axis 2). If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`   | `str` or `None`                | `None`     | Aggregation method applied **across epochs/trials** (axis 1). If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `tmin`           | `float` or `None`              | `None`     | Start time (seconds) of the analysis window. Applied as a **time mask** on the filtered data after filtering. If `None`, uses the start of the epochs. |
| `tmax`           | `float` or `None`              | `None`     | End time (seconds) of the analysis window. Applied as a **time mask** on the filtered data after filtering. If `None`, uses the end of the epochs. |
| `equipment`      | `str`                          | `"egi256"` | Equipment configuration used for resolving semantic ROI names. Supported values: `"egi256"`, `"egi128"`, `"egi64"`, `"standard"`. |
| `on`             | `str` or `None`                | `"EEG"`    | Data type to apply this marker to. |
| `name`           | `str` or `None`                | `None`     | Name for the marker. If `None`, the class name is used. |

---

## Input

- **Required type:** `mne.BaseEpochs` (data must have been epoched in preprocessing)
- Epochs must be non-empty

---

## Output

Feature name: **`"permutationentropy"`**

The output is **always a 3D tensor** `(n_taus, n_epochs, n_channels)`, even when only a single tau is provided.

Output shape depends on which aggregations are applied:

| `channel_method` | `trial_method` | Output shape |
|------------------|----------------|--------------|
| `None`           | `None`         | `(n_taus, n_epochs, n_channels)` |
| set              | `None`         | `(n_taus, n_epochs, 1)` |
| `None`           | set            | `(n_taus, 1, n_channels)` |
| set              | set            | `(n_taus, 1, 1)` |

The tau dimension (axis 0) is **never aggregated**. Collapsed dimensions are preserved as size-1.

`col_names` (channel names) are stored alongside the data only when `channel_method=None`. When channel aggregation is applied, channel names are no longer meaningful and are not stored.

---

## Aggregation Methods

Both `channel_method` and `trial_method` accept the following values:

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
2. **ROI filtering** (if `rois` is set): channels are filtered to the specified subset before computation.
3. **For each tau value:**
   - All epochs are concatenated channel-wise, then an **adaptive low-pass filter** is applied: cutoff = `sfreq / kernel / tau` Hz, 6th-order Butterworth. Filtered data is split back into epochs.
   - A **time mask** is applied using `tmin` and `tmax`.
   - For each epoch × channel, PE is computed:
     - `L = n_samples − tau × (kernel − 1)` windows of length `kernel` are extracted (sampled with step `tau`).
     - Each window is mapped to an ordinal rank pattern (one of `kernel!` symbols).
     - Symbol frequencies are counted and normalised to probabilities.
     - PE = `−Σ p · log(p) / log(kernel!)`, producing a value in `[0, 1]`.
     - If `L ≤ 0` (too few samples for the kernel/tau combination), the value is `nan`.
4. **Stacking:** PE arrays for all tau values are stacked → always `(n_taus, n_epochs, n_channels)`.
5. **Aggregation:** applied to the 3D tensor, preserving collapsed dimensions as size-1.

---

## Notes

- Values are normalised to `[0, 1]`. A value of 0 means maximum regularity (all windows map to the same pattern). A value of 1 means maximum complexity (all patterns equally probable).
- If the time window is too short for a given `kernel`/`tau` combination, the result for that epoch × channel is `nan` rather than raising an error.
- Numba acceleration is used automatically when numba is installed. If not available, falls back to a NumPy implementation with identical results.

---

## YAML Examples

### Single tau, no aggregation

```yaml
markers:
  - kind: PermutationEntropy
    taus: 8
    kernel: 3
    name: pe_tau8
```

### Multiple tau values

```yaml
markers:
  - kind: PermutationEntropy
    taus: [4, 8, 16]
    kernel: 3
    name: pe_multiscale
```

### With ROI selection and trial aggregation

```yaml
markers:
  - kind: PermutationEntropy
    taus: [4, 8, 16]
    kernel: 3
    rois: [scalp]
    trial_method: mean
    equipment: egi256
    name: pe_scalp_mean
```

### Time-windowed, both aggregations

```yaml
markers:
  - kind: PermutationEntropy
    taus: 8
    kernel: 3
    tmin: 0.0
    tmax: 1.0
    channel_method: mean
    trial_method: mean
    name: pe_windowed_collapsed
```

---

## See Also

- [KolmogorovComplexity](kolmogorov_complexity.md) — compression-based complexity, single scale per marker call, no `taus` dimension.
- [SymbolicMutualInformation](symbolic_mutual_information.md) — ordinal-pattern-based connectivity between channel pairs.
