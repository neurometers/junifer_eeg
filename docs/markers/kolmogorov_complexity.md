# KolmogorovComplexity

`junifer_eeg.markers.KolmogorovComplexity`

Computes the **Kolmogorov complexity** of EEG signals using a compression-based approximation. Each signal is first symbolised into discrete amplitude bins, then compressed with zlib, and the complexity is the compression ratio: `len(compressed) / len(original)`. A more compressible (more regular/repetitive) signal yields a **lower** value; a more random signal yields a **higher** value.

Computation uses a **singleton with internal caching**: if multiple `KolmogorovComplexity` instances are run on the same epochs object with the same `nbins`/`tmin`/`tmax`, the raw KC values are computed only once and reused.

---

## Parameters

| Parameter        | Type                            | Default     | Description |
|------------------|---------------------------------|-------------|-------------|
| `nbins`          | `int`                           | `16`        | Number of amplitude bins used for signal symbolisation. The signal amplitude range is defined by its 10th–90th percentile (the outer 10% on each side is excluded). |
| `rois`           | `list of str or int` or `None`  | `None`      | Channels to include in the computation, applied **before** KC is computed. Each item can be a channel index (`int`), a channel name (`str`, e.g. `"E1"`), or a semantic ROI name (`str`, e.g. `"scalp"`). If `None`, all EEG channels are used. |
| `channel_method` | `str` or `None`                 | `None`      | Aggregation method applied **across channels** (axis 1). If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`   | `str` or `None`                 | `None`      | Aggregation method applied **across epochs/trials** (axis 0). If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `tmin`           | `float` or `None`               | `None`      | Start time (seconds) for analysis. If `None`, uses the start of the epochs. Cropping is applied inside the KC computation. |
| `tmax`           | `float` or `None`               | `None`      | End time (seconds) for analysis. If `None`, uses the end of the epochs. Cropping is applied inside the KC computation. |
| `equipment`      | `str`                           | `"egi256"`  | Equipment configuration used for resolving semantic ROI names. Supported values: `"egi256"`, `"egi128"`, `"egi64"`, `"standard"`. |
| `on`             | `str` or `None`                 | `"EEG"`     | Data type to apply this marker to. |
| `name`           | `str` or `None`                 | `None`      | Name for the marker. If `None`, the class name is used. |

---

## Input

- **Required type:** `mne.BaseEpochs` (data must have been epoched in preprocessing)
- Epochs must be non-empty

---

## Output

The result is stored under the feature name **`"kolmogorovcomplexity"`**.

Output shape depends on which aggregations are applied:

| `channel_method` | `trial_method` | Output shape |
|------------------|----------------|--------------|
| `None`           | `None`         | `(n_epochs, n_channels)` |
| set              | `None`         | `(n_epochs, 1)` |
| `None`           | set            | `(1, n_channels)` |
| set              | set            | `(1, 1)` |

Aggregated dimensions are **preserved as size-1** (never squeezed), ensuring the 2D tensor shape is always consistent.

`col_names` (channel names) are stored alongside the data only when `channel_method=None`. When channel aggregation is applied, channel names are no longer meaningful and are not stored.

---

## Aggregation Methods

Both `channel_method` and `trial_method` accept the following values:

| Value          | Description |
|----------------|-------------|
| `"mean"`       | Arithmetic mean |
| `"median"`     | Median |
| `"std"`        | Standard deviation |
| `"min"`        | Minimum |
| `"max"`        | Maximum |
| `"sum"`        | Sum |
| `"trim_mean80"` | Trimmed mean removing the top and bottom 10% (uses central 80%) |
| `"trim_mean90"` | Trimmed mean removing the top and bottom 5% (uses central 90%) |

---

## Computation Details

1. All non-EEG channels are dropped first (EGI `E*` naming is detected first; standard MNE channel types as fallback).
2. If `rois` is provided, channels are filtered to only those specified before computation.
3. For each epoch × channel pair, the KC value is computed as:
   - The signal's amplitude range is defined from its 10th to 90th percentile.
   - The signal is symbolised by mapping each sample to one of `nbins` amplitude bins (encoded as ASCII letters starting from `"A"`).
   - The byte string is compressed with `zlib`.
   - KC = `len(compressed_bytes) / len(original_bytes)`.
4. If `tmin` or `tmax` is set, the epochs are cropped **inside** the KC computation (on a copy).
5. Aggregation is applied to the resulting `(n_epochs, n_channels)` array, preserving collapsed dimensions as size-1.

---

## YAML Example

### No aggregation (full epoch × channel output)

```yaml
markers:
  - kind: KolmogorovComplexity
    nbins: 16
    name: kc_full
```

### With ROI selection and trial aggregation

```yaml
markers:
  - kind: KolmogorovComplexity
    nbins: 16
    rois: [scalp]
    trial_method: mean
    equipment: egi256
    name: kc_scalp_mean
```

### Time-windowed, both aggregations

```yaml
markers:
  - kind: KolmogorovComplexity
    nbins: 16
    tmin: 0.0
    tmax: 1.0
    channel_method: mean
    trial_method: mean
    name: kc_windowed
```

---

## See Also

- [PermutationEntropy](permutation_entropy.md) — another complexity marker using ordinal patterns.
- [SpectralPowerBands](spectral_power_bands.md) — frequency-domain marker.
