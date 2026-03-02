# SpectralPowerBands

`junifer_eeg.markers.SpectralPowerBands`

Computes **spectral power in frequency bands** from EEG epochs using Welch's method. For each band, PSD values within the frequency range are summed (or used to compute spectral entropy). Supports optional dB conversion, normalisation to relative power, and both channel and trial aggregation.

Uses a **singleton with internal caching** (`SpectralPowerBase`): multiple markers that share the same Epochs object, FFT parameters, and time window will reuse the cached PSD rather than recomputing it.

---

## Parameters

| Parameter        | Type                           | Default                       | Description |
|------------------|--------------------------------|-------------------------------|-------------|
| `bands`          | `dict` or `None`               | `None`                        | Frequency bands as `{name: (fmin, fmax)}`. If `None`, uses standard EEG bands: `delta` (1–4 Hz), `theta` (4–8 Hz), `alpha` (8–12 Hz), `beta` (12–30 Hz), `gamma` (30–45 Hz). |
| `rois`           | `list of str or int` or `None` | `None`                        | Channels to include, applied **before** PSD computation. Each item can be a channel index (`int`), a channel name (`str`, e.g. `"E1"`), or a semantic ROI name (`str`, e.g. `"scalp"`). If `None`, all EEG channels are used. |
| `channel_method` | `str` or `None`                | `None`                        | Aggregation method applied **across channels** (axis 2 of the 3D output). If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`   | `str` or `None`                | `None`                        | Aggregation method applied **across epochs/trials** (axis 1 of the 3D output). If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `normalize`      | `bool`                         | `False`                       | If `True`, PSD is divided by total power across all frequencies (1.0 to min(45.0, Nyquist−1) Hz) before band extraction, producing **relative power**. Required when `entropy=True`. |
| `dB`             | `bool`                         | `True`                        | If `True`, converts band power to decibels via `10 * log10`. Applied only when `entropy=False`. A minimum threshold is applied before conversion (see `db_threshold`). |
| `entropy`        | `bool`                         | `False`                       | If `True`, computes **spectral entropy** within each band instead of summed power. Requires `normalize=True`. When `entropy=True`, `dB` is not applied even if set to `True`. |
| `n_fft`          | `int` or `None`                | `None`                        | Length of the FFT for Welch's method. If `None`, MNE's default is used. |
| `n_per_seg`      | `int` or `None`                | `None`                        | Length of each segment for Welch's method. If `None`, uses `min(64, n_samples // 2)`. |
| `n_overlap`      | `int` or `None`                | `None`                        | Number of overlap points between segments. If `None`, uses `min(32, n_per_seg // 2)`. |
| `db_threshold`   | `float` or `None`              | `None`                        | Minimum value for dB clamping before `log10`. If `None`, an adaptive threshold is computed as `max(min_nonzero * 0.01, machine_epsilon)`. |
| `tmin`           | `float` or `None`              | `None`                        | Start time (seconds) of the analysis window. Applied as a **crop** on the epochs before PSD computation. If `None`, uses the start of the epochs. |
| `tmax`           | `float` or `None`              | `None`                        | End time (seconds) of the analysis window. Applied as a **crop** on the epochs before PSD computation. If `None`, uses the end of the epochs. |
| `equipment`      | `str`                          | `"egi256"`                    | Equipment configuration used for resolving semantic ROI names. Supported values: `"egi256"`, `"egi128"`, `"egi64"`, `"standard"`. |
| `on`             | `str` or `None`                | `"EEG"`                       | Data type to apply this marker to. |
| `name`           | `str` or `None`                | `None`                        | Name for the marker. If `None`, the class name is used. |

---

## Input

- **Required type:** `mne.BaseEpochs` (data must have been epoched in preprocessing)
- Epochs must be non-empty

---

## Output

Feature name: **`"spectralpower"`**

The output is **always a 3D tensor** `(n_bands, n_epochs, n_channels)`, even when only a single band is provided.

Output shape depends on which aggregations are applied:

| `channel_method` | `trial_method` | Output shape |
|------------------|----------------|--------------|
| `None`           | `None`         | `(n_bands, n_epochs, n_channels)` |
| set              | `None`         | `(n_bands, n_epochs, 1)` |
| `None`           | set            | `(n_bands, 1, n_channels)` |
| set              | set            | `(n_bands, 1, 1)` |

The band dimension (axis 0) is **never aggregated**. Collapsed dimensions are preserved as size-1.

`col_names` (channel names) are stored alongside the data only when `channel_method=None`.

Aggregation is applied in **trial-then-channel** order.

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
2. **ROI filtering** (if `rois` is set): channels are filtered to the specified subset before any computation.
3. **Band defaults:** If `bands=None`, the following bands are used:
   ```
   delta: (1, 4) Hz
   theta: (4, 8) Hz
   alpha: (8, 12) Hz
   beta:  (12, 30) Hz
   gamma: (30, 45) Hz
   ```
4. **PSD frequency range:**
   - If `normalize=True`: PSD is computed over `[1.0, min(45.0, sfreq/2 − 1)]` Hz so that the full normalisation denominator is consistent across all frequency bands.
   - If `normalize=False`: PSD is computed over `[min(band fmins), min(max(band fmaxs), sfreq/2 − 1)]` Hz.
5. **Welch PSD computation:** Uses MNE's `compute_psd(method="welch", ...)`. Adaptive parameters: `n_per_seg = min(64, n_samples // 2)`, `n_overlap = min(32, n_per_seg // 2)` when not explicitly set.
6. **Time cropping:** If `tmin` or `tmax` is set, the epochs are cropped before PSD computation.
7. **Normalisation** (if `normalize=True`): PSD is divided element-wise by total power (sum over all frequency bins), clamped to a minimum of `1e-12` to avoid division by zero.
8. **Band extraction:** For each band, the PSD bins with `fmin ≤ freq < fmax` are selected. If no bins fall within a band, the band power is set to zero.
   - If `entropy=False`: band power = sum of PSD values within the band.
   - If `entropy=True`: spectral entropy = `−Σ p·log(p) / log(n_bins)`, where `n_bins` is the number of frequency bins in the band. Requires `normalize=True`.
9. **dB conversion** (if `dB=True` and `entropy=False`): each band value is clamped to `db_threshold` and converted via `10 * log10`.
10. **Stacking:** band arrays are stacked → always `(n_bands, n_epochs, n_channels)`.
11. **Aggregation:** applied to the 3D tensor in trial-then-channel order, preserving collapsed dimensions as size-1.

### Caching

`SpectralPowerBase` is a **singleton** that caches PSD results keyed by `(id(epochs), n_fft, n_per_seg, n_overlap, tmin, tmax, fmin, fmax)`. The cache is valid as long as the epochs object remains in memory. Multiple `SpectralPowerBands` instances with different band or aggregation settings but sharing the same epochs and Welch parameters will reuse the cached PSD.

---

## Notes

- `entropy=True` requires `normalize=True`; a `ValueError` is raised at initialisation if `normalize=False` when `entropy=True`.
- When `entropy=True`, the `dB` parameter is silently ignored.
- Bands with no PSD bins within the frequency range (e.g. if the band exceeds Nyquist) are returned as zero-power, not as errors.

---

## YAML Examples

### Standard bands, dB, no aggregation

```yaml
markers:
  - kind: SpectralPowerBands
    name: spb_default
```

### Custom bands with relative power

```yaml
markers:
  - kind: SpectralPowerBands
    bands:
      alpha: [8, 12]
      beta: [12, 30]
    normalize: true
    dB: false
    name: spb_relative
```

### Spectral entropy with ROI and trial aggregation

```yaml
markers:
  - kind: SpectralPowerBands
    normalize: true
    entropy: true
    rois: [scalp]
    trial_method: mean
    equipment: egi256
    name: spb_entropy_scalp
```

### Time-windowed, both aggregations

```yaml
markers:
  - kind: SpectralPowerBands
    tmin: 0.0
    tmax: 2.0
    channel_method: mean
    trial_method: mean
    name: spb_windowed_collapsed
```

---

## See Also

- [PowerSpectralDensity](power_spectral_density.md) — raw PSD estimator and spectral edge frequency summary.
- [PermutationEntropy](permutation_entropy.md) — ordinal-pattern entropy, an alternative spectral complexity measure.
