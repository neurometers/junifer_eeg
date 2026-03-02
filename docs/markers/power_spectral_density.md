# Power Spectral Density Markers

This module contains two PSD markers sharing a common base (`PowerSpectralDensityBase`):

- **`PowerSpectralDensityEstimator`** — returns the raw PSD, frequency array, and normalised PSD for every channel.
- **`PowerSpectralDensitySummary`** — computes the **Spectral Edge Frequency (SEF)** at a given percentile for each epoch and channel.

Both use **Welch's method** via MNE's `compute_psd`, and share the same time-cropping and Welch parameter logic.

---

## PowerSpectralDensityEstimator

`junifer_eeg.markers.PowerSpectralDensityEstimator`

Computes PSD using Welch's method and returns three outputs: raw PSD, the frequency axis, and normalised PSD. Works on both **Raw** and **Epochs** input. When the input is Epochs, PSD is averaged across epochs before output.

### Parameters

| Parameter    | Type              | Default    | Description |
|--------------|-------------------|------------|-------------|
| `tmin`       | `float` or `None` | `None`     | Start time (seconds). Applied as a **crop** before PSD computation. If `None`, uses the start of the data. |
| `tmax`       | `float` or `None` | `None`     | End time (seconds). Applied as a **crop** before PSD computation. If `None`, uses the end of the data. |
| `fmin`       | `float`           | `0.0`      | Minimum frequency (Hz) for PSD computation. |
| `fmax`       | `float` or `None` | `None`     | Maximum frequency (Hz) for PSD computation. If `None`, uses the Nyquist frequency (`sfreq / 2`). |
| `psd_method` | `str`             | `"welch"`  | PSD computation method. Only `"welch"` is supported; other values raise `ValueError`. |
| `n_per_seg`  | `int` or `None`   | `None`     | Length of each segment for Welch's method. If `None`, MNE's default is used. |
| `n_overlap`  | `int` or `None`   | `None`     | Number of overlap points between segments. If `None`, MNE's default is used. |
| `n_fft`      | `int` or `None`   | `None`     | Length of the FFT. If `None`, MNE's default is used. |
| `on`         | `str` or `None`   | `"EEG"`    | Data type to apply this marker to. |
| `name`       | `str` or `None`   | `None`     | Name for the marker. If `None`, the class name is used. |

### Input

- **Accepted types:** `mne.io.BaseRaw` or `mne.BaseEpochs`

### Output

This marker produces **three separate sub-features** in a single call:

| Feature name     | Shape                    | Description |
|------------------|--------------------------|-------------|
| `"psd_data"`     | `(n_channels, n_freqs)`  | Raw PSD in µV²/Hz. `col_names` = frequency labels; `row_names` = channel names. |
| `"psd_freqs"`    | `(1, n_freqs)`           | Frequency axis in Hz. `col_names` = frequency labels. |
| `"psd_data_norm"`| `(n_channels, n_freqs)`  | PSD normalised by the sum of power across all frequencies. `col_names` = frequency labels; `row_names` = channel names. |

Frequency column names are formatted as `"freq_X.XXHz"` (e.g. `"freq_10.00Hz"`).

When the input is Epochs, PSD is computed per-epoch by MNE and then **averaged across epochs** before output.

### Computation Details

1. **EEG channel selection:** Non-EEG channels are dropped.
2. **Time cropping:** If `tmin` or `tmax` is set, the data is cropped before computing PSD.
3. **Welch PSD:** MNE's `compute_psd(method="welch", fmin=fmin, fmax=fmax, ...)` is called. Optional `n_per_seg`, `n_overlap`, and `n_fft` are passed only when explicitly set.
4. **Epoch averaging** (Epochs only): output is averaged across the epoch axis → `(n_channels, n_freqs)`.
5. **Normalisation:** `psd_data_norm = psd_data / sum(psd_data, axis=-1, keepdims=True)`.

---

## PowerSpectralDensitySummary

`junifer_eeg.markers.PowerSpectralDensitySummary`

Computes the **Spectral Edge Frequency (SEF)** for each epoch and channel: the frequency below which a given percentile of the cumulative power is contained. At `percentile=0.5` this is the **Median Spectral Frequency (MSF)**.

Works on **Epochs only**.

### Parameters

| Parameter        | Type                           | Default    | Description |
|------------------|--------------------------------|------------|-------------|
| `percentile`     | `float`                        | `0.5`      | Fraction of total power that defines the spectral edge (0–1). `0.5` = 50th percentile (MSF). |
| `rois`           | `list of str or int` or `None` | `None`     | Channels to include, applied **after** PSD computation. Each item can be a channel index (`int`), a channel name (`str`), or a semantic ROI name (`str`, e.g. `"scalp"`). If `None`, uses all channels. |
| `channel_method` | `str` or `None`                | `None`     | Aggregation method applied **across channels**. If `None`, no channel aggregation. See [Aggregation Methods](#aggregation-methods). |
| `trial_method`   | `str` or `None`                | `None`     | Aggregation method applied **across epochs/trials**. If `None`, no trial aggregation. See [Aggregation Methods](#aggregation-methods). |
| `equipment`      | `str`                          | `"egi256"` | Equipment configuration used for resolving semantic ROI names. |
| `tmin`           | `float` or `None`              | `None`     | Start time (seconds). Applied as a **crop** before PSD computation. |
| `tmax`           | `float` or `None`              | `None`     | End time (seconds). Applied as a **crop** before PSD computation. |
| `fmin`           | `float`                        | `0.0`      | Minimum frequency (Hz) for PSD computation. |
| `fmax`           | `float` or `None`              | `None`     | Maximum frequency (Hz). If `None`, uses Nyquist. |
| `psd_method`     | `str`                          | `"welch"`  | PSD method. Only `"welch"` is supported. |
| `n_per_seg`      | `int` or `None`                | `None`     | Segment length for Welch's method. |
| `n_overlap`      | `int` or `None`                | `None`     | Overlap points between segments. |
| `n_fft`          | `int` or `None`                | `None`     | FFT length. |
| `on`             | `str` or `None`                | `"EEG"`    | Data type to apply this marker to. |
| `name`           | `str` or `None`                | `None`     | Name for the marker. |

### Input

- **Required type:** `mne.BaseEpochs` (raises `ValueError` for Raw input)
- Epochs must be non-empty

### Output

Feature name: **`"psdsummary"`**

The output is a 2D tensor `(n_epochs, n_channels)` containing the SEF value (in Hz) per epoch and channel.

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
2. **Time cropping:** If `tmin` or `tmax` is set, epochs are cropped before PSD computation.
3. **Welch PSD:** Computed preserving per-epoch structure → `(n_epochs, n_channels, n_freqs)`.
4. **SEF computation** (vectorised over all epochs and channels):
   - Cumulative sum of PSD across frequencies: `cum = cumsum(psd, axis=-1)`.
   - Threshold: `thresh = total_power × percentile`.
   - SEF = first frequency where `cum ≥ thresh`.
   - If total power is zero → SEF = `fmin`.
   - If threshold is never reached (but some power exists) → SEF = `fmax`.
5. **ROI filtering** (if `rois` is set): applied to the `(n_epochs, n_channels)` SEF array after computation.
6. **Aggregation:** applied to the 2D tensor in trial-then-channel order.

---

## Aggregation Methods

Both markers' `channel_method` and `trial_method` (where applicable) accept:

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

## YAML Examples

### PowerSpectralDensityEstimator — full spectrum

```yaml
markers:
  - kind: PowerSpectralDensityEstimator
    fmin: 1.0
    fmax: 45.0
    name: psd_estimator
```

### PowerSpectralDensityEstimator — time-windowed with custom Welch params

```yaml
markers:
  - kind: PowerSpectralDensityEstimator
    tmin: 0.0
    tmax: 2.0
    fmin: 1.0
    fmax: 40.0
    n_per_seg: 256
    n_overlap: 128
    name: psd_estimator_windowed
```

### PowerSpectralDensitySummary — MSF (default percentile=0.5)

```yaml
markers:
  - kind: PowerSpectralDensitySummary
    name: msf_default
```

### PowerSpectralDensitySummary — SEF95 with ROI and trial mean

```yaml
markers:
  - kind: PowerSpectralDensitySummary
    percentile: 0.95
    rois: [scalp]
    trial_method: mean
    equipment: egi256
    name: sef95_scalp_mean
```

---

## See Also

- [SpectralPowerBands](spectral_power_bands.md) — band-averaged spectral power with optional dB conversion and spectral entropy.
- [PermutationEntropy](permutation_entropy.md) — complexity measure based on ordinal patterns.
