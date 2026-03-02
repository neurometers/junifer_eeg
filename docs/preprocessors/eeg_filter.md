# EEGFilter

`junifer_eeg.preprocessors.EEGFilter`

Applies high-pass, low-pass, and/or notch filters to EEG data, with optional resampling.

---

## Parameters

| Parameter           | Type                      | Default  | Description |
|---------------------|---------------------------|----------|-------------|
| `high_freq`         | `float` or `None`         | `None`   | High-pass filter cutoff frequency (Hz). Removes signal components below this frequency. If `None`, no high-pass filter is applied. |
| `low_freq`          | `float` or `None`         | `None`   | Low-pass filter cutoff frequency (Hz). Removes signal components above this frequency. If `None`, no low-pass filter is applied. |
| `notches`           | `list of float` or `None` | `None`   | Notch filter frequencies (Hz), e.g. `[50, 100]`. Frequencies above Nyquist are automatically skipped. If `None`, no notch filter is applied. Notch filtering always uses FFT method regardless of `filter_method`. |
| `resample_freq`     | `float` or `None`         | `None`   | Target sampling rate (Hz). If `None`, or if the data is already at this rate, no resampling is done. |
| `hp_order`          | `int`                     | `4`      | Order of the Butterworth high-pass filter. Only used when `filter_method="iir"`. |
| `lp_order`          | `int`                     | `8`      | Order of the Butterworth low-pass filter. Only used when `filter_method="iir"`. |
| `l_trans_bandwidth` | `float`                   | `0.1`    | Transition bandwidth (Hz) for the high-pass filter's lower edge. Only used when `high_freq` is set. |
| `filter_method`     | `"iir"` or `"fir"`        | `"iir"`  | Filter implementation for high-pass and low-pass filters. `"iir"` uses a Butterworth design; `"fir"` uses MNE's FIR design. Does not affect notch filtering. |
| `n_jobs`            | `int`                     | `1`      | Number of parallel jobs for filtering. |
| `dump_path`         | `str` or `None`           | `None`   | If provided, saves the filtered data to this path in FIF format. If the path does not end with `.fif`, the extension is added automatically. |
| `on`                | `list of str` or `None`   | `None`   | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.io.BaseRaw` or `mne.BaseEpochs`
- **Output:** same type as input, filtered in-place on a copy

---

## Notes

- All filters are applied to a **copy** of the data; the original is not modified.
- The filter is applied to EEG, MEG, and ECG channel types if present.
- Notch filtering always uses MNE's FFT method, regardless of the `filter_method` setting.
- Notch frequencies at or above the current Nyquist frequency are silently skipped.
- Resampling is skipped if the data is already at `resample_freq`.

---

## YAML Example

```yaml
preprocess:
  - kind: EEGFilter
    high_freq: 0.1
    low_freq: 40
    notches: [50, 100]
    resample_freq: 250
    hp_order: 4
    lp_order: 8
    filter_method: iir
    n_jobs: 1
```

---

## See Also

- [EEGEpoching](eeg_epoching.md) — typically applied after filtering on continuous data.
- [EEGReference](eeg_reference.md) — referencing, also applied on continuous data.
